# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
# Companion implementation for Differentiable Game Canonicalization.
# The article targets strict ordinal 2x2 games.
# forward() uses soft ranks and permutations; hard_canonical() and
# class_id() also use discrete operations outside the differentiable path.

from dataclasses import dataclass
from typing import List, Tuple, Dict

import torchsort
import torch
from torch import nn

import hashlib
from scipy.optimize import linear_sum_assignment as hungarian



class GameCanonicalizer(nn.Module):

    def __init__(
            self,
            num_players: int,
            tau_players: float = 0.02,
            tau_actions: float = 0.02,
            sinkhorn_iters: int = 30,
            rank_reg: float = 1e-4,
            tiny_tie: Tuple[float, float] = (1e-3, 1e-6)
            ):
        super().__init__()

        self.num_players = num_players

        self.rank_reg = rank_reg
        self.tau_players = tau_players
        self.tau_actions = tau_actions
        self.sinkhorn_iters = sinkhorn_iters
        self.tiny_tie = tiny_tie  # (w_var, w_max)

    def ordinate(self, payoffs: torch.Tensor) -> torch.Tensor:
        original_shape = payoffs.shape
        flattened = payoffs.reshape(self.num_players, -1)
        rankings = torchsort.soft_rank(flattened, regularization_strength=self.rank_reg)
        return rankings.view(original_shape)

    def sinkhorn(self, log_alpha: torch.Tensor, n_iters: int = 30, eps: float = 1e-9) -> torch.Tensor:
        # Work in log-space for stability
        log_P = log_alpha
        for _ in range(n_iters):
            log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True) # rownorm
            log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True) # colnorm
        return torch.exp(log_P).clamp_min(eps)

    def soft_perm_from_scores(self, scores: torch.Tensor, tau: float = 0.05, n_iters: int = 30) -> torch.Tensor:
        n = scores.shape[0]
        # target positions on [0,1] for numeric stability; linear spacing
        positions = torch.linspace(0.0, 1.0, n, device=scores.device, dtype=scores.dtype)
        s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)  # normalize scores to [0,1]
        cost = (s[:, None] - positions[None, :]) ** 2
        log_alpha = -cost / (2 * tau)
        P = self.sinkhorn(log_alpha, n_iters=n_iters)
        return P

    @staticmethod
    def _project_soft_to_perm(P_soft: torch.Tensor) -> torch.Tensor:
        n = P_soft.shape[0]
        idx = torch.arange(n, device=P_soft.device, dtype=P_soft.dtype)
        eps = 1e-11 * (idx[:, None] + 0.73 * idx[None, :])

        cost = (-P_soft + eps).detach().cpu().numpy()
        r, c = hungarian(cost)
        Pi = torch.zeros_like(P_soft)
        Pi[r, c] = 1.0
        return Pi

    @staticmethod
    def _integerize_ordinals(ord_tensor: torch.Tensor) -> torch.Tensor:
        P = ord_tensor.shape[0]
        M = ord_tensor[0].numel()
        out = []
        for p in range(P):
            x = ord_tensor[p].flatten()
            idx = torch.arange(M, device=x.device, dtype=x.dtype)
            x_eps = x + 1e-9 * ((idx * 0.61803398875) % 1.0)
            order = torch.argsort(x_eps, stable=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(M, device=x.device)
            out.append(ranks.view_as(ord_tensor[p]))
        return torch.stack(out, dim=0).to(torch.int32)

    def hard_canonical(self, payoffs: torch.Tensor):
        P = payoffs.shape[0]

        # 1) ordinalize
        ordinal = self.ordinate(payoffs)

        # 2) ACTION perms from ORIGINAL ordinals -> hard -> apply on fixed axes
        Pi_actions = []
        hard = ordinal
        for i in range(P):
            s_actions = self._action_scores(ordinal[i], player_idx=i)
            P_i_soft = self.soft_perm_from_scores(s_actions, tau=self.tau_actions, n_iters=self.sinkhorn_iters)
            Pi_i = self._project_soft_to_perm(P_i_soft.detach())
            hard = self._mode_matmul(hard, Pi_i, axis=1 + i)
            Pi_actions.append(Pi_i)

        # 3) PLAYER perm after actions -> hard -> apply on axis 0
        s_players = self._player_scores(hard)
        P_players_soft = self.soft_perm_from_scores(s_players, tau=self.tau_players, n_iters=self.sinkhorn_iters)
        Pi_players = self._project_soft_to_perm(P_players_soft.detach())
        hard = self._mode_matmul(hard, Pi_players, axis=0)

        # Resolve remaining ties by ordering integer-rank slices.
        # Canonicalize by lexicographically sorting along each axis on the INTEGER ordinals.
        ranks = self._integerize_ordinals(hard)

        order0 = self._axis_lexperm(ranks, axis=0)

        # apply to hard (float) via matrix multiply
        E0_h = torch.eye(P, device=hard.device, dtype=hard.dtype)[order0]
        hard  = self._mode_matmul(hard, E0_h, axis=0)

        # apply to ranks (int) via index permutation
        ranks = self._permute_along_axis(ranks, order0, axis=0)

        # Each action axis (1..P)
        for i in range(P):
            ord_i = self._axis_lexperm(ranks, axis=1 + i)

            # float path
            n_i  = hard.shape[1 + i]
            Ei_h = torch.eye(n_i, device=hard.device, dtype=hard.dtype)[ord_i]
            hard  = self._mode_matmul(hard, Ei_h, axis=1 + i)

            # int path
            ranks = self._permute_along_axis(ranks, ord_i, axis=1 + i)
        return hard, Pi_players, tuple(Pi_actions)

    def class_id(self, payoffs: torch.Tensor):
        hard_ord, _, _ = self.hard_canonical(payoffs)
        ranks = self._integerize_ordinals(hard_ord)
        b = ranks.detach().cpu().numpy().tobytes()
        digest = hashlib.sha256(b).hexdigest()
        return digest[:12], digest, ranks


    @staticmethod
    def _mode_matmul(t: torch.Tensor, P: torch.Tensor, axis: int) -> torch.Tensor:

        # Move axis to front, matmul, then move back.
        perm = list(range(t.ndim))
        perm[axis], perm[0] = perm[0], perm[axis]
        t_perm = t.permute(perm)  # axis now at 0
        n = t_perm.shape[0]
        assert P.shape == (n, n)
        # rows=items, cols=positions. We want to map items -> sorted positions: multiply on the left by P^T
        t_sorted = torch.tensordot(P.T, t_perm, dims=([1], [0]))  # (n, ...) after applying P^T
        # put axis back
        inv = list(range(t.ndim))
        inv[0], inv[axis] = inv[axis], inv[0]
        return t_sorted.permute(inv)

    def _action_scores(self, ordinal: torch.Tensor, player_idx: int) -> torch.Tensor:
        axes = list(range(ordinal.ndim))
        reduce_axes = [a for a in axes if a != player_idx]
        # mean per action
        mean = ordinal.mean(dim=reduce_axes)
        # tie-breakers
        var = ordinal.var(dim=reduce_axes, unbiased=False)
        mx = ordinal.amax(dim=reduce_axes)
        w_var, w_max = self.tiny_tie
        return mean + w_var * var + w_max * mx

    def _player_scores(self, ordinal: torch.Tensor) -> torch.Tensor:
        P = ordinal.shape[0]
        flat = ordinal.reshape(P, -1)
        mean = flat.mean(dim=1)
        var  = flat.var(dim=1, unbiased=False)
        mx   = flat.max(dim=1).values
        w_var, w_max = self.tiny_tie
        return mean + w_var * var + w_max * mx

    def _axis_lexperm(self, ranks: torch.Tensor, axis: int) -> torch.Tensor:
        # Move `axis` to front
        perm = list(range(ranks.ndim))
        perm[axis], perm[0] = perm[0], perm[axis]
        X = ranks.permute(perm)  # shape: (n_axis, ...)

        n = X.shape[0]
        # Flatten each slice (n, prod(other))
        S = X.reshape(n, -1)

        # Encode each rank row as a positional numeric key and sort the keys.
        # This targets the small 2x2 case; larger rows can lose key precision.
        maxv = int(S.max().item()) if S.numel() > 0 else 0
        base = maxv + 1
        # Float64 extends the range but does not guarantee exact keys at arbitrary sizes.
        K = torch.zeros(n, dtype=torch.float64, device=S.device)
        pow_ = 1.0
        for j in range(S.shape[1]-1, -1, -1):
            K += (S[:, j].to(torch.float64)) * pow_
            pow_ *= base

        order = torch.argsort(K, stable=True)
        return order

    @staticmethod
    def _permute_along_axis(t: torch.Tensor, order: torch.Tensor, axis: int) -> torch.Tensor:
        perm = list(range(t.ndim))
        perm[axis], perm[0] = perm[0], perm[axis]
        t0 = t.permute(perm)
        t0 = t0.index_select(0, order.to(t0.device))
        inv = list(range(t.ndim))
        inv[0], inv[axis] = inv[axis], inv[0]
        return t0.permute(inv)

    def forward(self, payoffs: torch.Tensor) -> torch.Tensor:
        P = payoffs.shape[0]
        ordinated_payoffs = self.ordinate(payoffs)

        # --- 1) Per-player ACTION soft perms computed from ORIGINAL ordinals (fixed ownership) ---
        P_actions = []
        for i in range(P):
            s_actions = self._action_scores(ordinated_payoffs[i], player_idx=i)
            P_i = self.soft_perm_from_scores(s_actions, tau=self.tau_actions, n_iters=self.sinkhorn_iters)
            P_actions.append(P_i)

        # Apply action perms on their FIXED global axes (1+i)
        canon = ordinated_payoffs
        for i, P_i in enumerate(P_actions):
            canon = self._mode_matmul(canon, P_i, axis=1 + i)

        # --- 2) PLAYER soft perm computed after action relabeling, then applied on axis 0 ---
        s_players = self._player_scores(canon)
        P_players = self.soft_perm_from_scores(s_players, tau=self.tau_players, n_iters=self.sinkhorn_iters)
        canon = self._mode_matmul(canon, P_players, axis=0)

        return canon, P_players, tuple(P_actions)


if __name__ == "__main__":
    # Create a simple 2x2 game (like Prisoner's Dilemma)
    # Player 1 payoffs: [[3,0], [5,1]]
    # Player 2 payoffs: [[3,5], [0,1]]
    payoffs = torch.tensor([
        [[3.0, 0.0], [5.0, 1.0]],  # Player 1
        [[3.0, 5.0], [0.0, 1.0]]   # Player 2
    ])

    print("Original payoffs:")
    print("Player 1:", payoffs[0])
    print("Player 2:", payoffs[1])

    # Create normalizer
    canonicalizer = GameCanonicalizer(num_players=2)

    # Get ordinal structure
    canon, P_players, P_actions = canonicalizer(payoffs)

    print("\nOrdinal rankings (0=worst, 3=best for 4 outcomes):")
    print("Player 1:", canon[0])
    print("Player 2:", canon[1])

    print(P_players)
    print(P_actions)

    hard_ord, Pi_players, Pi_actions = canonicalizer.hard_canonical(payoffs)
    short_id, full_id, ranks = canonicalizer.class_id(payoffs)

    print("\nHard canonical ordinal:\n", hard_ord)
    print("Pi_players (hard):\n", Pi_players)
    print("Pi_actions (hard):")
    for i, Pi_i in enumerate(Pi_actions):
        print(f"  Player {i}:\n{Pi_i}")

    print("\nCanonical integer ranks per player:\n", ranks)
    print("ClassID (short):", short_id)
    print("ClassID (full): ", full_id)
