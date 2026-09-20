# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
# Companion implementation for Differentiable Game Canonicalization.
# The article targets strict ordinal 2x2 games.
# forward() uses soft ranks and permutations; hard_canonical() and
# class_id() also use discrete operations outside the differentiable path.

from typing import Tuple

import torchsort
import torch
from torch import nn

import hashlib



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

    def hard_canonical(self, payoffs: torch.Tensor):
        """Exact ordinal representative of finite two-player 2x2 games.

        Enumerates the eight action/player relabelings. Equal payoffs retain
        equal dense ranks. The returned float tensor uses one-based ranks;
        class_id returns zero-based integer ranks. Neither path has gradients.

        Permutation metadata: apply Pi_actions on original axes 1 and 2,
        then Pi_players on axis 0. If players exchange, also transpose axes
        1 and 2 so strategy ownership follows the players.
        """
        if self.num_players != 2 or tuple(payoffs.shape) != (2, 2, 2):
            raise ValueError("Exact canonicalization requires shape (2, 2, 2)")
        if payoffs.is_complex() or not bool(torch.isfinite(payoffs).all()):
            raise ValueError("Payoffs must be finite real numbers")
        # Rank comparisons are exact in the input dtype; soft_rank and its
        # regularization are intentionally absent from the exact path.
        ranks = torch.stack([
            torch.unique(player.detach(), sorted=True, return_inverse=True)[1]
            for player in payoffs
        ]).to(torch.int32)
        best_key = None
        for row_swap in (False, True):
            for col_swap in (False, True):
                candidate = ranks
                if row_swap:
                    candidate = candidate.flip(1)
                if col_swap:
                    candidate = candidate.flip(2)
                for player_swap in (False, True):
                    transformed = candidate.flip(0).transpose(1, 2) if player_swap else candidate
                    key = tuple(transformed.reshape(-1).cpu().tolist())
                    if best_key is None or key < best_key:
                        best_key, best = key, transformed
                        swaps = row_swap, col_swap, player_swap
        dtype = payoffs.dtype if payoffs.is_floating_point() else torch.get_default_dtype()
        eye = torch.eye(2, device=payoffs.device, dtype=dtype)
        row_swap, col_swap, player_swap = swaps
        actions = tuple(eye.flip(0) if swap else eye.clone() for swap in (row_swap, col_swap))
        players = eye.flip(0) if player_swap else eye.clone()
        return best.to(dtype) + 1, players, actions

    def class_id(self, payoffs: torch.Tensor):
        """Version-2 ordinal ID; player exchange included, ties preserved.

        Hash a versioned byte sequence of eight ranks in 0..3, independent
        of machine endianness and input floating-point dtype. Old IDs change.
        """
        hard_ord, _, _ = self.hard_canonical(payoffs)
        ranks = (hard_ord - 1).to(torch.int32)
        payload = b"demonstrandom:ordinal-2x2:players-unlabelled:v2\0"
        payload += bytes(ranks.reshape(-1).cpu().tolist())
        digest = hashlib.sha256(payload).hexdigest()
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

    print("\nSoft representation (one-based ranks before soft permutations):")
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
