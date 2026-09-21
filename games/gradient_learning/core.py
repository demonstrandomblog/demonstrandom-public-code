# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI assistance/review status: see AI_NOTICE.md at the repository root.

from abc import ABC, abstractmethod

from typing import List
import functools


import torch
import torch.nn as nn

def _run_once(method):
    attr_flag = f"__{method.__name__}_has_run"

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, attr_flag, False):
            return
        setattr(self, attr_flag, True)
        return method(self, *args, **kwargs)

    return wrapper


class Policy(ABC):
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # If the subclass overrides 'initialize', wrap it exactly once.
        if "initialize" in cls.__dict__:
            cls.initialize = _run_once(cls.__dict__["initialize"])

    @abstractmethod
    def initialize(self, actions: List[str]) -> None:
        pass

    @abstractmethod
    def forward(self, actions: List[str]) -> torch.Tensor:
        pass

    def __call__(self, actions: List[str]) -> torch.Tensor:
        if hasattr(self, "initialize"):
            self.initialize(actions)
        return super().__call__(actions)

class LogitsPolicy(Policy, nn.Module):
    def __init__(self, initialization='uniform'):
        Policy.__init__(self)
        nn.Module.__init__(self)
        self.initialization = initialization
        self.logits = None

    def initialize(self, actions: List[str]) -> None:
        num_actions = len(actions)
        if self.initialization == 'uniform':
            self.logits = nn.Parameter(torch.zeros(num_actions))
        elif self.initialization == 'random':
            self.logits = nn.Parameter(torch.rand(num_actions))

    def forward(self, actions: List[str]) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

class Agent:
    def __init__(self, policy, name: str):
        self.name = name
        self.policy = policy

    def act(self, actions: List[str]):
        probs = self.policy(actions)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        return action_idx, actions[action_idx]

class Game:
    def __init__(self, payoffs: torch.Tensor | nn.Parameter, actions: List[List[str]], name=None):
        self.num_players = payoffs.shape[0]
        self.payoffs = payoffs
        self.actions = actions
        self.name = name or "Unnamed Game"
        self._size = payoffs.shape

    @property
    def size(self):
        return self._size

    def payoff(self, action_indices):
        return self.payoffs[(slice(None),) + tuple(action_indices)]

    def to(self, device):
        return Game(self.payoffs.to(device), self.actions, self.name)

    def clone(self):
        if isinstance(self.payoffs, nn.Parameter):
            req = bool(self.payoffs.requires_grad)
            return Game(nn.Parameter(self.payoffs.detach().clone(), requires_grad=req), self.actions, self.name)
        else:
            return Game(self.payoffs.detach().clone(), self.actions, self.name)

    def __repr__(self):
        learnable = isinstance(self.payoffs, nn.Parameter) and self.payoffs.requires_grad
        return f'<Game "{self.name}" size={self._size} learnable={learnable}>'


class Arena:

    def __init__(self, game: Game, agents: List[Agent]):
        assert len(agents) == game.num_players
        self.game = game
        self.agents = agents

        for i, agent in enumerate(self.agents):
            if hasattr(agent.policy, 'initialize'):
                agent.policy.initialize(self.game.actions[i])

    def play(self):
        action_indices = []
        actions_chosen = []

        for player_idx, agent in enumerate(self.agents):
            actions = self.game.actions[player_idx]
            action_idx, action = agent.act(actions)
            action_indices.append(action_idx)
            actions_chosen.append(action)

        payoffs = self.game.payoff(action_indices)
        return actions_chosen, payoffs

    def expected_payoffs(self):
        dists = [agent.policy(self.game.actions[i]) for i, agent in enumerate(self.agents)]
        joint_dist = dists[0]
        for dist in dists[1:]:
            joint_dist = torch.einsum('i,j->ij', joint_dist.flatten(), dist).flatten()

        payoffs_flat = self.game.payoffs.view(self.game.num_players, -1)
        exp_payoffs = (joint_dist * payoffs_flat).sum(-1)
        return exp_payoffs
