# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI assistance/review status: see AI_NOTICE.md at the repository root.

import itertools
import pytest
import torch
from .core import Agent, Arena, Game, LogitsPolicy

@pytest.fixture
def arena():
    actions = [['a','b'], ['a','b','c'], ['a','b']]
    payoffs = torch.arange(36, dtype=torch.float64).reshape(3,2,3,2) / 7
    agents = [Agent(LogitsPolicy(), str(i)) for i in range(3)]
    result = Arena(Game(payoffs, actions), agents)
    for agent, logits in zip(agents, [[.3,-.2],[-.4,.6,.1],[.2,.7]]):
        agent.policy.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float64))
    return result

def test_expected_payoffs_by_enumeration(arena):
    distributions = [a.policy(actions) for a, actions in zip(arena.agents, arena.game.actions)]
    expected = torch.zeros(3, dtype=torch.float64)
    for profile in itertools.product(*[range(len(a)) for a in arena.game.actions]):
        probability = torch.stack([p[i] for p,i in zip(distributions, profile)]).prod()
        expected += probability * arena.game.payoff(profile)
    torch.testing.assert_close(arena.expected_payoffs(), expected)

def test_own_payoff_gradients_by_finite_difference(arena):
    for player, agent in enumerate(arena.agents):
        logits = agent.policy.logits
        analytic = torch.autograd.grad(arena.expected_payoffs()[player], logits)[0]
        for j in range(len(logits)):
            with torch.no_grad():
                old = logits[j].item()
                logits[j] = old + 1e-6
                plus = arena.expected_payoffs()[player].item()
                logits[j] = old - 1e-6
                minus = arena.expected_payoffs()[player].item()
                logits[j] = old
            assert analytic[j].item() == pytest.approx((plus-minus)/2e-6, abs=2e-8)

def test_article_expected_payoffs():
    game = Game(torch.tensor([[[8.,0.],[2.,3.]], [[3.,1.],[0.,2.]]]), [['Stag','Hare']]*2)
    arena = Arena(game, [Agent(lambda _: torch.tensor([1.,0.]), 'Alice'),
                         Agent(lambda _: torch.tensor([.5,.5]), 'Bob')])
    torch.testing.assert_close(arena.expected_payoffs(), torch.tensor([4.,2.]))

def test_lazy_initialization_keeps_optimizer_parameters(arena):
    before = [a.policy.logits for a in arena.agents]
    arena.expected_payoffs()
    assert all(a.policy.logits is p for a,p in zip(arena.agents,before))
