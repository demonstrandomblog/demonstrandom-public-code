# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI assistance/review status: see AI_NOTICE.md at the repository root.

import math
import pytest
import torch
from .core import Arena, EulerIntegrator, RK4Integrator, NormalFormConverter, build_stag_hunt

@pytest.mark.parametrize('integrator,ratio',[(EulerIntegrator,1.8),(RK4Integrator,14)])
def test_decay_refinement(integrator,ratio):
    class Decay:
        def derivative(self,state,controls):
            return -state
    errors = []
    for h in [.1,.05]:
        state = torch.ones(1,dtype=torch.float64)
        for _ in range(round(1/h)):
            state = integrator().step(Decay(),state,{},h)
        errors.append(abs(state.item()-math.exp(-1)))
    assert errors[1] < errors[0]/ratio

def test_article_stag_hunt_matrix():
    game,_ = build_stag_hunt()
    matrix = NormalFormConverter.to_payoff_matrix(Arena(game,dt=.02,max_time=15),['c1','c2'])
    torch.testing.assert_close(matrix,torch.tensor([[[4.,0.],[3.,3.]],[[4.,3.],[0.,3.]]]))

@pytest.mark.parametrize('dt',[0,-1,float('nan'),float('inf')])
def test_invalid_step_rejected(dt):
    game,_ = build_stag_hunt()
    with pytest.raises(ValueError):
        Arena(game,dt=dt)

@pytest.mark.parametrize('until',[float('nan'),float('inf'),-1.])
def test_invalid_end_time_rejected(until):
    game,_ = build_stag_hunt()
    with pytest.raises(ValueError):
        Arena(game).simulate({},until=until)

def test_fractional_end_time_retains_fixed_step_convention():
    game,_ = build_stag_hunt()
    arena = Arena(game,dt=.1)
    policies = {name:a.get_policy(a.strategy_names[0]) for name,a in game.agents.items()}
    trajectory = arena.simulate(policies,until=.25)
    assert trajectory[-1].time == pytest.approx(.3)
    assert len(trajectory) == 4
