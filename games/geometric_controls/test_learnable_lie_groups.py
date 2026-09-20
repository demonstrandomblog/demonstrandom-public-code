# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
from typing import Tuple, List, Optional, Union

import pytest

import torch
from torch import nn

from .lie_groups import LieGroup, LieGroupElement, SOn, Rn, SE
from .learnable_lie_groups import LearnableLieElement, LieGroupActionLayer

# WARNING - This file entirely GPT generated. NOT REVIEWED

@pytest.fixture
def so3():
    """SO(3) group"""
    return SOn(3)


@pytest.fixture
def rn():
    """R^3 group"""
    return Rn(3)


@pytest.fixture
def se3():
    """SE(3) group"""
    return SE(3)

def test_learnable_so3_element_backward(so3):
    elem = LearnableLieElement(so3, init="random", init_std=0.01)
    g = elem()  # LieGroupElement
    R = g.tensor
    loss = (R * torch.arange(9, dtype=R.dtype, device=R.device).reshape(3, 3)).sum()
    loss.backward()
    assert elem.omega.grad is not None
    assert torch.norm(elem.omega.grad) > 0

def test_lie_group_action_layer_backward(so3):
    layer = LieGroupActionLayer(so3, init="random", init_std=0.01, bias=True)
    x = torch.randn(16, 3, requires_grad=True)
    y = layer(x)
    loss = (y * torch.arange(y.numel(), dtype=y.dtype, device=y.device).reshape_as(y)).sum()
    loss.backward()
    # Gradients flow to both omega and x
    assert layer.element.omega.grad is not None
    assert torch.norm(layer.element.omega.grad) > 0
    assert x.grad is not None
    assert torch.norm(x.grad) > 0
