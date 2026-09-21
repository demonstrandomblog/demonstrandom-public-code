# AI-assisted research code; no blanket human or mathematical review is claimed.
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .lie_groups import LieGroup, LieGroupElement

# WARNING - This file entirely GPT generated. NOT REVIEWED
# Learn an element of a fixed Lie group through its Lie algebra coordinates.

class LearnableLieElement(nn.Module):
    """
    Learnable Lie group element parameterized in the Lie algebra.

    Internally stores a tensor of shape (*batch_shape, group.dim)
    and uses group.exp to obtain the actual group element.

    Example
    -------
    so3 = SOn(3)
    elem = LearnableLieElement(so3)  # learnable SO(3) element
    g = elem.group_element()         # LieGroupElement
    R = g.tensor                     # rotation matrix
    """

    def __init__(
        self,
        group: LieGroup,
        batch_shape: Tuple[int, ...] = (),
        init: str = "identity",
        init_std: float = 1e-2,
        like: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.group = group
        self.batch_shape = tuple(batch_shape)

        dim = group.dim

        if like is not None:
            device = like.device
            dtype = like.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        if init == "identity":
            data = torch.zeros(*batch_shape, dim, device=device, dtype=dtype)
        elif init == "random":
            data = torch.randn(*batch_shape, dim, device=device, dtype=dtype) * init_std
        else:
            raise ValueError(f"Unknown init '{init}', expected 'identity' or 'random'")

        self.omega = nn.Parameter(data)

    def group_element(self) -> LieGroupElement:
        """
        Map the learnable Lie algebra parameter to a LieGroupElement via exp.
        """
        return self.group.exp(self.omega)

    def tensor(self) -> torch.Tensor:
        """
        Convenience: directly get the underlying group tensor, e.g. rotation matrix.
        """
        return self.group_element().tensor

    def forward(self) -> LieGroupElement:
        """
        Forward returns the LieGroupElement, so you can write:

            g = elem()
            y = g @ x

        or

            k = g * h
        """
        return self.group_element()

    def __repr__(self) -> str:
        return (
            f"LearnableLieElement(group={self.group}, "
            f"batch_shape={self.batch_shape}, dim={self.group.dim})"
        )


class LieGroupActionLayer(nn.Module):
    """
    Layer that applies a learnable Lie group action to input points.

    For groups that implement `action`, e.g.:
      - SOn(n): rotation of vectors
      - Rn(n): translation of vectors

    The group must implement `action`; SE(n) does not currently provide it.

    Inputs:
      x: Tensor of shape (*batch, point_dim) or matching group.action conventions.

    Example
    -------
    so3 = SOn(3)
    layer = LieGroupActionLayer(so3, batch_shape=(), init="identity")
    x = torch.randn(32, 3)
    y = layer(x)  # rotated vectors
    """

    def __init__(
        self,
        group: LieGroup,
        batch_shape: Tuple[int, ...] = (),
        init: str = "identity",
        init_std: float = 1e-2,
        bias: bool = False,
        like: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.group = group
        self.element = LearnableLieElement(
            group,
            batch_shape=batch_shape,
            init=init,
            init_std=init_std,
            like=like,
        )

        self.use_bias = bias
        if bias:
            if like is not None:
                device = like.device
                dtype = like.dtype
            else:
                device = torch.device("cpu")
                dtype = torch.get_default_dtype()
            # Bias dimension is not known a priori; will lazily init.
            self._bias = nn.Parameter(torch.zeros(0, device=device, dtype=dtype))
            self._bias_initialized = False
        else:
            self.register_parameter("_bias", None)
            self._bias_initialized = False

    def _maybe_init_bias(self, x: torch.Tensor):
        if not self.use_bias or self._bias_initialized:
            return

        # Assume bias acts on the last dimension of x
        out_dim = x.shape[-1]
        device = x.device
        dtype = x.dtype
        new_bias = torch.zeros(out_dim, device=device, dtype=dtype)
        self._bias = nn.Parameter(new_bias)
        self._bias = nn.Parameter(self._bias.to(device=device, dtype=dtype))
        self._bias_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the learnable group action (and optional bias) to x.
        """
        g = self.element()  # LieGroupElement
        y = self.group.action(g, x)

        if self.use_bias:
            self._maybe_init_bias(x)
            y = y + self._bias

        return y

    def __repr__(self) -> str:
        return (
            f"LieGroupActionLayer(group={self.group}, "
            f"learnable_element={self.element}, "
            f"bias={self.use_bias})"
        )
