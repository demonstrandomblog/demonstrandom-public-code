# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Tensor-backed Lie group elements, algebra coordinates, and group operations.

Rn represents translations; SOn represents rotations using upper-triangle
coordinates of skew matrices. Products compose components; the SO(n)-acting-
on-Rn semidirect product includes rotation/translation coupling. Principal
rotation logarithms are implemented for dimensions two and three, and are
discontinuous at a half-turn. Group multiplication is exposed through *.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
import torch

DEFAULT_EPS = 1e-6
DEFAULT_SKEW_TOLERANCE = 1e-5

class LieGroupElement(ABC):

    def __init__(self, group: 'LieGroup'):
        self.group = group

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        pass

    def __mul__(self, other: 'LieGroupElement') -> 'LieGroupElement':
        if not isinstance(other, LieGroupElement):
            raise TypeError(f"Cannot multiply with {type(other)}")
        return self.group.compose(self, other)

    def __matmul__(self, point: torch.Tensor) -> torch.Tensor:
        return self.group.action(self, point)

    def inverse(self) -> 'LieGroupElement':
        return self.group.inverse(self)

    def log(self) -> torch.Tensor:
        return self.group.log(self)

    @abstractmethod
    def __repr__(self) -> str:
        pass

class TensorElement(LieGroupElement):
    def __init__(self, group: 'LieGroup', data: torch.Tensor):
        super().__init__(group)
        self._data = data

    @property
    def tensor(self) -> torch.Tensor:
        return self._data

    def __repr__(self) -> str:
        return f"Element(type={self.group},shape={self._data.shape})"


class ProductElement(LieGroupElement):

    def __init__(self, group: 'Product', components: Tuple[LieGroupElement, ...]):
        super().__init__(group)
        self.components = components

        if len(components) != len(group.factors):
            raise ValueError(
                f"Expected {len(group.factors)} components, got {len(components)}"
            )

    @property
    def tensor(self) -> torch.Tensor:
        tensors = []
        for elem in self.components:
            t = elem.tensor
            batch_shape = t.shape[:-len(elem.group._element_shape())]
            tensors.append(t.reshape(*batch_shape, -1))
        return torch.cat(tensors, dim=-1)

    def __getitem__(self, index: int) -> LieGroupElement:
        return self.components[index]

    def __repr__(self) -> str:
        return f"Element(type={self.group},components={len(self.components)})"

class LieGroup(ABC):

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def _identity_impl(self, batch_shape: Tuple[int, ...]) -> LieGroupElement:
        pass

    @abstractmethod
    def _compose_impl(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        pass

    @abstractmethod
    def _inverse_impl(self, g: LieGroupElement) -> LieGroupElement:
        pass

    @abstractmethod
    def _exp_impl(self, omega: torch.Tensor) -> LieGroupElement:
        pass

    @abstractmethod
    def _log_impl(self, g: LieGroupElement) -> torch.Tensor:
        pass

    # Public API
    def identity(self, batch_shape: Tuple[int, ...] = ()) -> LieGroupElement:
        return self._identity_impl(batch_shape)

    def compose(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        return self._compose_impl(g, h)

    def inverse(self, g: LieGroupElement) -> LieGroupElement:
        return self._inverse_impl(g)

    def exp(self, omega: torch.Tensor) -> LieGroupElement:
        return self._exp_impl(omega)

    def log(self, g: LieGroupElement) -> torch.Tensor:
        return self._log_impl(g)

    def hat(self, omega_coords: torch.Tensor) -> torch.Tensor:
        # Coordinates to "natural form" of tensor (for readability/debugging)
        # Identity by default
        return omega_coords

    def vee(self, omega_natural: torch.Tensor) -> torch.Tensor:
        # "Natural form" to coordinates of tensor (for readability/debugging)
        # Identity by default
        return omega_natural

    # Optional operations
    def random(self, batch_shape: Tuple[int, ...] = ()) -> LieGroupElement:
        omega = torch.randn(*batch_shape, self.dim)
        return self.exp(omega)

    def action(self, g: LieGroupElement, point: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement action"
        )

    # Derived operations
    def rplus(self, g: LieGroupElement, omega: torch.Tensor) -> LieGroupElement:
        return self.compose(g, self.exp(omega))

    def rminus(self, g: LieGroupElement, h: LieGroupElement) -> torch.Tensor:
        return self.log(self.compose(g.inverse(), h))

    def lplus(self, g: LieGroupElement, omega: torch.Tensor) -> LieGroupElement:
        return self.compose(self.exp(omega), g)

    def lminus(self, g: LieGroupElement, h: LieGroupElement) -> torch.Tensor:
        return self.log(self.compose(h, g.inverse()))

    def adjoint(self, g: LieGroupElement) -> torch.Tensor:
        # Override for analytical formula (much faster)
        batch_shape = g.tensor.shape[:-len(self._element_shape())]
        Ad = torch.zeros(*batch_shape, self.dim, self.dim,
                        dtype=g.tensor.dtype, device=g.tensor.device)

        g_inv = g.inverse()

        for i in range(self.dim):
            omega = torch.zeros(*batch_shape, self.dim,
                              dtype=g.tensor.dtype, device=g.tensor.device)
            omega[..., i] = DEFAULT_EPS

            perturbed = g * self.exp(omega) * g_inv
            Ad[..., :, i] = self.log(perturbed) / DEFAULT_EPS

        return Ad

    # Utilities
    def _element_shape(self) -> Tuple[int, ...]:
        return self.identity().tensor.shape

    def __call__(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        return self.compose(g, h)

    @property
    def is_compact(self) -> bool:
        return False

    @abstractmethod
    def __repr__(self) -> str:
        pass

class SOn(LieGroup):
    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"SO(n) requires n >= 2, got {n}")
        self._n = n

    @property
    def dim(self) -> int:
        return self._n * (self._n - 1) // 2

    def _identity_impl(self, batch_shape: Tuple[int, ...]) -> LieGroupElement:
        I = torch.eye(self._n).expand(*batch_shape, self._n, self._n).clone()
        return TensorElement(self, I)

    def _compose_impl(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        result = torch.matmul(g.tensor, h.tensor)
        return TensorElement(self, result)

    def _inverse_impl(self, g: LieGroupElement) -> LieGroupElement:
        result = g.tensor.transpose(-2, -1)
        return TensorElement(self, result)

    def _exp_impl(self, omega: torch.Tensor) -> LieGroupElement:
        """Exponentiate upper-triangle coordinates of shape (..., dim)."""
        return TensorElement(self, torch.linalg.matrix_exp(self.hat(omega)))

    def exp_matrix(self, omega: torch.Tensor) -> LieGroupElement:
        """Exponentiate an explicitly supplied skew matrix of shape (..., n, n)."""
        if omega.ndim < 2 or omega.shape[-2:] != (self._n, self._n):
            raise ValueError("Expected a square skew matrix matching the group dimension")
        if not torch.allclose(omega, -omega.transpose(-1, -2), atol=DEFAULT_SKEW_TOLERANCE):
            raise ValueError("The Lie algebra matrix must be skew-symmetric")
        return TensorElement(self, torch.linalg.matrix_exp(omega))

    def _log_impl(self, g: LieGroupElement) -> torch.Tensor:
        """Return principal upper-triangle coordinates for SO(2) or SO(3).

        The principal logarithm is discontinuous at a half-turn. Higher-dimensional
        logarithms are explicitly unsupported rather than approximated silently.
        """
        R = g.tensor
        if self._n not in (2, 3):
            raise NotImplementedError("Principal logarithms are implemented for SO(2) and SO(3)")
        if R.shape[-2:] != (self._n, self._n):
            raise ValueError("Rotation matrix shape does not match the group")
        identity = torch.eye(self._n, dtype=R.dtype, device=R.device)
        if not torch.allclose(R.transpose(-1, -2) @ R, identity.expand_as(R), atol=1e-5, rtol=1e-5) or not torch.all(torch.linalg.det(R) > 0):
            raise ValueError("Logarithm requires a proper rotation")
        if self._n == 2:
            # hat(a) has +a above the diagonal, so use R[0, 1].
            return torch.atan2(R[..., 0, 1], R[..., 0, 0]).unsqueeze(-1)

        # Recover a unit quaternion from its largest squared component. Its
        # denominator stays away from zero, including at identity and half-turns.
        r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
        diagonal = torch.stack((1+r00+r11+r22, 1+r00-r11-r22,
                                1-r00+r11-r22, 1-r00-r11+r22), dim=-1)
        wx = R[..., 2, 1] - R[..., 1, 2]
        wy = R[..., 0, 2] - R[..., 2, 0]
        wz = R[..., 1, 0] - R[..., 0, 1]
        xy = R[..., 0, 1] + R[..., 1, 0]
        xz = R[..., 0, 2] + R[..., 2, 0]
        yz = R[..., 1, 2] + R[..., 2, 1]
        numerators = torch.stack((
            torch.stack((diagonal[..., 0], wx, wy, wz), -1),
            torch.stack((wx, diagonal[..., 1], xy, xz), -1),
            torch.stack((wy, xy, diagonal[..., 2], yz), -1),
            torch.stack((wz, xz, yz, diagonal[..., 3]), -1)), dim=-2)
        index = diagonal.argmax(dim=-1, keepdim=True)
        row = numerators.gather(-2, index.unsqueeze(-1).expand(*index.shape, 4)).squeeze(-2)
        denominator = 2 * torch.sqrt(diagonal.gather(-1, index))
        quaternion = row / denominator
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
        real, vector = quaternion[..., :1], quaternion[..., 1:]
        squared = vector.square().sum(dim=-1, keepdim=True)
        length = torch.sqrt(squared.clamp_min(torch.finfo(R.dtype).eps ** 2))
        factor = 2 * torch.atan2(length, real) / length
        factor = torch.where(squared < 1e-8, 2 + squared/3 + 3*squared.square()/20, factor)
        axial = factor * vector
        return torch.stack((-axial[..., 2], axial[..., 1], -axial[..., 0]), dim=-1)

    def random(self, batch_shape: Tuple[int, ...] = ()) -> LieGroupElement:
        """Sample proper rotations by QR decomposition and determinant correction."""
        A = torch.randn(*batch_shape, self._n, self._n)
        Q, R = torch.linalg.qr(A)
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(-2)
        correction = torch.ones_like(signs)
        correction[..., -1] = torch.where(torch.linalg.det(Q) < 0, -1., 1.)
        return TensorElement(self, Q * correction.unsqueeze(-2))

    def action(self, g: LieGroupElement, point: torch.Tensor) -> torch.Tensor:
        return torch.matmul(g.tensor, point.unsqueeze(-1)).squeeze(-1)

    def hat(self, omega: torch.Tensor) -> torch.Tensor:
        """Convert coordinates (..., dim) to skew matrices (..., n, n)."""
        if omega.ndim < 1 or omega.shape[-1] != self.dim:
            raise ValueError(f"Expected coordinates with final dimension {self.dim}")
        skew = omega.new_zeros(*omega.shape[:-1], self._n, self._n)
        i, j = torch.triu_indices(self._n, self._n, offset=1, device=omega.device)
        skew[..., i, j] = omega
        skew[..., j, i] = -omega
        return skew

    def vee(self, omega: torch.Tensor) -> torch.Tensor:
        """Convert skew matrices (..., n, n) to coordinates (..., dim)."""
        if omega.ndim < 2 or omega.shape[-2:] != (self._n, self._n):
            raise ValueError("Expected a matrix matching the group dimension")
        i, j = torch.triu_indices(self._n, self._n, offset=1, device=omega.device)
        return omega[..., i, j]

    @property
    def is_compact(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"SO({self._n})"

class Rn(LieGroup):

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"R^n requires n >= 1, got {n}")
        self._n = n

    @property
    def dim(self) -> int:
        return self._n

    def _identity_impl(self, batch_shape: Tuple[int, ...]) -> LieGroupElement:
        zeros = torch.zeros(*batch_shape, self._n)
        return TensorElement(self, zeros)

    def _compose_impl(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        result = g.tensor + h.tensor
        return TensorElement(self, result)

    def _inverse_impl(self, g: LieGroupElement) -> LieGroupElement:
        return TensorElement(self, -g.tensor)

    def _exp_impl(self, omega: torch.Tensor) -> LieGroupElement:
        return TensorElement(self, omega)

    def _log_impl(self, g: LieGroupElement) -> torch.Tensor:
        return g.tensor

    def action(self, g: LieGroupElement, point: torch.Tensor) -> torch.Tensor:
        # Translation
        return point + g.tensor

    def adjoint(self, g: LieGroupElement) -> torch.Tensor:
        batch_shape = g.tensor.shape[:-1]
        return torch.eye(self._n, dtype=g.tensor.dtype, device=g.tensor.device).expand(
            *batch_shape, self._n, self._n
        ).clone()

    def __repr__(self) -> str:
        return f"R^{self._n}"


class Product(LieGroup):

    def __init__(self, factors: List[LieGroup]):
        if len(factors) < 2:
            raise ValueError("Product requires at least 2 groups")
        self.factors = factors

    @property
    def dim(self) -> int:
        return sum(g.dim for g in self.factors)

    def _identity_impl(self, batch_shape: Tuple[int, ...]) -> LieGroupElement:
        components = tuple(g.identity(batch_shape) for g in self.factors)
        return ProductElement(self, components)

    def _compose_impl(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        h_prod = h if isinstance(h, ProductElement) else self._to_product(h)

        components = tuple(
            g_comp * h_comp
            for g_comp, h_comp in zip(g_prod.components, h_prod.components)
        )
        return ProductElement(self, components)

    def _inverse_impl(self, g: LieGroupElement) -> LieGroupElement:
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        components = tuple(comp.inverse() for comp in g_prod.components)
        return ProductElement(self, components)

    def _exp_impl(self, omega: torch.Tensor) -> LieGroupElement:
        omega_split = self._split_tangent(omega)
        components = tuple(
            self.factors[i].exp(omega_split[i])
            for i in range(len(self.factors))
        )
        return ProductElement(self, components)

    def _log_impl(self, g: LieGroupElement) -> torch.Tensor:
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)

        logs = []
        for comp, factor in zip(g_prod.components, self.factors):
            logs.append(factor.log(comp))

        return torch.cat(logs, dim=-1)

    def random(self, batch_shape: Tuple[int, ...] = ()) -> LieGroupElement:
        components = tuple(g.random(batch_shape) for g in self.factors)
        return ProductElement(self, components)

    def action(self, g: LieGroupElement, point: torch.Tensor) -> torch.Tensor:
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        point_split = self._split_tangent(point)

        results = [
            self.factors[i].action(g_prod.components[i], point_split[i])
            for i in range(len(self.factors))
        ]

        batch_shape = point.shape[:-1]
        flat_results = [r.reshape(*batch_shape, -1) for r in results]
        return torch.cat(flat_results, dim=-1)

    def adjoint(self, g: LieGroupElement) -> torch.Tensor:
        # Block diagonal adjoint
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        batch_shape = g.tensor.shape[:-1]

        Ad = torch.zeros(*batch_shape, self.dim, self.dim,
                        dtype=g.tensor.dtype, device=g.tensor.device)

        offset = 0
        for i, (comp, factor) in enumerate(zip(g_prod.components, self.factors)):
            dim_i = factor.dim
            Ad[..., offset:offset+dim_i, offset:offset+dim_i] = factor.adjoint(comp)
            offset += dim_i

        return Ad

    def _split_tangent(self, omega: torch.Tensor) -> List[torch.Tensor]:
        # Split tangent vector into components
        dims = [g.dim for g in self.factors]
        offsets = [0] + list(torch.cumsum(torch.tensor(dims), dim=0))
        return [omega[..., offsets[i]:offsets[i+1]] for i in range(len(dims))]

    def _to_product(self, g: LieGroupElement) -> ProductElement:
        # Convert generic element to ProductElement if needed
        if isinstance(g, ProductElement):
            return g
        raise TypeError(f"Expected ProductElement, got {type(g)}")

    def __repr__(self) -> str:
        return " × ".join(repr(g) for g in self.factors)


# ============================================================================
# Semidirect Products
# ============================================================================

class SemidirectProduct(LieGroup):

    def __init__(self, normal: LieGroup, actor: LieGroup):
        self.normal = normal
        self.actor = actor
        self.factors = [actor, normal]

    @property
    def dim(self) -> int:
        return self.normal.dim + self.actor.dim

    def _identity_impl(self, batch_shape: Tuple[int, ...]) -> LieGroupElement:
        components = (self.actor.identity(batch_shape), self.normal.identity(batch_shape))
        return ProductElement(self, components)

    def _compose_impl(self, g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
        """Compose the actor and normal components using the actor action."""
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        h_prod = h if isinstance(h, ProductElement) else self._to_product(h)

        g_actor, g_normal = g_prod.components
        h_actor, h_normal = h_prod.components

        result_actor = g_actor * h_actor
        result_normal = g_normal * TensorElement(
            self.normal,
            self.actor.action(g_actor, h_normal.tensor)
        )

        return ProductElement(self, (result_actor, result_normal))

    def _inverse_impl(self, g: LieGroupElement) -> LieGroupElement:
        """Semidirect product inverse"""
        g_prod = g if isinstance(g, ProductElement) else self._to_product(g)
        g_actor, g_normal = g_prod.components

        g_actor_inv = g_actor.inverse()
        g_normal_inv = TensorElement(
            self.normal,
            self.actor.action(g_actor_inv, -g_normal.tensor)
        )

        return ProductElement(self, (g_actor_inv, g_normal_inv))

    def _rigid_motion_dimension(self) -> int:
        if not isinstance(self.normal, Rn) or not isinstance(self.actor, SOn) or self.normal.dim != self.actor._n:
            raise NotImplementedError("Exponential/logarithm support R^n acted on by SO(n)")
        return self.normal.dim

    def _exp_impl(self, omega: torch.Tensor) -> LieGroupElement:
        """Exponentiate a rigid-motion twist, including rotation/translation coupling."""
        n = self._rigid_motion_dimension()
        if omega.ndim < 1 or omega.shape[-1] != self.dim:
            raise ValueError(f"Expected twist coordinates with final dimension {self.dim}")
        generator = omega.new_zeros(*omega.shape[:-1], n+1, n+1)
        generator[..., :n, :n] = self.actor.hat(omega[..., :self.actor.dim])
        generator[..., :n, n] = omega[..., self.actor.dim:]
        transform = torch.linalg.matrix_exp(generator)
        return ProductElement(self, (TensorElement(self.actor, transform[..., :n, :n]),
                                     TensorElement(self.normal, transform[..., :n, n])))

    def _log_impl(self, g: LieGroupElement) -> torch.Tensor:
        """Invert the coupled exponential using the actor's principal logarithm."""
        n = self._rigid_motion_dimension()
        actor, normal = self._to_product(g).components
        rotation = self.actor.log(actor)
        skew = self.actor.hat(rotation)
        # The upper-right block is integral_0^1 exp(s * skew) ds, including at zero.
        block = skew.new_zeros(*skew.shape[:-2], 2*n, 2*n)
        block[..., :n, :n] = skew
        block[..., :n, n:] = torch.eye(n, dtype=skew.dtype, device=skew.device)
        jacobian = torch.linalg.matrix_exp(block)[..., :n, n:]
        translation = torch.linalg.solve(jacobian, normal.tensor.unsqueeze(-1)).squeeze(-1)
        return torch.cat((rotation, translation), dim=-1)

    def random(self, batch_shape: Tuple[int, ...] = ()) -> LieGroupElement:
        components = (self.actor.random(batch_shape), self.normal.random(batch_shape))
        return ProductElement(self, components)

    def _to_product(self, g: LieGroupElement) -> ProductElement:
        if isinstance(g, ProductElement):
            return g
        raise TypeError(f"Expected ProductElement, got {type(g)}")

    def __repr__(self) -> str:
        return f"{self.normal} ⋊ {self.actor}"


def SE(n: int) -> SemidirectProduct:
    return SemidirectProduct(Rn(n), SOn(n))


# ============================================================================
# Examples
# ============================================================================

if __name__ == "__main__":
    print("Clean Lie Group Design with Wrappers")

    # SO(3)
    so3 = SOn(3)
    g = so3.random()
    h = so3.random()

    print(f"g: {g}")
    print(f"h: {h}")

    k = g * h
    print("g * h")
    print(f"k: {k}")
    print(k.tensor)

    g_inv = g.inverse()
    identity_check = g * g_inv
    print(f"g * g^-1 : {identity_check.tensor}")
    print(f"{torch.allclose(identity_check.tensor, torch.eye(3), atol=1e-5)}")

    x = torch.randn(3)
    rotated = g @ x
    print(f"Action (g @ x): {rotated}")
    print(f"Norm preserved: {torch.allclose(torch.norm(x), torch.norm(rotated))}")

    # Extract tensor when needed
    R_matrix = g.tensor
    print(f"Can unwrap to tensor: {R_matrix.shape}")

    # Product Group
    prod = Product([SOn(3), Rn(3)])
    T = prod.random()
    print(f"Product element: {T}")
    print(f"Components as tuple: {len(T.components)} parts")
    print(f"Access rotation: {T[0]}")
    print(f"Access translation: {T[1]}")

    # Composition works naturally - no tensor concatenation!
    T1 = prod.random()
    T2 = prod.random()
    T3 = T1 * T2
    print(f"Product composition: {T3}")

    # SE(3) - Semidirect Product
    se3 = SE(3)
    print(f"Group: {se3}")
    T1 = se3.random()
    T2 = se3.random()

    print(f"T1 components: rotation={T1[0].tensor.shape}, translation={T1[1].tensor.shape}")

    T3 = T1 * T2
    print(f"Semidirect product composition: {T3}")

    # Verify proper SE(3) composition
    R1, t1 = T1[0].tensor, T1[1].tensor
    R2, t2 = T2[0].tensor, T2[1].tensor
    R3, t3 = T3[0].tensor, T3[1].tensor

    expected_R = R1 @ R2
    expected_t = t1 + (R1 @ t2.unsqueeze(-1)).squeeze(-1)

    print(f"Rotation correct: {torch.allclose(R3, expected_R, atol=1e-5)}")
    print(f"Translation correct: {torch.allclose(t3, expected_t, atol=1e-5)}")

    # Derived operations
    omega = torch.randn(6)
    T_perturbed = se3.rplus(T1, omega)
    print(f"Right plus: {T_perturbed}")

    diff = se3.rminus(T1, T2)
    print(f"Right minus: {diff}")
    print(f"Difference in Lie algebra: {diff.shape}")

    # Batching
    batch_g = so3.random(batch_shape=(10,))
    batch_T = se3.random(batch_shape=(5,))
    print(f"SO(3) batch: {batch_g.tensor.shape}")
    print(f"SE(3) batch: {batch_T.tensor.shape}")

    # Gradients
    omega = torch.randn(3, 3, requires_grad=True)
    omega_skew = (omega - omega.T) / 2
    g = so3.exp_matrix(omega_skew)
    loss = g.tensor.sum()
    loss.backward()
    print(f"Gradient computed: {omega.grad is not None}")
