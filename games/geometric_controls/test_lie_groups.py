# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Check Lie group axioms, coordinates, batching, and gradients on CPU.

From the repository root, run:
    python -m pytest -q games/geometric_controls/test_lie_groups.py
"""

import pytest
import torch
from .lie_groups import SOn, Rn, Product, SE

# ============================================================================
# Fixtures
# ============================================================================

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


# ============================================================================
# Test Group Axioms
# ============================================================================

def test_identity_left(so3):
    """e ∘ g = g"""
    g = so3.random()
    e = so3.identity()
    result = e * g
    assert torch.allclose(result.tensor, g.tensor, atol=1e-6)

def test_identity_right(so3):
    """g ∘ e = g"""
    g = so3.random()
    e = so3.identity()
    result = g * e
    assert torch.allclose(result.tensor, g.tensor, atol=1e-6)

def test_inverse_left(so3):
    """g^{-1} ∘ g = e"""
    g = so3.random()
    g_inv = g.inverse()
    e = so3.identity()
    result = g_inv * g
    assert torch.allclose(result.tensor, e.tensor, atol=1e-5)

def test_inverse_right(so3):
    """g ∘ g^{-1} = e"""
    g = so3.random()
    g_inv = g.inverse()
    e = so3.identity()
    result = g * g_inv
    assert torch.allclose(result.tensor, e.tensor, atol=1e-5)

def test_associativity(so3):
    """(g ∘ h) ∘ k = g ∘ (h ∘ k)"""
    g = so3.random()
    h = so3.random()
    k = so3.random()

    left = (g * h) * k
    right = g * (h * k)

    assert torch.allclose(left.tensor, right.tensor, atol=1e-5)

# ============================================================================
# Test Exp/Log
# ============================================================================

def test_exp_log_inverse(so3):
    """log(exp(omega)) = omega for small omega"""
    omega = torch.randn(3, 3) * 0.001
    omega_skew = (omega - omega.T) / 2

    g = so3.exp_matrix(omega_skew)
    omega_recovered = g.log()
    print(so3.vee(omega_skew) - omega_recovered)
    assert torch.allclose(so3.vee(omega_skew), omega_recovered, atol=1e-6)

def test_exp_zero_is_identity(so3):
    """exp(0) = e"""
    zero = torch.zeros(3)
    g = so3.exp(zero)
    e = so3.identity()
    assert torch.allclose(g.tensor, e.tensor, atol=1e-7)

def test_log_identity_is_zero(so3):
    """log(e) = 0 in coordinates"""
    e = so3.identity()
    omega = e.log()
    assert torch.allclose(omega, torch.zeros(3), atol=1e-7)

# ============================================================================
# Test SO(3) Specific Properties
# ============================================================================

def test_so3_orthogonality(so3):
    """R^T R = I for rotation matrices"""
    g = so3.random()
    R = g.tensor
    should_be_identity = R.T @ R
    assert torch.allclose(should_be_identity, torch.eye(3), atol=1e-6)

def test_so3_determinant(so3):
    """Every sampled rotation has determinant +1."""
    determinants = torch.linalg.det(so3.random((128,)).tensor)
    assert torch.allclose(determinants, torch.ones_like(determinants), atol=1e-5)


def test_so3_action_preserves_norm(so3):
    """||R @ x|| = ||x|| for rotation R"""
    g = so3.random()
    x = torch.randn(3)
    x_rotated = g @ x
    assert torch.allclose(torch.norm(x), torch.norm(x_rotated), atol=1e-6)

# ============================================================================
# Test R^n (Abelian Group)
# ============================================================================

def test_rn_commutativity(rn):
    """v1 + v2 = v2 + v1 (abelian)"""
    v1 = rn.random()
    v2 = rn.random()

    result1 = v1 * v2
    result2 = v2 * v1

    assert torch.allclose(result1.tensor, result2.tensor, atol=1e-7)

def test_rn_exp_is_identity_map(rn):
    """exp(v) = v for vector spaces"""
    v = torch.randn(3)
    g = rn.exp(v)
    assert torch.allclose(g.tensor, v, atol=1e-7)

def test_rn_log_is_identity_map(rn):
    """log(v) = v for vector spaces"""
    v = rn.random()
    omega = v.log()
    assert torch.allclose(omega, v.tensor, atol=1e-7)

# ============================================================================
# Test Product Groups
# ============================================================================

def test_product_identity():
    """Product identity is tuple of component identities"""
    so3 = SOn(3)
    r3 = Rn(3)
    prod = Product([so3, r3])

    e = prod.identity()
    assert len(e.components) == 2
    assert torch.allclose(e.components[0].tensor, so3.identity().tensor)
    assert torch.allclose(e.components[1].tensor, r3.identity().tensor)

def test_product_composition():
    """Product composition is componentwise"""
    so3 = SOn(3)
    r3 = Rn(3)
    prod = Product([so3, r3])

    g1 = prod.random()
    g2 = prod.random()
    g3 = g1 * g2

    # Check components compose correctly
    expected_rot = g1.components[0] * g2.components[0]
    expected_trans = g1.components[1] * g2.components[1]

    assert torch.allclose(g3.components[0].tensor, expected_rot.tensor, atol=1e-6)
    assert torch.allclose(g3.components[1].tensor, expected_trans.tensor, atol=1e-6)

def test_product_indexing():
    """Can access product components via indexing"""
    prod = Product([SOn(3), Rn(3)])
    g = prod.random()

    rot = g[0]
    trans = g[1]

    assert rot.tensor.shape == (3, 3)
    assert trans.tensor.shape == (3,)

# ============================================================================
# Test SE(3) - Semidirect Product
# ============================================================================

def test_se3_composition(se3):
    """SE(3) composition includes action: (R1,t1)(R2,t2) = (R1R2, t1 + R1t2)"""
    T1 = se3.random()
    T2 = se3.random()
    T3 = T1 * T2

    R1, t1 = T1[0].tensor, T1[1].tensor
    R2, t2 = T2[0].tensor, T2[1].tensor
    R3, t3 = T3[0].tensor, T3[1].tensor

    # Check rotation
    expected_R = R1 @ R2
    assert torch.allclose(R3, expected_R, atol=1e-5)

    # Check translation
    expected_t = t1 + (R1 @ t2.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(t3, expected_t, atol=1e-5)

def test_se3_identity(se3):
    """SE(3) identity is (I, 0)"""
    e = se3.identity()

    R, t = e[0].tensor, e[1].tensor

    assert torch.allclose(R, torch.eye(3), atol=1e-7)
    assert torch.allclose(t, torch.zeros(3), atol=1e-7)

def test_se3_inverse(se3):
    """SE(3) inverse: (R,t)^{-1} = (R^T, -R^T t)"""
    T = se3.random()
    T_inv = T.inverse()

    R, t = T[0].tensor, T[1].tensor
    R_inv, t_inv = T_inv[0].tensor, T_inv[1].tensor

    # Check rotation inverse
    assert torch.allclose(R_inv, R.T, atol=1e-6)

    # Check translation inverse
    expected_t_inv = -(R.T @ t.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(t_inv, expected_t_inv, atol=1e-6)

def test_se3_is_group(se3):
    """SE(3) satisfies group axioms"""
    T = se3.random()
    T_inv = T.inverse()

    result = T * T_inv

    # Should get identity (within tolerance)
    R, t = result[0].tensor, result[1].tensor
    assert torch.allclose(R, torch.eye(3), atol=1e-5)
    assert torch.allclose(t, torch.zeros(3), atol=1e-5)

# ============================================================================
# Test Perturbation Operators
# ============================================================================

def test_rplus_rminus_inverse(so3):
    g = so3.random()
    omega = torch.randn(3) * 0.001  # coords

    h = so3.rplus(g, omega)
    omega_recovered = so3.rminus(g, h)

    assert torch.allclose(omega, omega_recovered, atol=1e-6)

def test_lplus_lminus_inverse(so3):
    g = so3.random()
    omega = torch.randn(3) * 0.001  # coords

    h = so3.lplus(g, omega)
    omega_recovered = so3.lminus(g, h)

    assert torch.allclose(omega, omega_recovered, atol=1e-6)

# ============================================================================
# Test Batching
# ============================================================================

def test_batch_identity(so3):
    """Identity works with batch dimensions"""
    e_batch = so3.identity(batch_shape=(5,))
    assert e_batch.tensor.shape == (5, 3, 3)

def test_batch_random(so3):
    """Random works with batch dimensions"""
    g_batch = so3.random(batch_shape=(10,))
    assert g_batch.tensor.shape == (10, 3, 3)

def test_batch_compose(so3):
    """Composition works on batches"""
    g = so3.random(batch_shape=(5,))
    h = so3.random(batch_shape=(5,))
    k = g * h
    assert k.tensor.shape == (5, 3, 3)

def test_batch_exp(so3):
    """Exp works on batches"""
    omega_batch = torch.randn(5, 3, 3)
    omega_batch_skew = omega_batch - omega_batch.transpose(-2, -1)
    g_batch = so3.exp_matrix(omega_batch_skew)
    print("----")
    print(omega_batch_skew)
    assert g_batch.tensor.shape == (5, 3, 3)

# ============================================================================
# Test Gradients
# ============================================================================

def test_exp_gradient(so3):
    """Exp is differentiable"""
    omega = torch.randn(3, 3, requires_grad=True)
    omega_skew = (omega - omega.T) / 2

    g = so3.exp_matrix(omega_skew)
    loss = g.tensor.sum()
    loss.backward()

    assert omega.grad is not None
    assert torch.norm(omega.grad) > 1e-7

def test_compose_gradient(so3):
    """Composition is differentiable"""
    omega1 = torch.randn(3, 3, requires_grad=True)
    omega2 = torch.randn(3, 3, requires_grad=True)

    omega1_skew = (omega1 - omega1.T) / 2
    omega2_skew = (omega2 - omega2.T) / 2

    g = so3.exp_matrix(omega1_skew)
    h = so3.exp_matrix(omega2_skew)
    k = g * h

    loss = k.tensor.sum()
    loss.backward()

    assert omega1.grad is not None
    assert omega2.grad is not None

def test_action_gradient(so3):
    """Action is differentiable"""
    omega = torch.randn(3, 3, requires_grad=True)
    x = torch.randn(3, requires_grad=True)

    omega_skew = (omega - omega.T) / 2
    g = so3.exp_matrix(omega_skew)
    y = g @ x

    loss = y.sum()
    loss.backward()

    assert omega.grad is not None
    assert x.grad is not None

# ============================================================================
# Test Tensor Unwrapping
# ============================================================================

def test_tensor_unwrap(so3):
    """Can unwrap to get raw tensor"""
    g = so3.random()
    R = g.tensor

    assert isinstance(R, torch.Tensor)
    assert R.shape == (3, 3)

def test_product_tensor_flatten():
    """Product element can be flattened to tensor"""
    prod = Product([SOn(3), Rn(3)])
    T = prod.random()

    flat = T.tensor
    assert isinstance(flat, torch.Tensor)
    assert flat.shape == (12,)  # 9 for SO(3) + 3 for R^3

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
