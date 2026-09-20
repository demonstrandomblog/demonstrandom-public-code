# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
import math
import pytest
import torch
from .lie_groups import SOn, Rn, SE, SemidirectProduct, TensorElement
from .learnable_lie_groups import LearnableLieElement


@pytest.mark.parametrize("n", [2, 3, 4])
def test_sampler_returns_proper_rotations(n):
    torch.manual_seed(123)
    group = SOn(n)
    matrices = group.random((2, 64)).tensor
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(2, 64), atol=2e-6)
    assert torch.allclose(matrices.transpose(-1, -2) @ matrices, torch.eye(n).expand_as(matrices), atol=2e-6)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("angle", [0., 1e-9, .8, math.pi-1e-6, math.pi, math.pi+1e-4, 4.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_principal_log_reconstructs_rotations(n, angle, dtype):
    group = SOn(n)
    axis = torch.arange(1, group.dim+1, dtype=dtype)
    coordinates = axis / torch.linalg.vector_norm(axis) * angle
    rotation = group.exp(coordinates)
    recovered = group.log(rotation)
    assert recovered.shape == coordinates.shape
    assert torch.allclose(group.exp(recovered).tensor, rotation.tensor, atol=4e-6, rtol=4e-6)
    if angle < math.pi-1e-5:
        assert torch.allclose(recovered, coordinates, atol=4e-6, rtol=4e-6)


def test_three_coordinate_vectors_are_a_batch_not_a_skew_matrix():
    group = SOn(3)
    coordinates = torch.tensor([[.2, .3, .4], [.6, .2, -.1], [0., 0., 0.]], dtype=torch.float64)
    matrices = group.hat(coordinates)
    assert matrices.shape == (3, 3, 3)
    assert torch.equal(group.vee(matrices), coordinates)
    assert torch.allclose(group.exp(coordinates).tensor, group.exp_matrix(matrices).tensor)
    assert torch.allclose(group.log(group.exp(coordinates)), coordinates, atol=1e-12)


def test_matrix_entry_point_rejects_non_skew_input():
    with pytest.raises(ValueError, match="skew"):
        SOn(3).exp_matrix(torch.eye(3))


def test_log_rejects_reflections_and_unsupported_dimension():
    so3 = SOn(3)
    with pytest.raises(ValueError, match="proper rotation"):
        so3.log(TensorElement(so3, torch.diag(torch.tensor([1., 1., -1.]))))
    with pytest.raises(NotImplementedError, match=r"SO\(2\).*SO\(3\)"):
        SOn(4).log(SOn(4).identity())


@pytest.mark.parametrize("n", [2, 3, 4])
def test_semidirect_exponential_is_a_one_parameter_subgroup(n):
    group = SE(n)
    coordinates = torch.linspace(-.8, 1., group.dim, dtype=torch.float64).repeat(3, 1)
    coordinates[1] *= .2
    coordinates[2] *= 0
    whole = group.exp(coordinates)
    composed = group.exp(.3 * coordinates) * group.exp(.7 * coordinates)
    assert torch.allclose(whole.tensor, composed.tensor, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_rigid_motion_matches_analytic_planar_translation(n):
    group = SE(n)
    angle = 1.2
    twist = torch.zeros(group.dim, dtype=torch.float64)
    twist[0] = angle
    twist[group.actor.dim] = 1
    result = group.exp(twist)
    expected = torch.zeros(n, dtype=torch.float64)
    expected[0] = math.sin(angle)/angle
    expected[1] = -(1-math.cos(angle))/angle
    assert torch.allclose(result[1].tensor, expected, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_semidirect_log_roundtrip_and_pure_translation(n):
    group = SE(n)
    twist = torch.linspace(-.5, .7, group.dim, dtype=torch.float64).repeat(4, 1)
    twist[0, :group.actor.dim] = 0
    twist[1] = 0
    element = group.exp(twist)
    assert torch.allclose(element[1].tensor[0], twist[0, group.actor.dim:], atol=1e-12)
    assert torch.allclose(group.log(element), twist, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("factory", [lambda: SOn(2), lambda: SOn(3), lambda: SE(2), lambda: SE(3)])
@pytest.mark.parametrize("at_identity", [False, True])
def test_log_exp_gradients(factory, at_identity):
    group = factory()
    x = torch.zeros(group.dim, dtype=torch.float64) if at_identity else torch.linspace(.1, .4, group.dim, dtype=torch.float64)
    x.requires_grad_()
    assert torch.autograd.gradcheck(lambda v: group.log(group.exp(v)), (x,), atol=2e-5, rtol=2e-4)


def test_unsupported_semidirect_exp_is_explicit():
    group = SemidirectProduct(Rn(2), Rn(2))
    with pytest.raises(NotImplementedError, match="SO"):
        group.exp(torch.zeros(4))


def test_learnable_batch_of_three_keeps_coordinate_shape():
    layer = LearnableLieElement(SOn(3), batch_shape=(3,), init="random")
    rotation = layer().tensor
    assert rotation.shape == (3, 3, 3)
    weights = torch.arange(27, dtype=rotation.dtype).reshape(3, 3, 3)
    (rotation * weights).sum().backward()
    assert torch.isfinite(layer.omega.grad).all()
    assert layer.omega.grad.abs().sum() > 0
