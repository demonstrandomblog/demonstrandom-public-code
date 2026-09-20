"""Independent finite-space verification of ordinal 2x2 equivalence."""
from itertools import permutations, product
import pytest
import torch
from .robinson_goforth import GameCanonicalizer


def generators(game):
    # Flat indices for row swap, column swap, and player/strategy exchange.
    for indices in ((2,3,0,1,6,7,4,5), (1,0,3,2,5,4,7,6), (4,6,5,7,0,2,1,3)):
        yield tuple(game[i] for i in indices)


def orbit(seed):
    found, pending = {seed}, [seed]
    while pending:
        for game in generators(pending.pop()):
            if game not in found:
                found.add(game)
                pending.append(game)
    return found


def test_all_576_strict_games_partition_into_78_orbits():
    canon = GameCanonicalizer(2)
    games = {a+b for a,b in product(permutations(range(4)), repeat=2)}
    ids = set()
    count = 0
    while games:
        equivalent = orbit(next(iter(games)))
        games.difference_update(equivalent)
        observed = set()
        for game in equivalent:
            _, identifier, ranks = canon.class_id(torch.tensor(game).reshape(2,2,2))
            observed.add(identifier)
            assert tuple(ranks.flatten().tolist()) == min(equivalent)
        assert len(observed) == 1
        identifier = observed.pop()
        assert identifier not in ids, "Different reference orbits merged"
        ids.add(identifier)
        count += len(equivalent)
    assert count == 576 and len(ids) == 78


def test_known_player_exchange_and_metadata():
    canon = GameCanonicalizer(2)
    source = torch.tensor([[[0.,1.],[2.,3.]], [[0.,1.],[2.,3.]]], dtype=torch.float64)
    assert canon.class_id(source)[1] == canon.class_id(source.flip(0).transpose(1,2))[1]
    # Check returned transforms on every strict ranking pair, including ownership.
    for a,b in product(permutations(range(4)), repeat=2):
        original = torch.tensor(a+b, dtype=torch.float64).reshape(2,2,2) + 1
        hard, players, actions = canon.hard_canonical(original)
        actual = original
        for axis, action in enumerate(actions, 1):
            actual = canon._mode_matmul(actual, action, axis)
        actual = canon._mode_matmul(actual, players, 0)
        if players[0,1] == 1:
            actual = actual.transpose(1,2)
        assert torch.equal(actual, hard)


def test_ties_preserved_and_numeric_scale_independent():
    canon = GameCanonicalizer(2)
    source = torch.tensor([[[1.,1.],[2.,3.]], [[0.,2.],[2.,4.]]], dtype=torch.float64)
    for game in orbit(tuple(source.flatten().tolist())):
        assert canon.class_id(torch.tensor(game).reshape(2,2,2))[1] == canon.class_id(source)[1]
    _, identifier, ranks = canon.class_id(source)
    assert torch.unique(ranks[0]).numel() == 3 and torch.unique(ranks[1]).numel() == 3
    assert identifier == canon.class_id(source.float())[1]
    # A soft-rank regularization threshold must not collapse distinct payoffs.
    assert identifier == canon.class_id(source * 1e-12)[1]
    assert identifier == canon.class_id(source**3)[1]


@pytest.mark.parametrize("source", [torch.zeros(2,3,3), torch.full((2,2,2), float('nan')), torch.full((2,2,2), float('inf')), torch.ones(2,2,2,dtype=torch.complex64)])
def test_unsupported_inputs(source):
    with pytest.raises(ValueError):
        GameCanonicalizer(2).class_id(source)


def test_hard_has_no_gradient_and_soft_is_separate():
    source = torch.tensor([[[0.,1.],[2.,3.]], [[1.,0.],[3.,2.]]], requires_grad=True)
    canon = GameCanonicalizer(2)
    assert not canon.hard_canonical(source)[0].requires_grad
    soft, _, _ = canon(source)
    soft.square().sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
