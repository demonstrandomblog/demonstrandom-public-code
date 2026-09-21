# AI-assisted research code; no blanket human or mathematical review is claimed.
import torch
import pytest

from .robinson_goforth import GameCanonicalizer


def affine(payoffs: torch.Tensor, a0=1.0, b0=0.0, a1=1.0, b1=0.0) -> torch.Tensor:
    P = payoffs.clone().float()
    P[0] = a0 * P[0] + b0
    P[1] = a1 * P[1] + b1
    return P

def permute_actions(payoffs: torch.Tensor, perm0=(0,1), perm1=(0,1)) -> torch.Tensor:
    P = payoffs.clone()
    P = P[:, perm0, :]
    P = P[:, :, perm1]
    return P

def swap_players(payoffs: torch.Tensor) -> torch.Tensor:
    P0 = payoffs[0].transpose(0,1)
    P1 = payoffs[1].transpose(0,1)
    return torch.stack([P1, P0], dim=0)

def class_id(canon, U):
    short, full, _ = canon.class_id(U)
    return full  # use full hash for tests

# --- Base games ---------------------------------------------------------------
def prisoners_dilemma():
    return torch.tensor([
        [[3.0, 0.0],[5.0, 1.0]],
        [[3.0, 5.0],[0.0, 1.0]],
    ])

def stag_hunt():
    return torch.tensor([
        [[4.0, 0.0],[3.0, 3.0]],
        [[4.0, 3.0],[0.0, 3.0]],
    ])

def battle_of_sexes():
    return torch.tensor([
        [[2.0, 0.0],[0.0, 1.0]],
        [[1.0, 0.0],[0.0, 2.0]],
    ])

def matching_pennies():
    P1 = torch.tensor([[1.0, -1.0],[-1.0, 1.0]])
    P2 = -P1
    return torch.stack([P1, P2], dim=0)

def hawk_dove():
    return torch.tensor([
        [[0.0, 3.0],[1.0, 2.0]],
        [[0.0, 1.0],[3.0, 2.0]],
    ])

# --- Families of equivalent variants -----------------------------------------

def pd_variants():
    U = prisoners_dilemma()
    return [
        U,
        affine(U, a0=2.0, b0=5.0, a1=0.5, b1=-1.0),
        permute_actions(U, perm0=(1,0), perm1=(1,0)),
        swap_players(U),
        permute_actions(affine(U, a0=3.0, b0=7.0, a1=1.7, b1=2.0), (1,0), (0,1)),
    ]

def sh_variants():
    U = stag_hunt()
    return [
        U,
        affine(U, a0=4.0, b0=10.0, a1=2.0, b1=-3.0),
        permute_actions(U, perm0=(1,0), perm1=(1,0)),
        swap_players(U),
        permute_actions(affine(U, a0=0.7, b0=0.0, a1=5.0, b1=1.0), (0,1), (1,0)),
    ]

def bos_variants():
    U = battle_of_sexes()
    return [
        U,
        permute_actions(U, perm0=(1,0), perm1=(1,0)),
        swap_players(U),
        affine(U, a0=2.0, b0=1.0, a1=3.0, b1=-2.0),
        permute_actions(affine(U, 1.0, 0.0, 4.0, 10.0), (1,0), (0,1)),
    ]

def mp_variants():
    U = matching_pennies()
    return [
        U,
        permute_actions(U, perm0=(1,0), perm1=(1,0)),
        swap_players(U),
        affine(U, a0=5.0, b0=3.0, a1=2.0, b1=-7.0),
        permute_actions(affine(U, 2.0, 9.0, 1.0, -4.0), (1,0), (0,1)),
    ]

def hd_variants():
    U = hawk_dove()
    return [
        U,
        permute_actions(U, perm0=(1,0), perm1=(1,0)),
        swap_players(U),
        affine(U, a0=2.0, b0=0.0, a1=3.0, b1=1.0),
        permute_actions(affine(U, 0.5, 2.0, 1.5, -3.0), (0,1), (1,0)),
    ]

# --- Tests --------------------------------------------------------------------

@pytest.fixture(name="canonicalizer")
def _canonicalizer_fixture():
    # Isolate repeatable example randomness from other tests in the process.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield GameCanonicalizer(num_players=2)


def test_family_equivalence(canonicalizer):
    families = [
        ("PrisonersDilemma", pd_variants()),
        ("StagHunt",         sh_variants()),
        ("BattleOfSexes",    bos_variants()),
        ("MatchingPennies",  mp_variants()),
        ("HawkDove",         hd_variants()),
    ]

    for name, variants in families:
        ids = [class_id(canonicalizer, U) for U in variants]
        assert len(set(ids)) == 1, f"{name}: expected all variants equivalent, got IDs: {ids}"
        print(f"[OK] {name}. ID {ids[0]} (x{len(variants)})")

def test_negative_controls(canonicalizer):
    reps = [
        prisoners_dilemma(),
        stag_hunt(),
        battle_of_sexes(),
        matching_pennies(),
        hawk_dove(),
    ]
    ids = [class_id(canonicalizer, U) for U in reps]
    assert len(set(ids)) == len(ids), f"Negative control failed: collisions among base games: {ids}"
    print("[OK] Negative controls. Distinct IDs: ", ids)

def test_stability_small_noise(canonicalizer):
    U = stag_hunt()
    base = class_id(canonicalizer, U)
    eps = 1e-9
    U_noisy = U + eps * torch.randn_like(U)
    pert = class_id(canonicalizer, U_noisy)
    assert base == pert, "Tiny noise changed ID; consider epsilon tie-handling."
    print("[OK] Stability to tiny noise.")

# --- Run all ------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=6, linewidth=120, sci_mode=False)
    canonicalizer = GameCanonicalizer(num_players=2)

    test_family_equivalence(canonicalizer)
    test_negative_controls(canonicalizer)
    test_stability_small_noise(canonicalizer)
