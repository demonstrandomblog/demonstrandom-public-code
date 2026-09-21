# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Thirteen explicit three-player, three-strategy games and their contrast diagnostics.
Builders return u[p,s1,s2,s3] with shape (3,3,3,3). Per-player constants are
removed before computing seven family Gram matrices and selected cubic values.
The command writes atlas_3x3_results.json and atlas_3x3_table.md in the working
directory. These examples and diagnostics are not a complete game classification."""

import json
from itertools import product as iproduct
import numpy as np

from .contrast_blocks_3x3 import all_family_matrices, mean_zero_payoff
from .generators_3x3_strategy_only import diagnostic_degree3


# ---------------------------------------------------------------------------
# Named (3,3) games
# ---------------------------------------------------------------------------


def _zeros():
    return np.zeros((3, 3, 3, 3), dtype=float)


def rps_3player():
    u = _zeros()
    for p in range(3):
        for s1 in range(3):
            for s2 in range(3):
                for s3 in range(3):
                    profile = (s1, s2, s3)
                    my_s = profile[p]
                    payoff = 0
                    for q in range(3):
                        if q == p:
                            continue
                        opp_s = profile[q]
                        diff = (my_s - opp_s) % 3
                        if diff == 1:
                            payoff += 1  # my_s beats opp_s
                        elif diff == 2:
                            payoff -= 1  # opp_s beats my_s
                    u[p, s1, s2, s3] = payoff
    return u


def stag_hunt_3player():
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        my_s = profile[p]
        if my_s == 1:
            u[p, s1, s2, s3] = 2.0
        elif my_s == 0:
            u[p, s1, s2, s3] = 4.0 if profile == (0, 0, 0) else 0.0
        else:  # my_s == 2
            u[p, s1, s2, s3] = 5.0 if profile == (2, 2, 2) else 0.0
    return u


def public_goods_3player():
    r = 1.5  # return multiplier
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        total = sum(profile)
        my_cost = profile[p]
        u[p, s1, s2, s3] = -my_cost + r * total / 3.0
    return u


def pure_coordination_3player():
    u = _zeros()
    for s in range(3):
        for p in range(3):
            u[p, s, s, s] = 1.0
    return u


def volunteers_dilemma_3player():
    c, b = 1.0, 3.0
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        my_s = profile[p]
        someone_volunteered = any(profile[q] == 0 for q in range(3))
        payoff = (b if someone_volunteered else 0.0)
        if my_s == 0:
            payoff -= c
        u[p, s1, s2, s3] = payoff
    return u


def battle_of_sexes_3player():
    u = _zeros()
    for s in range(3):
        for p in range(3):
            u[p, s, s, s] = 2.0 if p == s else 1.0
    return u


def commons_3player():
    cap = 2
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        total = s1 + s2 + s3
        if total <= cap:
            u[p, s1, s2, s3] = float((s1, s2, s3)[p])
        else:
            u[p, s1, s2, s3] = 0.0
    return u


def majority_3player():
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        counts = [profile.count(s) for s in range(3)]
        my_count = counts[profile[p]]
        max_count = max(counts)
        if counts.count(max_count) > 1:
            # tied: e.g., all distinct (1,1,1) or pair (2,1,0)... actually
            # if one strategy has 2 and another 1, max is unique. Tied only
            # when all 3 strategies appear once each (counts = (1,1,1)).
            u[p, s1, s2, s3] = 0.0
        else:
            u[p, s1, s2, s3] = 1.0 if my_count == max_count else -1.0
    return u


def symmetric_anticoord_3player():
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        distinct = len(set(profile))
        if distinct == 3:
            u[p, s1, s2, s3] = 1.0
        elif distinct == 1:
            u[p, s1, s2, s3] = -1.0
        else:
            u[p, s1, s2, s3] = 0.0
    return u


def dictator_3player():
    base = np.array([
        [3, 1, 0],   # if player 0 plays 0: payoffs (3, 1, 0)
        [0, 3, 1],   # if player 0 plays 1: payoffs (0, 3, 1)
        [1, 0, 3],   # if player 0 plays 2: payoffs (1, 0, 3)
    ], dtype=float)
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        u[p, s1, s2, s3] = base[s1, p]  # s1 = player 0's strategy
    return u


def chicken_3player():
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        my_s = profile[p]
        stayers = [q for q in range(3) if profile[q] != 0]
        n_stay = len(stayers)
        if my_s == 0:  # swerve
            u[p, s1, s2, s3] = 0.0 if n_stay > 0 else 1.0
        else:  # stay
            if n_stay == 1:
                u[p, s1, s2, s3] = 4.0  # sole stayer wins
            else:
                u[p, s1, s2, s3] = -8.0  # multiple stayers collide
    return u


def matching_pennies_3player():
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        profile = (s1, s2, s3)
        if p == 0:
            u[p, s1, s2, s3] = 1.0 if profile[0] == profile[1] else -1.0
        elif p == 1:
            u[p, s1, s2, s3] = 1.0 if profile[1] == profile[2] else -1.0
        else:  # p == 2
            u[p, s1, s2, s3] = -1.0 if profile[2] == profile[0] else 1.0
    return u


def common_interest_3player():
    Phi = np.zeros((3, 3, 3))
    for s1, s2, s3 in iproduct(range(3), repeat=3):
        # Reward distinct-strategy profiles more
        distinct = len({s1, s2, s3})
        Phi[s1, s2, s3] = float(distinct + (s1 + s2 + s3) * 0.5)
    u = _zeros()
    for p, s1, s2, s3 in iproduct(range(3), repeat=4):
        u[p, s1, s2, s3] = Phi[s1, s2, s3]
    return u


# Registry of named games
NAMED_GAMES = {
    "3p_Rock_Paper_Scissors": rps_3player,
    "3p_Stag_Hunt": stag_hunt_3player,
    "3p_Public_Goods": public_goods_3player,
    "3p_Pure_Coordination": pure_coordination_3player,
    "3p_Volunteers_Dilemma": volunteers_dilemma_3player,
    "3p_Battle_of_Sexes": battle_of_sexes_3player,
    "3p_Tragedy_of_Commons": commons_3player,
    "3p_Majority": majority_3player,
    "3p_Symmetric_AntiCoord": symmetric_anticoord_3player,
    "3p_Asymmetric_Dictator": dictator_3player,
    "3p_Chicken": chicken_3player,
    "3p_Matching_Pennies": matching_pennies_3player,
    "3p_Common_Interest": common_interest_3player,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _type_label(S):
    """Label a contrast type tuple (axis indices) by a 1-based subset string."""
    return "{" + ",".join(str(i + 1) for i in S) + "}"


def evaluate_game(u):
    record = {}

    # Family matrices
    fmats = all_family_matrices(u)
    family = {}
    for S, M in fmats.items():
        label = _type_label(S)
        family[label] = {
            "matrix": M.tolist(),
            "trace": float(np.trace(M)),
            "det": float(np.linalg.det(M)),
            "rank_numerical": int(np.linalg.matrix_rank(M, tol=1e-10)),
            "diagonal": np.diag(M).tolist(),
            "offdiag_sum": float(np.sum(M) - np.trace(M)),
            "offdiag_min": float(np.min(M[~np.eye(3, dtype=bool)])),
            "offdiag_max": float(np.max(M[~np.eye(3, dtype=bool)])),
        }
    record["family_matrices"] = family

    # Diagnostic degree-3 invariants
    record["degree3_diagnostics"] = diagnostic_degree3(u)

    # Coordination-type / potential-type / harmonic-type quick flags from M_{1,2,3}
    M3 = fmats[(0, 1, 2)]
    offdiag_3 = M3[~np.eye(3, dtype=bool)]
    record["flags"] = {
        "M_three_way_rank": int(np.linalg.matrix_rank(M3, tol=1e-10)),
        "M_three_way_all_positive_offdiag": bool(np.all(offdiag_3 > 1e-10)),
        "M_three_way_all_negative_offdiag": bool(np.all(offdiag_3 < -1e-10)),
        "M_three_way_trace": float(np.trace(M3)),
        "M_three_way_det": float(np.linalg.det(M3)),
        "potential_candidate": bool(
            np.linalg.matrix_rank(M3, tol=1e-10) <= 1 and np.trace(M3) > 1e-10
        ),
        "coordination_type": bool(np.all(offdiag_3 > 1e-10)),
    }

    return record


def _format_payoff_tensor_compact(u):
    """Compact payoff tensor representation: list of (s1,s2,s3) -> (u0,u1,u2)."""
    rows = []
    for s1, s2, s3 in iproduct(range(3), repeat=3):
        rows.append({
            "profile": [s1, s2, s3],
            "payoffs": [float(u[p, s1, s2, s3]) for p in range(3)],
        })
    return rows


def build_atlas():
    """Evaluate all named games and assemble the atlas record."""
    atlas = {}
    for name, builder in NAMED_GAMES.items():
        print(f"  evaluating {name} ...")
        u = builder()
        record = evaluate_game(u)
        record["payoff_tensor_compact"] = _format_payoff_tensor_compact(u)
        atlas[name] = record
    return atlas


def write_json(atlas, path):
    """Write the full atlas as JSON."""
    with open(path, "w") as f:
        json.dump(atlas, f, indent=2)


def write_markdown_table(atlas, path):
    """Write a Markdown table of the named games and their local diagnostics."""
    header = [
        "Game",
        "rank $M_{1,2,3}$",
        "tr $M_{1,2,3}$",
        "off-diag $M_{1,2,3}$",
        "Flag",
        "tr $M_{\\{1\\}}$",
        "tr $M_{\\{1,2\\}}$",
    ]
    rows = []
    for name, rec in atlas.items():
        flag_parts = []
        if rec["flags"]["coordination_type"]:
            flag_parts.append("coord")
        if rec["flags"]["M_three_way_all_negative_offdiag"]:
            flag_parts.append("anti-coord")
        if rec["flags"]["potential_candidate"]:
            flag_parts.append("potential?")
        if not flag_parts:
            flag_parts.append("mixed")
        flag = ", ".join(flag_parts)

        M3 = rec["family_matrices"]["{1,2,3}"]
        rank3 = M3["rank_numerical"]
        tr3 = M3["trace"]
        offdiag_summary = (
            f"min={M3['offdiag_min']:+.2f}, max={M3['offdiag_max']:+.2f}"
        )
        trM1 = rec["family_matrices"]["{1}"]["trace"]
        trM12 = rec["family_matrices"]["{1,2}"]["trace"]

        rows.append([
            name.replace("3p_", "").replace("_", " "),
            str(rank3),
            f"{tr3:+.3f}",
            offdiag_summary,
            flag,
            f"{trM1:+.3f}",
            f"{trM12:+.3f}",
        ])

    # Build markdown
    md_lines = ["# Atlas of (3,3)-game invariant signatures",
                "",
                f"Generated by `atlas_3x3.py`. {len(atlas)} named games.",
                "",
                "| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")
    md_lines.append("Notes:")
    md_lines.append("- rank $M_S$ = numerical rank of the 3x3 family matrix for")
    md_lines.append("  contrast type $S$ (3 = full, 1 = potential candidate, etc.).")
    md_lines.append("- Flag: coordination-type = all $M_{1,2,3}$ off-diagonals positive;")
    md_lines.append("  anti-coord = all negative; potential? = rank-1 $M_{1,2,3}$.")
    md_lines.append("- Full 42 degree-2 entries and 24 degree-3 diagnostics in "
                    "`atlas_3x3_results.json`.")

    with open(path, "w") as f:
        f.write("\n".join(md_lines))


def main():
    print(f"Building atlas of {len(NAMED_GAMES)} named (3,3)-games ...")
    atlas = build_atlas()

    json_path = "atlas_3x3_results.json"
    md_path = "atlas_3x3_table.md"

    write_json(atlas, json_path)
    write_markdown_table(atlas, md_path)

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")

    # Quick summary
    print("\nSummary of M_{1,2,3} structure per game:")
    print(f"{'Game':<35s} {'rank':>5s} {'trace':>10s} {'off-diag flag':>20s}")
    for name, rec in atlas.items():
        M3 = rec["family_matrices"]["{1,2,3}"]
        rank3 = M3["rank_numerical"]
        tr3 = M3["trace"]
        flags = rec["flags"]
        flag = ("coord" if flags["coordination_type"]
                else "anti-coord" if flags["M_three_way_all_negative_offdiag"]
                else "potential?" if flags["potential_candidate"]
                else "mixed")
        print(f"{name:<35s} {rank3:>5d} {tr3:>+10.3f} {flag:>20s}")


if __name__ == "__main__":
    main()
