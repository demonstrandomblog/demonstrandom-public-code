# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Numerical contrast-family patterns for three-player, three-strategy games.
Games have shape (3,3,3,3), payoff player first. Seven nonempty subsets of
strategy axes label contrast blocks. Layer 1 records block presence, while
layer 2 refines by Gram-matrix signs and ranks. Sampled cells describe the
constructed games only and do not certify all realizable types."""

from collections import Counter
from itertools import combinations, product
import json
import numpy as np

from .contrast_blocks_3x3 import (
    mean_zero_payoff, project_contrast_block, family_matrix, all_family_matrices,
)
from .classify_3x3 import sign3, TOL


TYPES = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
TYPE_LABELS = ["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}", "{1,2,3}"]


def _pattern_bits(pattern: tuple) -> str:
    return "".join(str(b) for b in pattern)


def _all_patterns():
    return list(product([0, 1], repeat=7))


def construct_game(pattern: tuple, rng: np.random.Generator) -> np.ndarray:
    u = np.zeros((3, 3, 3, 3))
    for k, S in enumerate(TYPES):
        if pattern[k] == 0:
            continue
        for p in range(3):
            raw = rng.standard_normal((3, 3, 3))
            block = project_contrast_block(raw, S)
            u[p] += block
    return u


def measure_layer1(u: np.ndarray) -> tuple:
    bits = []
    for S in TYPES:
        active = 0
        for p in range(3):
            block = project_contrast_block(mean_zero_payoff(u)[p], S)
            if float(np.max(np.abs(block))) > TOL:
                active = 1
                break
        bits.append(active)
    return tuple(bits)


def measure_layer2(u: np.ndarray) -> tuple:
    fmats = all_family_matrices(u)
    sig = []
    for S in TYPES:
        M = fmats[S]
        rk = int(np.linalg.matrix_rank(M, tol=1e-10))
        offdiag = M[~np.eye(3, dtype=bool)]
        signs = set(sign3(v) for v in offdiag)
        if 0 in signs and len(signs) > 1:
            signs.discard(0)
        if rk == 0 and abs(np.trace(M)) < TOL:
            label = ("0", 0)
        else:
            if len(signs) > 1:
                offlabel = "M"
            elif signs == {1}:
                offlabel = "+"
            elif signs == {-1}:
                offlabel = "-"
            else:
                offlabel = "0"
            label = (str(rk), offlabel)
        sig.append(label)
    return tuple(sig)


def pattern_interpretation(pattern: tuple) -> str:
    bits = pattern
    n_main = sum(bits[:3])
    n_pair = sum(bits[3:6])
    n_three = bits[6]
    descriptors = []
    if n_main == 0 and n_pair == 0 and n_three == 0:
        return "trivial (zero game)"
    if n_main > 0 and n_pair == 0 and n_three == 0:
        descriptors.append("linear" if n_main == 3 else f"linear in {n_main}/3 axes")
    if n_main == 0 and n_pair > 0 and n_three == 0:
        descriptors.append("pure pairwise" if n_pair == 3 else f"pairwise in {n_pair}/3 pairs")
    if n_main == 0 and n_pair == 0 and n_three == 1:
        descriptors.append("pure three-way")
    if n_main > 0 and n_pair > 0 and n_three == 0:
        descriptors.append("main + pairwise")
    if n_main > 0 and n_pair == 0 and n_three == 1:
        descriptors.append("main + three-way")
    if n_main == 0 and n_pair > 0 and n_three == 1:
        descriptors.append("pairwise + three-way")
    if n_main > 0 and n_pair > 0 and n_three == 1:
        descriptors.append("full structure")
    if not descriptors:
        descriptors.append("mixed")
    return ", ".join(descriptors)


def pass1_layer1_census(rng_seed=0, attempts=8):
    rng = np.random.default_rng(rng_seed)
    cells = {}
    for pattern in _all_patterns():
        success = False
        rep_u = None
        for _ in range(attempts):
            u = construct_game(pattern, rng)
            measured = measure_layer1(u)
            if measured == pattern:
                success = True
                rep_u = u
                break
        cells[_pattern_bits(pattern)] = {
            "pattern": list(pattern),
            "realizable": success,
            "representative": rep_u.tolist() if rep_u is not None else None,
            "n_main": sum(pattern[:3]),
            "n_pair": sum(pattern[3:6]),
            "n_three": int(pattern[6]),
            "interpretation": pattern_interpretation(pattern),
        }
    return cells


def pass2_layer2_refinement(cells: dict, n_samples=80, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    for key, cell in cells.items():
        if not cell["realizable"]:
            cell["layer2_subcells"] = []
            continue
        pattern = tuple(cell["pattern"])
        seen = {}
        for _ in range(n_samples):
            u = construct_game(pattern, rng)
            if measure_layer1(u) != pattern:
                continue
            l2 = measure_layer2(u)
            l2_key = "|".join(f"{a}{b}" for (a, b) in l2)
            seen.setdefault(l2_key, {
                "signature": [list(t) for t in l2],
                "count": 0,
            })
            seen[l2_key]["count"] += 1
        cell["layer2_subcells"] = [
            {"key": k, "signature": v["signature"], "sample_count": v["count"]}
            for k, v in sorted(seen.items(), key=lambda kv: -kv[1]["count"])
        ]
    return cells


def place_named_games(cells: dict):
    from .atlas_3x3 import NAMED_GAMES
    placements = {}
    for name, builder in NAMED_GAMES.items():
        u = builder()
        pattern = measure_layer1(u)
        l2 = measure_layer2(u)
        key = _pattern_bits(pattern)
        placements[name] = {
            "layer1_key": key,
            "layer1_pattern": list(pattern),
            "layer2_signature": [list(t) for t in l2],
        }
        if key in cells:
            cells[key].setdefault("named_games", []).append(name)
    return placements


def write_markdown_table(cells: dict, placements: dict, path: str):
    realizable = [v for v in cells.values() if v["realizable"]]
    empty = [v for v in cells.values() if not v["realizable"]]

    lines = []
    lines.append("# Typology census of $(3,3)$-games under $(S_3)^3$")
    lines.append("")
    lines.append(f"- Layer-1 patterns: 128 total")
    lines.append(f"- Realizable: {len(realizable)}")
    lines.append(f"- Empty/unrealizable: {len(empty)}")

    n_l2 = sum(len(v.get("layer2_subcells", [])) for v in realizable)
    lines.append(f"- Total Layer-2 sub-cells (sampled): {n_l2}")
    lines.append(f"- Named-game placements: {sum(len(v.get('named_games', [])) for v in cells.values())}")
    lines.append("")

    by_struct = {}
    for key, cell in cells.items():
        if not cell["realizable"]:
            continue
        struct = cell["interpretation"]
        by_struct.setdefault(struct, []).append((key, cell))

    lines.append("## Layer-1 cells by structural type")
    lines.append("")
    lines.append("| Structural type | # cells | # Layer-2 sub-cells | Named games placed |")
    lines.append("|---|---:|---:|---|")
    for struct in sorted(by_struct.keys()):
        rows = by_struct[struct]
        n_cells = len(rows)
        n_l2_struct = sum(len(c.get("layer2_subcells", [])) for _, c in rows)
        named = []
        for _, c in rows:
            named.extend(c.get("named_games", []))
        named_str = ", ".join(n.replace("3p_", "").replace("_", " ") for n in named) if named else "—"
        lines.append(f"| {struct} | {n_cells} | {n_l2_struct} | {named_str} |")
    lines.append("")

    lines.append("## Full Layer-1 census (128 cells)")
    lines.append("")
    lines.append("| Pattern | Structural type | Layer-2 sub-cells | Named games |")
    lines.append("|---|---|---:|---|")
    for key in sorted(cells.keys()):
        cell = cells[key]
        if not cell["realizable"]:
            type_str = cell["interpretation"] + " (empty)"
        else:
            type_str = cell["interpretation"]
        n_l2 = len(cell.get("layer2_subcells", []))
        named = cell.get("named_games", [])
        named_str = ", ".join(n.replace("3p_", "").replace("_", " ") for n in named) if named else ""
        lines.append(f"| `{key}` | {type_str} | {n_l2} | {named_str} |")
    lines.append("")

    lines.append("## Pattern bit positions")
    lines.append("")
    lines.append("Bit order in the 7-bit signature: ($\\{1\\}, \\{2\\}, \\{3\\}, \\{1,2\\}, \\{1,3\\}, \\{2,3\\}, \\{1,2,3\\}$).")
    lines.append("Each bit indicates whether the corresponding contrast family is active in the game.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    print("Pass 1: enumerating 128 Layer-1 patterns ...")
    cells = pass1_layer1_census(rng_seed=0, attempts=8)
    n_real = sum(1 for v in cells.values() if v["realizable"])
    print(f"  Realizable: {n_real} / 128")
    if n_real < 128:
        for key, v in cells.items():
            if not v["realizable"]:
                print(f"    empty: {key} ({v['interpretation']})")

    print()
    print("Pass 2: Layer-2 refinement (sampling each realizable cell) ...")
    cells = pass2_layer2_refinement(cells, n_samples=80, rng_seed=1)
    total_l2 = sum(len(v.get("layer2_subcells", [])) for v in cells.values())
    print(f"  Total Layer-2 sub-cells: {total_l2}")

    print()
    print("Placing 13 named games into their Layer-1 cells ...")
    placements = place_named_games(cells)
    for name, info in placements.items():
        short = name.replace("3p_", "").replace("_", " ")
        print(f"  {short:<30s} -> Layer-1 cell {info['layer1_key']}")

    out_json = "typology_census_3x3.json"
    out_md = "typology_census_3x3_table.md"

    serializable = {
        "cells": {
            k: {kk: vv for kk, vv in v.items() if kk != "representative"}
            for k, v in cells.items()
        },
        "named_game_placements": placements,
    }
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote {out_json}")

    write_markdown_table(cells, placements, out_md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
