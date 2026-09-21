# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Verify symbolic/tensor agreement, relabeling invariance, and constrained control certificates."""
import itertools
import numpy as np
import sympy as sp
import torch


def verify(m):
    q = sp.Rational
    checks = {}
    assert m.project_solution_valid and m.project_answer.verify()
    assert list(m.project_solution.command) == [q(1, 4)] * 2
    assert list(m.project_measurements(m.project_solution.game())) == [q(1, 12), q(1, 24), q(1, 24)]
    v = sp.symbols('v1 v2', real=True)
    symbolic = m.project_measurements(m.project_games(m.exact(v)))
    jac = sp.Matrix(symbolic).jacobian(v)
    point = torch.tensor([0.17, 0.21], dtype=torch.float64)
    assert np.allclose(torch.func.jacfwd(m.project_types)(point), np.array(jac, dtype=float))
    batch = torch.tensor([[0.17, 0.21], [0.25, 0.25]], dtype=torch.float64)
    assert torch.allclose(m.project_types(batch), torch.stack([m.project_types(x) for x in batch]))
    perms = tuple(itertools.product(*(itertools.permutations(range(size)) for size in m.project_shape)))
    for order in perms:
        game = m.project_games(m.exact(v))
        assert all((sp.expand(a - b) == 0 for a, b in zip(m.project_measurements(game), m.project_measurements(game.relabel(order)))))
    checks['running_family_relabelings'] = len(perms)
    assert m.project_route_valid and (not m.project_direct_safe)
    assert all((m.segment_ok(m.project_safe_commands, a, b) for a, b in zip(m.project_route, m.project_route[1:])))
    assert m.matched_safe_set == sp.Union(sp.Interval(0, q(1, 8)), sp.Interval(q(1, 4), q(1, 2)))
    assert m.matched_target_set == sp.Interval(q(1, 4), q(3, 8))
    checks['safe_route_and_disconnected_actuator'] = True
    for branch in m.project_safe_games.branches:
        for constraint in branch.clauses[0]:
            result = constraint.value(m.project_games(batch))
            assert isinstance(result, torch.Tensor) and result.shape == (2,)
    assert torch.allclose(m.shared_project_games(torch.tensor([[0.25], [0.3]], dtype=torch.float64)).payoffs, torch.stack([m.shared_project_games(torch.tensor([t], dtype=torch.float64)).payoffs for t in (0.25, 0.3)]))
    assert m.drift_interval == sp.Interval(q(1, 4), q(3, 8))
    L, U, d = (q(1, 4), q(3, 8), q(1, 48))
    for t in (L, L + d, (L + U) / 2, U - d, U):
        answer, selected = m.drift_policy(t)
        assert answer.verify() and selected.verify()
        c = selected.command[0]
        assert c == max(L + d - t, min(0, U - d - t))
        for w in (-d, d):
            assert m.shared_target.contains(m.exact([t + c + w]))
    t = sp.Symbol('t', real=True)
    for lo, hi, c in ((L, L + d, L + d - t), (L + d, U - d, sp.S.Zero), (U - d, U, U - d - t)):
        for endpoint in (lo, hi):
            assert m.robust_moves.contains(m.exact([endpoint, c.subs(t, endpoint)]))
        assert sp.Poly(t + c, t).degree() <= 1
    checks['robust_policy_and_interval_closure'] = True
    assert m.permission_verified and m.added_permissions == (3, 4)
    assert len(m.permission_trials) == 4
    for added, answer in m.permission_trials:
        assert answer.verify()
        assert answer.status == ('certified_optimal' if len(added) == 2 else 'certified_infeasible')
    assert list(m.conserved_measurements) == [q(1, 6), 0, 0]
    support_checks = 0
    for size in range(3):
        for S in itertools.combinations(range(3), size):
            for actions in itertools.product(*(range(m.project_shape[i]) for i in S)):
                mask = np.ones(m.project_shape, dtype=int)
                for i, a in zip(S, actions):
                    mask *= (np.arange(m.project_shape[i]) == a).reshape(tuple((m.project_shape[i] if j == i else 1 for j in range(3))))
                for recipient in range(3):
                    payoff = m.exact(np.zeros((3,) + m.project_shape, dtype=int))
                    payoff[recipient] = mask
                    assert all((x == 0 for x in m.effect(m.Game(payoff, m.project_shape), (0, 1, 2)).flat))
                    support_checks += 1
    checks['lower_order_basis_elements_preserved'] = support_checks
    assert m.fiber_verified and sp.simplify(m.fiber_minimum['cost'] - sp.sqrt(2)) == 0
    assert m.fiber_minimum['domain'] == sp.Interval(q(1, 4), 1)
    curve = m.fiber_curve(m.exact([m.product_symbols[0]]))
    actual = m.product_problem.games(curve).payoffs
    reference = m.product_problem.games(m.reference_command).payoffs
    assert all((sp.cancel(a - b) == 0 for a, b in zip(actual.flat, reference.flat)))
    assert torch.linalg.norm(m.local_jacobian @ m.local_nullspace) < 1e-12
    assert torch.linalg.norm(m.finite_measurement_change) > 1e-06
    assert torch.count_nonzero(m.zero_product_jacobian) == 0
    checks['exact_fiber_minimum_and_local_limit'] = True
    assert m.response_equilibria == ((0, 0, 0), (1, 2, 1))
    assert m.response_traces[0][-1] == (0, 0, 0) and m.response_traces[1][-1] == (1, 2, 1)
    assert m.response_target_verified
    checks['behavioral_counterexample'] = True
    assert m.activation_rank == 6 and m.activation_preserves_target
    expected = {p for p in itertools.product(range(2), range(3), range(2)) if sum((x != 0 for x in p)) >= 2}
    assert m.release_profiles == expected
    for layers, policies in ((m.baseline_layers, m.baseline_policy), (m.activation_layers, m.activation_policy)):
        for rank, policy in enumerate(policies, 1):
            for state, options in policy.items():
                for command in options:
                    following = m.activation_successors(state, command)
                    assert following and all((x in layers[rank - 1] for x in following))
    assert all((np.array_equal(z, m.response_coordinates) for z in m.activation_measurements))
    checks['finite_policy_states'] = len(m.activation_states)
    assert m.network_verified and m.uniform_answer.status == 'certified_infeasible'
    assert list(m.network_selected.command[:6]) == [q(1, 4), q(1, 20), 0, q(3, 20), -q(1, 20), 0]
    assert m.network_answer.data['cost'] == q(1, 2)
    assert any((x != 0 for x in m.effect(m.network_selected.game(), (0, 2, 4)).flat))
    checks['network_entries'] = m.network_selected.game().payoffs.size
    assert m.policy_verified and (not m.policy_adaptive_equilibrium)
    assert isinstance(m.policy_rate_region, sp.Interval) and m.policy_rate_region.end == 4
    assert 1.6028 < float(m.policy_rate_region.start) < 1.603
    for p in m.policy_polynomials:
        assert sp.Poly(p, m.rate_symbol).degree() == 2
        assert sp.solve_univariate_inequality(p - q(1, 20) >= 0, m.rate_symbol, relational=False).is_superset(m.policy_rate_region)
    assert min((abs(float(p.subs(m.rate_symbol, m.policy_rate_region.start)) - 0.05) for p in m.policy_polynomials)) < 1e-12
    sample = m.exact([q(1, 2), q(3, 2), q(2, 3), q(5, 2)])
    assert np.array_equal(m.compiled_family(sample).payoffs, m.policy_family(sample).payoffs)
    sample_t = torch.tensor(np.array(sample, dtype=float), dtype=torch.float64)
    assert torch.allclose(m.compiled_family(sample_t).payoffs, m.policy_family(sample_t).payoffs, atol=1e-12, rtol=1e-12)
    r = torch.tensor([2.2], dtype=torch.float64)
    numerical = torch.func.jacfwd(m.policy_problem.games.then_apply(m.policy_measurements))(r).numpy().ravel()
    expected = np.array([float(sp.diff(p, m.rate_symbol).subs(m.rate_symbol, 2.2)) for p in m.policy_polynomials])
    assert np.allclose(numerical, expected, atol=1e-12)
    for shape, family, values, measurements in ((m.network_shape, m.network_family, m.network_weights, m.network_measurements), (m.policy_shape, m.policy_family, sample, m.policy_measurements)):
        rng = np.random.default_rng(8)
        game = family(values)
        for _ in range(8):
            relabel = tuple((tuple(rng.permutation(a)) for a in shape))
            assert all((sp.simplify(a - b) == 0 for a, b in zip(measurements(game), measurements(game.relabel(relabel)))))
    checks['policy_entries'] = m.policy_selected.game().payoffs.size
    assert m.information_status == 'sufficient' and m.information_commands == [[0], [2]]
    assert m.information_audit[0][1].status == 'certified_infeasible' and m.information_audit[0][1].verify()
    assert all((answer.verify() for ids, answer in m.refined_audit))
    assert m.information_separator_values == [q(5, 192), -q(5, 192)]
    for order in perms:
        for game in m.information_games:
            assert m.refined_observations[-1](game) == m.refined_observations[-1](game.relabel(order))
    checks['information_obstruction_and_discovered_cubic'] = True
    return checks
