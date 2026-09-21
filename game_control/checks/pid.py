# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Verify bounded PID feedback, integral-state updates, gradients, and relabeling behavior."""
import numpy as np
import sympy as sp
import torch


def relabeled_measure(m, order, parameters):
    return m.pid_measurement(m.shared_project_games(parameters).relabel(order))

def check_pid(m):
    q = sp.Rational
    L, U, d, b = (q(1, 4), q(3, 8), q(1, 48), q(1, 24))
    lower, upper, conditions = m.pid_bounds.function.args
    t = sp.Symbol('t', real=True)
    lower_expr = [np.asarray(f(m.exact([t])), dtype=object).item() for f in lower]
    upper_expr = [np.asarray(f(m.exact([t])), dtype=object).item() for f in upper]
    breaks = {L, U}
    for group in (lower_expr, upper_expr):
        for i, a in enumerate(group):
            for c in group[i + 1:]:
                roots = sp.solveset(a - c, t, domain=sp.Interval(L, U))
                if isinstance(roots, sp.FiniteSet):
                    breaks.update(roots)
    for current in sorted(breaks):
        interval = m.pid_bounds(m.exact([current]))
        assert list(interval) == [max(-b, L + d - current), min(b, U - d - current)]
        assert interval[0] <= interval[1]
        for command in interval:
            assert m.robust_moves.contains(m.exact([current, command]))
            for w in (-d, d):
                assert m.shared_target.contains(m.exact([current + command + w]))
    for proposed in (-10, q(1, 100), 10):
        for current in (L, q(5, 16), U):
            command = m.project_scalar_command(proposed, m.pid_bounds(m.exact([current])))
            assert m.robust_moves.contains(m.exact([current, command]))
    try:
        m.pid_bounds(m.exact([q(1, 8)]))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty interval silently accepted')
    for states in (m.pid_states, m.pid_tuned_states):
        for state in states:
            assert 0.25 - 1e-12 <= float(state.parameters[0]) <= 0.375 + 1e-12
            game = m.shared_project_games(state.parameters)
            assert bool((m.project_measurements(game) >= 1 / 24 - 1e-12).all())
        for state, following in zip(states, states[1:]):
            bounds = m.pid_bounds(state.parameters)
            assert bounds[0] - 1e-12 <= following.applied <= bounds[1] + 1e-12
    raw_parameters = torch.stack([s.parameters[0] for s in m.pid_unfiltered])
    raw_games = m.shared_project_games(raw_parameters[:, None])
    assert raw_parameters.max() > 0.375 and m.project_measurements(raw_games).min() < 1 / 24
    assert int((abs(m.pid_commands - m.pid_proposals) > 1e-10).sum()) == 11
    assert float(m.pid_loss_after) < float(m.pid_loss_before) * 0.8
    gains = m.pid_initial_gains.clone().requires_grad_()
    loss = m.pid_loss(m.pid_simulate, m.pid_output, m.pid_references, gains)
    gradient = torch.autograd.grad(loss, gains)[0]
    h = 0.0001
    finite = []
    for i in range(3):
        direction = torch.eye(3, dtype=torch.float64)[i] * h
        high = m.pid_loss(m.pid_simulate, m.pid_output, m.pid_references, gains.detach() + direction)
        low = m.pid_loss(m.pid_simulate, m.pid_output, m.pid_references, gains.detach() - direction)
        finite.append(float((high - low) / (2 * h)))
    assert np.allclose(gradient.detach().numpy(), finite, rtol=0.001, atol=1e-09), (gradient, finite)
    for order in (((1, 0), (2, 0, 1), (1, 0)), ((0, 1), (1, 2, 0), (0, 1))):
        observe = m.Map(m.partial(relabeled_measure, m, order))
        states = m.pid_rollout(observe, m.pid_bounds, m.drift_transition, m.pid_initial, m.pid_references, m.pid_disturbances, m.pid_initial_gains, m.pid_back_calculation)
        assert torch.allclose(torch.stack([s.parameters for s in states]), torch.stack([s.parameters for s in m.pid_states]), atol=1e-12, rtol=0)
    generator = torch.Generator().manual_seed(19)
    for trial in range(8):
        initial = torch.tensor([0.25 + 0.125 * trial / 7], dtype=torch.float64)
        disturbances = (2 * torch.randint(0, 2, (20,), generator=generator, dtype=torch.int64) - 1).double() * float(d)
        references = m.pid_output(0.25 + 0.125 * torch.rand(20, 1, generator=generator, dtype=torch.float64))
        gains = torch.randn(3, generator=generator, dtype=torch.float64) * 20
        states = m.pid_rollout(m.pid_output, m.pid_bounds, m.drift_transition, initial, references, disturbances, gains, 0.5)
        assert all((0.25 - 1e-12 <= float(s.parameters[0]) <= 0.375 + 1e-12 for s in states))
    return {'exact_filter_breakpoints': [str(x) for x in sorted(breaks)], 'filter_endpoint_certificates': True, 'full_invariant_target_checked': True, 'filtered_parameter_range': [min((float(s.parameters[0]) for s in m.pid_states)), max((float(s.parameters[0]) for s in m.pid_states))], 'actuator_only_peak': float(raw_parameters.max()), 'filtered_commands': 11, 'training_objective_before': float(m.pid_loss_before), 'training_objective_after': float(m.pid_loss_after), 'tuned_gains': m.pid_tuned_gains.tolist(), 'autodiff_matches_finite_differences': True, 'relabelled_feedback_matches': True, 'additional_adversarial_rollouts': 8}
