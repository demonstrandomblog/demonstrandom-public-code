# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Verify exact polynomial feasibility, optimization, disturbances, and solver edge cases."""
import sympy as sp


def square_commands(commands):
    return commands ** 2

def disturbed_parameter(m, current, radius, disturbance, commands):
    return m.stack([commands[..., 0] ** 2 + m.like(current + radius * (1 - disturbance ** 2), commands)], axis=-1)

def check(m):
    q = sp.Rational
    x, y, w = sp.symbols('x y w', real=True)
    reports = {}
    answer = m.solve_polynomial(m.project_problem, m.project_command_symbols)
    assert answer.status == 'verified_optimal' and answer.verify()
    assert answer.data['point'] == [q(1, 4), q(1, 4)]
    assert answer.data['cost'] == q(1, 2)
    assert m.project_problem.select(m.exact(answer.data['point'])).verify()
    reports['same_problem_as_affine_solver'] = answer.status
    print('affine comparison passed', flush=True)
    square_control = m.Control(m.Map(square_commands).then_apply(m.project_control.parameters), m.ge(m.Map(lambda v: v), 0), m.Map(lambda v: v.sum()))
    square_problem = m.design(m.project_family, square_control, m.project_target)
    answer = m.solve_polynomial(square_problem, (x, y))
    assert answer.status == 'verified_optimal' and answer.verify(), (answer.status, answer.data)
    assert answer.data['point'] == [q(1, 2), q(1, 2)]
    assert square_problem.select(m.exact(answer.data['point'])).verify()
    reports['nonlinear_actuator_full_invariant_target'] = answer.status
    print('nonlinear actuator passed', flush=True)
    one_game = m.Game(m.exact([[[0, 1], [1, 0]], [[0, 1], [1, 0]]]), (2, 2))
    gram_family = one_game.with_payoff_rule(lambda base, v: base * v[0])
    gram_target = m.ge(m.gram((0, 1), 0, 1), q(1, 4))
    gram_problem = m.design(gram_family, m.Control(m.Map(lambda v: v), m.Region(), m.Map(lambda v: v[0] ** 2)), gram_target)
    answer = m.solve_polynomial(gram_problem, (x,))
    assert answer.status == 'verified_optimal' and answer.verify(), (answer.status, answer.data)
    assert gram_problem.select(m.exact(answer.data['point'])).verify()
    reports['quadratic_gram_target'] = answer.status
    parameter_map = m.Map(m.partial(disturbed_parameter, m, q(1, 4), q(1, 16), w))
    uncertain_control = m.Control(parameter_map, m.ge(m.Map(lambda v: v), -1) & m.ge(m.Map(lambda v: -v), -1), m.Map(lambda v: (v[0] - 1) ** 2))
    uncertain_problem = m.design(m.shared_project_games, uncertain_control, m.project_target)
    answer = m.solve_polynomial(uncertain_problem, (x,), disturbances={w: (-1, 1)})
    assert answer.status == 'verified_optimal' and answer.verify(), (answer.status, answer.data)
    assert answer.data['point'] == [q(1, 4)] and answer.data['cost'] == q(9, 16)
    for disturbance in (-1, 0, 1):
        original = m.shared_project_games(disturbed_parameter(m, q(1, 4), q(1, 16), disturbance, m.exact(answer.data['point'])))
        assert m.project_target.contains(original)
    original = m.shared_project_games(disturbed_parameter(m, q(1, 4), q(1, 16), 0, m.exact([sp.sqrt(q(1, 8))])))
    assert not m.project_target.contains(original)
    reports['interior_disturbance_full_target'] = answer.status
    print('universal disturbance passed', flush=True)
    region = m.eq(m.Map(lambda v: v[0] ** 2), 2) & m.ge(m.Map(lambda v: v[0]), 0)
    problem = m.Problem(m.Map(lambda v: v), region, m.Map(lambda v: v[0]))
    answer = m.solve_polynomial(problem, (x,))
    assert answer.status == 'verified_optimal' and answer.verify()
    assert sp.simplify(answer.data['point'][0] ** 2 - 2) == 0
    assert problem.select(m.exact(answer.data['point'])).verify()
    reports['algebraic_witness'] = str(answer.data['point'][0])
    impossible = m.Problem(m.Map(lambda v: v), m.ge(m.Map(lambda v: -v[0] ** 2), 1), m.Map(lambda v: v[0] ** 2))
    answer = m.solve_polynomial(impossible, (x,))
    assert answer.status == 'verified_infeasible' and answer.verify()
    reports['nonlinear_infeasibility'] = answer.status
    alternatives = m.ge(m.Map(lambda v: w), 0) | m.ge(m.Map(lambda v: -w), 0)
    problem = m.Problem(m.Map(lambda v: v), alternatives, m.Map(lambda v: v[0] ** 2))
    answer = m.solve_polynomial(problem, (x,), disturbances={w: (-1, 1)})
    assert answer.status == 'verified_optimal' and answer.verify(), (answer.status, answer.data)
    reports['forall_over_union'] = answer.status
    positive = m.eq(m.Map(lambda v: v[0] * v[1]), 1) & m.ge(m.Map(lambda v: v[0]), 0)
    problem = m.Problem(m.Map(lambda v: v), positive, m.Map(lambda v: v[0]))
    answer = m.solve_polynomial(problem, (x, y))
    assert answer.status == 'verified_feasible' and answer.verify(), (answer.status, answer.data)
    reports['unattained_minimum'] = answer.data.get('reason')
    for expression in (lambda v: sp.sin(v[0]), lambda v: v[0] + 0.1):
        problem = m.Problem(m.Map(lambda v: v), m.ge(m.Map(expression), 0), m.Map(lambda v: v[0] ** 2))
        try:
            m.solve_polynomial(problem, (x,))
        except (ValueError, sp.PolynomialError, sp.CoercionFailed):
            pass
        else:
            raise AssertionError('Unsupported expression accepted')
    reports['unsupported_expressions_rejected'] = True
    return reports
