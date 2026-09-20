# Generated from demonstrandom/game_theory/posts/engineering_game_types/index.qmd
# Edit the article, then regenerate this file.
# AI-generated research code; no full human review is recorded.
# Validate results independently; see the article and package README.
from dataclasses import dataclass

from functools import partial, reduce

from itertools import (
    combinations,
    combinations_with_replacement,
    islice,
    permutations,
    product,
)

from math import prod

from operator import and_, mul

from typing import Callable

import numpy as np

import sympy as sp

import torch

from scipy.optimize import linprog

def compose_maps(first, second, values):
    return second(first(values))

@dataclass(frozen=True)
class Map:
    function: Callable

    def evaluate(self, values):
        return self.function(values)

    def __call__(self, values):
        return self.evaluate(values)

    def then_apply(self, next_map):
        return Map(partial(compose_maps, self, next_map))

@dataclass(frozen=True)
class Game:
    payoffs: torch.Tensor | np.ndarray
    shape: tuple

    def with_payoff_rule(self, payoff_rule):
        return Map(partial(instantiate_game, self, payoff_rule))

    def relabel(self, permutations):
        u = self.payoffs
        for i, order in enumerate(permutations):
            index = [slice(None)] * u.ndim
            index[u.ndim - len(self.shape) + i] = list(order)
            u = u[tuple(index)]
        return Game(u, self.shape)

def instantiate_game(base, payoff_rule, theta):
    return Game(payoff_rule(like(base.payoffs, theta), theta), base.shape)

def exact(values):
    array = np.asarray(values, dtype=object)
    return np.array([sp.sympify(v) for v in array.flat], dtype=object).reshape(
        array.shape
    )

def like(values, x):
    if isinstance(x, torch.Tensor):
        if isinstance(values, torch.Tensor):
            return values.to(x)
        return x.new_tensor(np.asarray(values, dtype=float))
    return exact(values)

def stack(values, axis=0):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)

def average(x, axes, keepdims=False):
    size = prod(x.shape[i] for i in axes)
    total = x.sum(axis=axes, keepdims=keepdims)
    return total / size if isinstance(x, torch.Tensor) else total * sp.Rational(1, size)

def evaluate_polynomial(terms, theta):
    result = 0
    for powers, coefficient in terms:
        monomial = reduce(mul, (theta[..., j] ** k for j, k in enumerate(powers)), 1)
        if not isinstance(monomial, torch.Tensor):
            monomial = np.asarray(monomial, dtype=object)
        array = like(coefficient, theta)
        result = result + monomial[(...,) + (None,) * array.ndim] * array
    return result

def polynomial(terms):
    terms = tuple(terms)
    return Map(partial(evaluate_polynomial, terms))

def subsets(n):
    return (S for k in range(n + 1) for S in combinations(range(n), k))

def effect(game, S):
    u = game.payoffs
    for i in range(len(game.shape)):
        axis = u.ndim - len(game.shape) + i
        mean = average(u, (axis,), keepdims=True)
        u = u - mean if i in S else mean
    return u

def recipient(u, p, n):
    return u[(..., p) + (slice(None),) * n]

def average_relabelings(function, relabelings, game):
    return sum(function(game.relabel(s)) for s in relabelings) / len(relabelings)

def reynolds(function, relabelings):
    return Map(partial(average_relabelings, function, relabelings))

def flatten_payoffs(game):
    return game.payoffs.reshape((*game.payoffs.shape[: -len(game.shape) - 1], -1))

def payoff_polynomial(terms):
    return Map(flatten_payoffs).then_apply(polynomial(terms))

def contract_moment(factors, game):
    n = len(game.shape)
    blocks = {S: effect(game, S) for S, p in factors}
    value = reduce(mul, (recipient(blocks[S], p, n) for S, p in factors))
    return average(value, tuple(range(value.ndim - n, value.ndim)))

def moment(*factors):
    factors = tuple((tuple(S), p) for S, p in factors)
    return Map(partial(contract_moment, factors))

def gram(S, p, q):
    return moment((S, p), (S, q))

def expected(game, strategies):
    values = game.payoffs
    for i in reversed(range(len(game.shape))):
        probability = like(strategies[i], values)
        weights = probability[(...,) + (None,) * (i + 1) + (slice(None),)]
        values = (values * weights).sum(axis=-1)
    return values

@dataclass(frozen=True)
class Constraint:
    value: Callable
    relation: str

@dataclass(frozen=True)
class Region:
    clauses: tuple = ((),)

    @property
    def branches(self):
        return tuple(Region((clause,)) for clause in self.clauses)

    @property
    def constraints(self):
        if len(self.clauses) != 1:
            raise ValueError(
                "Select a branch before extracting simultaneous constraints"
            )
        return self.clauses[0]

    def __and__(self, other):
        return Region(tuple(a + b for a in self.clauses for b in other.clauses))

    def __or__(self, other):
        return Region(self.clauses + other.clauses)

    def pullback(self, mapping):
        return Region(
            tuple(
                tuple(
                    Constraint(Map(mapping).then_apply(c.value), c.relation)
                    for c in clause
                )
                for clause in self.clauses
            )
        )

    def atoms(self, symbols):
        x = exact(symbols)
        return [
            (sp.expand(value), c.relation)
            for c in self.constraints
            for value in np.asarray(c.value(x), dtype=object).flat
        ]

    def contains(self, x):
        for clause in self.clauses:
            satisfied = True
            for constraint in clause:
                value = constraint.value(x)
                if not isinstance(value, torch.Tensor):
                    value = np.asarray(value, dtype=object)
                if constraint.relation == "eq" and not isinstance(value, torch.Tensor):
                    condition = np.asarray([
                        sp.sympify(entry).equals(0) is True for entry in value.flat
                    ])
                else:
                    condition = value == 0 if constraint.relation == "eq" else value >= 0
                satisfied = satisfied and bool(condition.all())
            if satisfied:
                return True
        return False

def subtract_target(target, measured):
    return measured - like(target, measured)

def ge(measure, lower):
    difference = Map(measure).then_apply(partial(subtract_target, lower))
    return Region(((Constraint(difference, "ge"),),))

def eq(measure, value):
    difference = Map(measure).then_apply(partial(subtract_target, value))
    return Region(((Constraint(difference, "eq"),),))

def action_relabelings(shape):
    return product(*(permutations(range(size)) for size in shape))

def comparison_groups(shape, preserve):
    indices = np.arange(len(shape) * prod(shape)).reshape((len(shape), *shape))
    if preserve == "preferences":
        return tuple(tuple(row.flat) for row in indices)
    groups = []
    for p, size in enumerate(shape):
        for profile in product(*(range(m) for m in shape)):
            if profile[p] == 0:
                group = tuple(
                    indices[(p, *profile[:p], action, *profile[p + 1 :])]
                    for action in range(size)
                )
                groups.append(group)
    return tuple(groups)

def reference_comparisons(values, groups):
    ties, advantages = [], []
    for group in groups:
        ordered = sorted(group, key=values.__getitem__)
        for lower, higher in zip(ordered, ordered[1:]):
            if values[higher] == values[lower]:
                ties.append(tuple(sorted((higher, lower))))
            else:
                advantages.append((higher, lower))
    return tuple(sorted(ties)), tuple(sorted(advantages))

def payoffs_for_shape(shape, game):
    if game.shape != shape:
        raise ValueError("Reference and candidate must have the same action counts")
    return flatten_payoffs(game)

def compare_entries(shape, pairs, game):
    values = payoffs_for_shape(shape, game)
    return values[..., [a for a, b in pairs]] - values[..., [b for a, b in pairs]]

def same_type_as(reference, *, preserve="preferences", margin=None):
    if preserve not in ("preferences", "incentives", "payoffs"):
        raise ValueError("Choose preferences, incentives, or payoffs")
    if preserve != "payoffs" and (margin is None or margin <= 0):
        raise ValueError("Supply a positive margin for strict payoff advantages")
    if tuple(reference.payoffs.shape) != (len(reference.shape), *reference.shape):
        raise ValueError("Supply one reference game")
    groups = (
        comparison_groups(reference.shape, preserve) if preserve != "payoffs" else ()
    )
    target, seen = Region(()), set()
    for relabeling in action_relabelings(reference.shape):
        values = flatten_payoffs(reference.relabel(relabeling))
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        key = (
            tuple(values)
            if preserve == "payoffs"
            else reference_comparisons(values, groups)
        )
        if key in seen:
            continue
        seen.add(key)
        if preserve == "payoffs":
            branch = eq(Map(partial(payoffs_for_shape, reference.shape)), values)
        else:
            ties, advantages = key
            branch = eq(Map(partial(compare_entries, reference.shape, ties)), 0)
            branch = branch & ge(
                Map(partial(compare_entries, reference.shape, advantages)), margin
            )
        target = target | branch
    return target

def compute_deviation_gaps(profile, game):
    values = []
    for p, size in enumerate(game.shape):
        chosen = game.payoffs[(..., p, *profile)]
        for b in range(size):
            if b != profile[p]:
                alternative = (*profile[:p], b, *profile[p + 1 :])
                values.append(chosen - game.payoffs[(..., p, *alternative)])
    return (
        stack(values, axis=-1)
        if values
        else game.payoffs.reshape((*game.payoffs.shape[: -len(game.shape) - 1], -1))[
            ..., :0
        ]
    )

def deviation_gaps(profile):
    return Map(partial(compute_deviation_gaps, profile))

def compute_dominance_gaps(actions, game):
    gaps = []
    for profile in product(*(range(m) for m in game.shape)):
        for p, chosen in enumerate(actions):
            if profile[p] != chosen:
                preferred = (*profile[:p], chosen, *profile[p + 1 :])
                gaps.append(
                    game.payoffs[(..., p, *preferred)]
                    - game.payoffs[(..., p, *profile)]
                )
    return (
        stack(gaps, axis=-1)
        if gaps
        else game.payoffs.reshape((*game.payoffs.shape[: -len(game.shape) - 1], -1))[
            ..., :0
        ]
    )

def dominance_gaps(actions):
    return Map(partial(compute_dominance_gaps, actions))

def compute_mixed_gaps(strategies, game):
    current = expected(game, strategies)
    gaps = []
    for p, size in enumerate(game.shape):
        for action in range(size):
            deviation = list(strategies)
            deviation[p] = exact([int(a == action) for a in range(size)])
            gaps.append(current[..., p] - expected(game, deviation)[..., p])
    return stack(gaps, axis=-1)

def mixed_gaps(strategies):
    return Map(partial(compute_mixed_gaps, strategies))

def potential_residual(game):
    n = len(game.shape)
    total = game.payoffs.sum() * 0
    for S in subsets(n):
        if len(S) < 2:
            continue
        block = effect(game, S)
        values = [recipient(block, p, n) for p in S]
        mean = sum(values) / len(S)
        for value in values:
            residual = (value - mean) ** 2
            total = total + average(
                residual, tuple(range(residual.ndim - n, residual.ndim))
            )
    return total

def potential(game):
    n = len(game.shape)
    return sum(
        sum(recipient(effect(game, S), p, n) for p in S) / len(S)
        for S in subsets(n)
        if S
    )

def compute_potential_differences(S, game):
    n = len(game.shape)
    block = effect(game, S)
    reference = recipient(block, S[0], n)
    return stack([recipient(block, p, n) - reference for p in S[1:]], axis=-1)

def potential_differences(S):
    S = tuple(S)
    return Map(partial(compute_potential_differences, S))

@dataclass(frozen=True)
class Control:
    parameters: Map
    allowed: Region
    cost: Map
    names: tuple = ()

def game_payoffs(game):
    return game.payoffs

@dataclass(frozen=True)
class Problem:
    games: Map
    feasible: Region
    cost: Map
    names: tuple = ()

    def select(self, command):
        return Intervention(self, command)

    def equivalent_to(self, command, observation=Map(game_payoffs)):
        values = self.games.then_apply(observation)
        return self.feasible & eq(values, values(command))

@dataclass(frozen=True)
class Intervention:
    problem: Problem
    command: torch.Tensor | np.ndarray

    def named(self):
        return dict(zip(self.problem.names, self.command))

    def game(self):
        return self.problem.games(self.command)

    def cost(self):
        return self.problem.cost(self.command)

    def verify(self):
        return self.problem.feasible.contains(self.command)

    def change_from(self, baseline, observation=Map(game_payoffs)):
        values = self.problem.games.then_apply(observation)
        return values(self.command) - values(baseline)

    def alternatives(self, observation=Map(game_payoffs)):
        return self.problem.equivalent_to(self.command, observation)

def design(family, control, target):
    games = control.parameters.then_apply(family)
    return Problem(
        games, control.allowed & target.pullback(games), control.cost, control.names
    )

def local_controls(values, command, desired_change):
    jacobian = torch.func.jacfwd(values)(command).reshape(-1, command.numel())
    direction = torch.linalg.lstsq(jacobian, desired_change.reshape(-1)).solution
    _, _, vh = torch.linalg.svd(jacobian, full_matrices=True)
    rank = int(torch.linalg.matrix_rank(jacobian))
    return direction, vh[rank:].T, jacobian @ direction - desired_change.reshape(-1)

def affine_system(problem, symbols):
    atoms = problem.feasible.atoms(symbols)
    A, b = sp.linear_eq_to_matrix(
        [-v for v, relation in atoms if relation == "ge"], symbols
    )
    E, e = sp.linear_eq_to_matrix(
        [v for v, relation in atoms if relation == "eq"], symbols
    )
    objective, offset = sp.linear_eq_to_matrix([problem.cost(exact(symbols))], symbols)
    return A.tolist(), list(b), E.tolist(), list(e), list(objective), -offset[0]

def eliminate(inequalities, variable):
    positive, negative, independent = [], [], []
    for expression in inequalities:
        coefficient = sp.expand(expression).coeff(variable)
        rest = sp.expand(expression - coefficient * variable)
        if coefficient > 0:
            positive.append((coefficient, rest))
        elif coefficient < 0:
            negative.append((coefficient, rest))
        else:
            independent.append(rest)
    return independent + [
        sp.expand(a * d - c * b) for a, b in positive for c, d in negative
    ]

def constraint_loss(region, command):
    loss = command.sum() * 0
    for constraint in region.constraints:
        value = constraint.value(command)
        error = value if constraint.relation == "eq" else torch.relu(-value)
        loss = loss + error.square().sum()
    return loss

def search(problem, initial, steps, rate, cost_weight):
    candidates = []
    for branch in problem.feasible.branches:
        command = initial.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([command], lr=rate)
        for _ in range(steps):
            loss = constraint_loss(branch, command) + cost_weight * problem.cost(
                command
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        candidate = command.detach()
        feasible = branch.contains(candidate)
        score = (
            problem.cost(candidate) if feasible else constraint_loss(branch, candidate)
        )
        candidates.append((not feasible, float(score), candidate))
    return min(candidates, key=lambda entry: entry[:2])[2] if candidates else None

def dot(left, right):
    return sum(a * b for a, b in zip(left, right))

def satisfies_linear_constraints(point, A, b, E, e):
    return all(dot(row, point) <= bound for row, bound in zip(A, b)) and all(
        dot(row, point) == bound for row, bound in zip(E, e)
    )

def verify_linear_certificate(source, certificate):
    c, A, b, E, e = (source[key] for key in ("c", "A", "b", "E", "e"))
    kind = certificate.get("kind")
    if kind not in ("feasible", "optimal", "infeasible"):
        return False
    if kind in ("feasible", "optimal"):
        point = certificate["point"]
        if len(point) != len(c) or not satisfies_linear_constraints(point, A, b, E, e):
            return False
        if kind == "feasible":
            return True
    lam, nu = certificate["lambda"], certificate["nu"]
    if len(lam) != len(A) or len(nu) != len(E) or any(value < 0 for value in lam):
        return False
    stationarity = [
        sum(weight * row[j] for weight, row in zip(lam + nu, A + E))
        for j in range(len(c))
    ]
    bound = dot(b, lam) + dot(e, nu)
    if kind == "infeasible":
        return all(value == 0 for value in stationarity) and bound < 0
    return all(value == -cost for value, cost in zip(stationarity, c)) and (
        dot(c, point) == -bound
    )

@dataclass
class Result:
    status: str
    data: dict
    certificate: dict
    source: dict

    def verify(self):
        return verify_linear_certificate(self.source, self.certificate)

def numerical_lp(c, A, b, E, e):
    return linprog(
        np.asarray(c, dtype=float),
        A_ub=np.asarray(A, dtype=float) if A else None,
        b_ub=np.asarray(b, dtype=float) if A else None,
        A_eq=np.asarray(E, dtype=float) if E else None,
        b_eq=np.asarray(e, dtype=float) if E else None,
        bounds=[(None, None)] * len(c),
        method="highs",
    )

def rational_point(point, A, b, E, e, tolerance=1e-7):
    variables = sp.symbols(f"_v0:{len(point)}")
    rows, bounds = list(E), list(e)
    for row, bound in zip(A, b):
        if abs(float(dot(row, point) - bound)) <= tolerance:
            rows.append(row)
            bounds.append(bound)
    rounded = [sp.Rational(float(value)).limit_denominator(10**8) for value in point]
    if rows:
        solutions = sp.linsolve((sp.Matrix(rows), sp.Matrix(bounds)), variables)
        if solutions is sp.EmptySet:
            return None
        substitutions = dict(zip(variables, rounded))
        candidate = [
            sp.cancel(value.subs(substitutions)) for value in next(iter(solutions))
        ]
    else:
        candidate = rounded
    if not satisfies_linear_constraints(candidate, A, b, E, e):
        return None
    return candidate

def exact_lp(c, A, b, E=(), e=()):
    c, b, e = ([sp.Rational(value) for value in values] for values in (c, b, e))
    A = [[sp.Rational(value) for value in row] for row in A]
    E = [[sp.Rational(value) for value in row] for row in E]
    source = dict(c=c, A=A, b=b, E=E, e=e)
    k, l = len(A), len(E)
    stationarity = [[row[j] for row in A + E] for j in range(len(c))]
    nonnegative = [[-int(i == j) for i in range(k + l)] for j in range(k)]

    if not c:
        lam, nu = [0] * k, [0] * l
        for j, bound in enumerate(b):
            if bound < 0:
                lam[j] = 1
                break
        else:
            for j, bound in enumerate(e):
                if bound != 0:
                    nu[j] = -sp.sign(bound)
                    break
        feasible = satisfies_linear_constraints([], A, b, E, e)
        certificate = dict(
            kind="optimal" if feasible else "infeasible",
            point=[],
            **{"lambda": lam, "nu": nu},
        )
        data = {"point": [], "objective": sp.S.Zero} if feasible else {}
        return Result(
            "certified_optimal" if feasible else "certified_infeasible",
            data,
            certificate,
            source,
        )

    answer = numerical_lp(c, A, b, E, e)
    if answer.status == 2:
        dual_A = nonnegative + [b + e]
        dual_b = [0] * k + [-1]
        dual_answer = numerical_lp(
            [0] * (k + l), dual_A, dual_b, stationarity, [0] * len(c)
        )
        dual = (
            rational_point(dual_answer.x, dual_A, dual_b, stationarity, [0] * len(c))
            if dual_answer.success
            else None
        )
        if dual is not None:
            certificate = {"kind": "infeasible", "lambda": dual[:k], "nu": dual[k:]}
            if verify_linear_certificate(source, certificate):
                return Result("certified_infeasible", {}, certificate, source)
        return Result(
            "unresolved",
            {"reason": "No exact infeasibility witness reconstructed"},
            {},
            source,
        )
    if not answer.success:
        return Result("unresolved", {"reason": answer.message}, {}, source)

    point = rational_point(answer.x, A, b, E, e)
    if point is None:
        return Result(
            "unresolved",
            {"reason": "No exact feasible point reconstructed"},
            {},
            source,
        )
    data = {"point": point, "objective": dot(c, point)}
    guess = list(-answer.ineqlin.marginals) + list(-answer.eqlin.marginals)
    dual = (
        rational_point(
            guess, nonnegative, [0] * k, stationarity, [-value for value in c]
        )
        if k + l
        else []
    )
    if dual is not None:
        certificate = {
            "kind": "optimal",
            "point": point,
            "lambda": dual[:k],
            "nu": dual[k:],
        }
        if verify_linear_certificate(source, certificate):
            return Result("certified_optimal", data, certificate, source)
    return Result(
        "certified_feasible", data, {"kind": "feasible", "point": point}, source
    )

@dataclass
class AlternativeResult:
    answers: tuple
    offsets: tuple
    branch_count: int

    def summary(self):
        if (
            len(self.answers) != self.branch_count
            or len(self.offsets) != self.branch_count
        ):
            return "unresolved", {}
        kinds, candidates = [], []
        for index, (answer, offset) in enumerate(zip(self.answers, self.offsets)):
            kind = answer.certificate.get("kind") if answer.verify() else "unresolved"
            kinds.append(kind)
            if kind in ("optimal", "feasible"):
                point = answer.certificate["point"]
                cost = dot(answer.source["c"], point) + offset
                candidates.append((cost, index, point))
        if all(kind == "infeasible" for kind in kinds):
            return "certified_infeasible", {}
        if not candidates:
            return "unresolved", {}
        cost, branch, point = min(candidates, key=lambda candidate: candidate[0])
        complete = all(kind in ("optimal", "infeasible") for kind in kinds)
        status = "certified_optimal" if complete else "certified_feasible"
        return status, dict(point=point, cost=cost, branch=branch)

    @property
    def status(self):
        return self.summary()[0]

    @property
    def data(self):
        return self.summary()[1]

    def verify(self):
        return self.status != "unresolved"

def solve_affine(problem, symbols):
    answers, offsets = [], []
    for branch in problem.feasible.branches:
        branch_problem = Problem(problem.games, branch, problem.cost, problem.names)
        A, b, E, e, c, offset = affine_system(branch_problem, symbols)
        answer = exact_lp(c, A, b, E, e)
        if "objective" in answer.data:
            answer.data["cost"] = answer.data["objective"] + offset
        answers.append(answer)
        offsets.append(offset)
    if len(answers) == 1:
        return answers[0]
    return AlternativeResult(
        tuple(answers), tuple(offsets), len(problem.feasible.branches)
    )

def equality_obstruction(region, symbols):
    if len(region.clauses) != 1:
        proofs = tuple(
            equality_obstruction(branch, symbols) for branch in region.branches
        )
        return proofs if all(proof is not None for proof in proofs) else None
    atoms = region.atoms(symbols)
    equalities = [v for v, relation in atoms if relation == "eq"]
    basis = sp.groebner(equalities, *symbols)
    if any(g.as_expr() == 1 for g in basis.polys):
        return basis, sp.S.One
    for value, relation in atoms:
        if relation == "ge":
            remainder = basis.reduce(value)[1]
            if not remainder.free_symbols and remainder < 0:
                return basis, remainder
    return None

def authority_obstruction(equations, symbols):
    matrix, rhs = sp.linear_eq_to_matrix(equations, symbols)
    for witness in matrix.T.nullspace():
        discrepancy = (witness.T * rhs)[0]
        if discrepancy != 0:
            return witness, discrepancy
    return None

def polynomial_term(expression, symbols, variables):
    import z3

    expression = sp.sympify(expression)
    if expression.has(sp.Float):
        raise ValueError("Use exact rational coefficients for polynomial decisions")
    polynomial = sp.Poly(expression, *symbols, domain=sp.QQ)
    return sum((
        z3.RealVal(str(coefficient))
        * prod(variable for variable, power in zip(variables, powers) for _ in range(power))
        for powers, coefficient in polynomial.terms()
    ), z3.RealVal(0))

def polynomial_formula(region, inputs, symbols, variables):
    import z3

    clauses = []
    for branch in region.branches:
        comparisons = []
        for expression, relation in branch.atoms(inputs):
            value = polynomial_term(expression, symbols, variables)
            if relation not in ("eq", "ge"):
                raise ValueError("Polynomial constraints require eq or ge")
            comparisons.append(value == 0 if relation == "eq" else value >= 0)
        clauses.append(z3.And(*comparisons))
    return z3.Or(*clauses)

def polynomial_decision(formula, timeout_ms):
    import z3

    solver = z3.SolverFor("NRA")
    solver.set(timeout=timeout_ms)
    solver.add(formula)
    try:
        status = solver.check()
    except z3.Z3Exception as error:
        return "unknown", str(error)
    if status == z3.sat:
        return "sat", solver.model()
    return str(status), solver.reason_unknown() if status == z3.unknown else None

def exact_algebraic_value(value):
    import z3

    if z3.is_rational_value(value):
        return sp.Rational(value.numerator_as_long(), value.denominator_as_long())
    if z3.is_algebraic_value(value):
        x = sp.Dummy("root")
        polynomial = sum(
            exact_algebraic_value(coefficient) * x ** degree
            for degree, coefficient in enumerate(value.poly())
        )
        return sp.CRootOf(polynomial, value.index() - 1)
    raise ValueError("The solver did not return an exact real algebraic value")

@dataclass
class PolynomialResult:
    status: str
    data: dict
    obligations: tuple
    timeout_ms: int

    def verify(self):
        return bool(self.obligations) and all(
            polynomial_decision(formula, self.timeout_ms)[0] == "unsat"
            for formula in self.obligations
        )

def solve_polynomial(problem, symbols, disturbances=None, minimize=True, timeout_ms=10000):
    import z3

    symbols = tuple(symbols)
    disturbances = dict(disturbances or {})
    all_symbols = symbols + tuple(disturbances)
    if not symbols or len(set(all_symbols)) != len(all_symbols):
        raise ValueError("Supply distinct command and disturbance symbols")
    variables = tuple(z3.FreshReal() for _ in all_symbols)
    commands = variables[:len(symbols)]
    # Evaluate the same Region on commands. Disturbance symbols remain free
    # in the supplied model and become universally quantified below.
    feasible = polynomial_formula(problem.feasible, symbols, all_symbols, variables)
    if disturbances:
        bounds = []
        for variable, (lower, upper) in zip(variables[len(symbols):], disturbances.values()):
            lower, upper = sp.sympify(lower), sp.sympify(upper)
            if not (lower.is_Rational and upper.is_Rational and lower <= upper):
                raise ValueError("Disturbance boxes need ordered rational endpoints")
            bounds.extend((variable >= z3.RealVal(str(lower)), variable <= z3.RealVal(str(upper))))
        feasible = z3.ForAll(
            variables[len(symbols):], z3.Implies(z3.And(*bounds), feasible)
        )
    cost_expression = np.asarray(problem.cost(exact(symbols)), dtype=object).item()
    cost = polynomial_term(cost_expression, symbols, commands)
    status, model = polynomial_decision(feasible, timeout_ms)
    if status == "unsat":
        return PolynomialResult("verified_infeasible", {}, (feasible,), timeout_ms)
    if status != "sat":
        return PolynomialResult("unresolved", {"reason": model}, (), timeout_ms)

    kind, reason = "feasible", None
    if minimize:
        alternatives = tuple(z3.FreshReal() for _ in symbols)
        substitution = tuple(zip(commands, alternatives))
        cheaper = z3.And(
            z3.substitute(feasible, *substitution),
            z3.substitute(cost, *substitution) < cost,
        )
        optimal = z3.And(feasible, z3.Not(z3.Exists(alternatives, cheaper)))
        optimum_status, optimum_model = polynomial_decision(optimal, timeout_ms)
        if optimum_status == "sat":
            kind, model = "optimal", optimum_model
        else:
            reason = "No attained minimum" if optimum_status == "unsat" else optimum_model

    point = tuple(model.eval(variable, model_completion=True) for variable in commands)
    substitution = tuple(zip(commands, point))
    actual_cost = z3.simplify(z3.substitute(cost, *substitution))
    obligations = (z3.Not(z3.substitute(feasible, *substitution)),)
    if kind == "optimal":
        obligations += (z3.And(feasible, cost < actual_cost),)
    data = dict(point=[exact_algebraic_value(value) for value in point],
                cost=exact_algebraic_value(actual_cost))
    if reason is not None:
        data["reason"] = reason
    answer = PolynomialResult("verified_" + kind, data, obligations, timeout_ms)
    if not answer.verify():
        return PolynomialResult("unresolved", {"reason": "Verification was inconclusive"}, (), timeout_ms)
    return answer

def interpolate_segment(start, end, values):
    return start + values[0] * (end - start)

def segment_ok(region, start, end):
    t = sp.Symbol("_t", real=True)
    path = Map(partial(interpolate_segment, start, end))
    clauses = [branch.atoms((t,)) for branch in region.pullback(path).branches]
    boundaries = {sp.S.Zero, sp.S.One}
    for clause in clauses:
        for expression, relation in clause:
            if expression != 0:
                roots = sp.Poly(expression, t).real_roots()
                boundaries.update(root for root in roots if 0 < root < 1)
    boundaries = sorted(boundaries)
    points = boundaries + [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
    for point in points:
        covered = False
        for clause in clauses:
            values = [
                (sp.simplify(expression.subs(t, point)), relation)
                for expression, relation in clause
            ]
            covered = covered or all(
                value == 0 if relation == "eq" else value >= 0
                for value, relation in values
            )
        if not covered:
            return False
    return True

def predecessor(states, actions, successors, target):
    choices = {}
    for state in states:
        allowed = []
        for action in actions(state):
            following = set(successors(state, action))
            if following and following <= target:
                allowed.append(action)
        choices[state] = tuple(allowed)
    return choices

def winning_layers(states, actions, successors, safe, goal):
    safe = set(states) & set(safe)
    layers, policies = [set(goal) & safe], []
    while True:
        choices = predecessor(safe, actions, successors, layers[-1])
        new = {x for x, options in choices.items() if options}
        enlarged = layers[-1] | new
        if enlarged == layers[-1]:
            return layers, policies
        policies.append({x: choices[x] for x in enlarged - layers[-1]})
        layers.append(enlarged)

def rollout(initial, policy, step, steps):
    states = [initial]
    for t in range(steps):
        state = states[-1]
        states.append(step(state, policy(t, state)))
    return states

def evaluate_policy_game(shape, initial, transition, reward, horizon, discount, theta):
    rows = []
    gamma = like(discount, theta)
    for profile in product(*(range(m) for m in shape)):
        distribution = like(initial, theta)
        value = theta.sum() * 0
        for t in range(horizon):
            P = transition(theta, t, profile)
            R = reward(theta, t, profile)
            flow = distribution[..., :, None] * P
            value = value + gamma**t * (R * flow[..., None, :, :]).sum(axis=(-2, -1))
            distribution = flow.sum(axis=-2)
        rows.append(value)
    payoffs = stack(rows, axis=-1)
    return Game(payoffs.reshape((*payoffs.shape[:-1], *shape)), shape)

def policy_game(shape, initial, transition, reward, horizon, discount):
    return Map(
        partial(
            evaluate_policy_game, shape, initial, transition, reward, horizon, discount
        )
    )

def observation_classes(games, measurements):
    classes = {}
    for i, game in enumerate(games):
        key = tuple(sp.simplify(f(game)) for f in measurements)
        classes.setdefault(key, []).append(i)
    return tuple(classes.values())

def zero_cost(command):
    return sp.S.Zero

def audit_observations(games, problems, measurements, symbols):
    results = []
    for indices in observation_classes(games, measurements):
        common = reduce(and_, (problems[i].feasible for i in indices))
        base = problems[indices[0]]
        problem = Problem(base.games, common, Map(zero_cost))
        results.append((indices, solve_affine(problem, symbols)))
    return results

def moment_candidates(n, degree):
    factors = [(S, p) for S in subsets(n) for p in range(n)]
    for d in range(1, degree + 1):
        for chosen in combinations_with_replacement(factors, d):
            yield moment(*chosen)

def refine_observations(games, problems, measurements, candidates, symbols):
    measurements = list(measurements)
    individual = [solve_affine(problem, symbols) for problem in problems]
    if any(
        answer.status != "certified_optimal" or not answer.verify()
        for answer in individual
    ):
        return measurements, individual, "individual_reachability_not_established"

    audit = audit_observations(games, problems, measurements, symbols)
    for candidate in candidates:
        if any(not answer.verify() for indices, answer in audit):
            return measurements, audit, "unresolved"
        failed = [
            indices
            for indices, answer in audit
            if answer.status == "certified_infeasible"
        ]
        if not failed:
            return measurements, audit, "sufficient"
        separates = any(
            len({sp.simplify(candidate(games[i])) for i in indices}) > 1
            for indices in failed
        )
        if separates:
            measurements.append(candidate)
            audit = audit_observations(games, problems, measurements, symbols)

    if any(not answer.verify() for indices, answer in audit):
        return measurements, audit, "unresolved"
    if all(
        answer.status == "certified_optimal" and answer.verify()
        for indices, answer in audit
    ):
        return measurements, audit, "sufficient"
    return measurements, audit, "candidate_library_exhausted"

def univariate_region(region, variable, domain):
    result = sp.S.EmptySet
    for branch in region.branches:
        part = domain
        for expression, relation in branch.atoms((variable,)):
            if not expression.has(variable):
                valid = expression == 0 if relation == "eq" else expression >= 0
                solution = domain if valid else sp.S.EmptySet
            elif relation == "eq":
                solution = sp.solveset(expression, variable, domain=sp.S.Reals)
            else:
                solution = sp.solve_univariate_inequality(
                    expression >= 0, variable, relational=False
                )
            part = part.intersect(solution)
        result = result.union(part)
    return result

def evaluate_rational_maps(numerators, denominators, values):
    return stack([p(values) / q(values) for p, q in zip(numerators, denominators)], axis=-1)

def rational_map(expressions, symbols):
    parts = [sp.fraction(sp.cancel(expression)) for expression in expressions]
    numerators = tuple(polynomial(sp.Poly(p, *symbols).terms()) for p, q in parts)
    denominators = tuple(polynomial(sp.Poly(q, *symbols).terms()) for p, q in parts)
    return Map(partial(evaluate_rational_maps, numerators, denominators))

def rational_curve_minimum(problem, curve, variable, domain):
    feasible = univariate_region(problem.feasible.pullback(curve), variable, domain)
    if not isinstance(feasible, sp.Interval) or feasible.left_open or feasible.right_open:
        raise ValueError("This procedure requires one closed bounded feasible interval")
    if feasible.start in (-sp.oo, sp.oo) or feasible.end in (-sp.oo, sp.oo):
        raise ValueError("Supply a bounded interval")
    cost = sp.cancel(problem.cost(curve(exact([variable]))))
    numerator, denominator = sp.fraction(cost)
    sp.Poly(numerator, variable)
    sp.Poly(denominator, variable)
    poles = sp.solveset(denominator, variable, domain=feasible)
    if poles != sp.S.EmptySet:
        raise ValueError("The cost must be continuous on the feasible interval")
    derivative = sp.fraction(sp.cancel(sp.diff(cost, variable)))[0]
    stationary = sp.S.EmptySet if derivative == 0 else sp.solveset(
        derivative, variable, domain=feasible
    )
    if not isinstance(stationary, sp.FiniteSet) and stationary != sp.S.EmptySet:
        raise ValueError("Stationary points were not completely isolated")
    points = set(stationary) | {feasible.start, feasible.end}
    values = [(sp.simplify(cost.subs(variable, p)), p) for p in points]
    minimum, parameter = min(values)
    command = curve(exact([parameter]))
    verified = problem.feasible.contains(command) and all(minimum <= value for value, p in values)
    return dict(parameter=parameter, command=command, cost=minimum, domain=feasible,
                stationary=stationary, cost_expression=cost, verified=bool(verified))

def find_axis_route(region, start, end):
    for order in permutations(range(len(start))):
        points = [exact(start)]
        for axis in order:
            point = points[-1].copy()
            point[axis] = end[axis]
            points.append(point)
        if all(segment_ok(region, a, b) for a, b in zip(points, points[1:])):
            return points
    return None

def transition_at_vertex(transition, disturbance, state_command):
    return transition(state_command, like(disturbance, state_command))

def box_preimage(target, transition, bounds):
    return reduce(and_, (
        target.pullback(Map(partial(transition_at_vertex, transition, vertex)))
        for vertex in product(*bounds)
    ), Region())

def scalar_command_bounds(region, state_symbols, command_symbol):
    lower, upper, state_conditions = [], [], []
    for expression, relation in region.atoms((*state_symbols, command_symbol)):
        if relation != "ge" or sp.Poly(expression, *state_symbols, command_symbol).total_degree() > 1:
            raise ValueError("Supply one conjunction of affine inequalities")
        coefficient = sp.diff(expression, command_symbol)
        remainder = expression.subs(command_symbol, 0)
        if coefficient == 0:
            state_conditions.append(polynomial(sp.Poly(remainder, *state_symbols).terms()))
            continue
        bound = polynomial(sp.Poly(-remainder / coefficient, *state_symbols).terms())
        (lower if coefficient > 0 else upper).append(bound)
    if not lower or not upper:
        raise ValueError("Supply both lower and upper command bounds")
    return Map(partial(evaluate_command_bounds, lower, upper, state_conditions))

def evaluate_command_bounds(lower, upper, state_conditions, state):
    lo = stack([bound(state) for bound in lower], axis=-1)
    hi = stack([bound(state) for bound in upper], axis=-1)
    if isinstance(state, torch.Tensor):
        lo, hi = lo.amax(dim=-1), hi.amin(dim=-1)
    else:
        lo, hi = sp.Max(*lo.flat), sp.Min(*hi.flat)
    if any(bool(condition(state) < 0) for condition in state_conditions) or bool(lo > hi):
        raise ValueError("No admissible command at this state")
    return stack([lo, hi], axis=-1)

def project_scalar_command(command, interval):
    lower, upper = interval
    if isinstance(command, torch.Tensor):
        return command.clamp(min=lower, max=upper)
    return max(lower, min(upper, command))

@dataclass(frozen=True)
class PIDState:
    parameters: object
    integral: object
    previous_error: object
    proposed: object
    applied: object

def pid_feedback(measure, bounds, gains, references, disturbances, back_calculation, time, state):
    error = references[time] - measure(state.parameters)
    integral = state.integral + gains[1] * error
    derivative = error - state.previous_error
    proposed = gains[0] * error + integral + gains[2] * derivative
    applied = project_scalar_command(proposed, bounds(state.parameters))
    integral = integral + back_calculation * (applied - proposed)
    return applied, integral, error, disturbances[time], proposed

def pid_transition(transition, state, decision):
    applied, integral, error, disturbance, proposed = decision
    state_command = stack([state.parameters[0], applied])
    parameters = transition(state_command, stack([disturbance]))
    return PIDState(parameters, integral, error, proposed, applied)

def pid_rollout(measure, bounds, transition, initial, references, disturbances, gains, back_calculation):
    zero = references[0] * 0
    initial_error = references[0] - measure(initial)
    state = PIDState(initial, zero, initial_error, zero, zero)
    policy = partial(pid_feedback, measure, bounds, gains, references, disturbances, back_calculation)
    return rollout(state, policy, partial(pid_transition, transition), len(references))

def pid_loss(simulate, measure, references, gains):
    states = simulate(gains)
    values = stack([measure(state.parameters) for state in states[:-1]])
    commands = stack([state.applied for state in states[1:]])
    return ((values - references) ** 2).mean() + .01 * (commands ** 2).mean()

def select_coordinates(indices, values):
    return values[..., list(indices)]

def restrict_controls(control, active):
    inactive = tuple(i for i in range(len(control.names)) if i not in active)
    allowed = control.allowed & eq(Map(partial(select_coordinates, inactive)), 0)
    return Control(control.parameters, allowed, control.cost, control.names)

def find_minimal_permissions(family, control, target, active, candidates, symbols):
    trials = []
    for size in range(len(candidates) + 1):
        successful = []
        for added in combinations(candidates, size):
            restricted = restrict_controls(control, tuple(active) + added)
            problem = design(family, restricted, target)
            answer = solve_affine(problem, symbols)
            trials.append((added, answer))
            if answer.status == "certified_optimal" and answer.verify():
                successful.append((added, problem, answer))
        if successful:
            smaller_fail = all(
                answer.status == "certified_infeasible" and answer.verify()
                for added, answer in trials if len(added) < size
            )
            return successful, trials, smaller_fail
    return [], trials, False

def pure_equilibria(game):
    return tuple(profile for profile in product(*(range(size) for size in game.shape))
                 if ge(deviation_gaps(profile), 0).contains(game))

def best_response_step(game, profile, actor):
    values = [game.payoffs[(actor, *profile[:actor], action, *profile[actor + 1:])]
              for action in range(game.shape[actor])]
    choice = profile[actor] if values[profile[actor]] == max(values) else values.index(max(values))
    return tuple(choice if i == actor else action for i, action in enumerate(profile))

def round_robin(time, state):
    return time % len(state)

def scheduled_responses(games, state, command):
    profile, remaining = state
    game = games(exact([command]))
    following = []
    for actor in range(len(profile)):
        if remaining & (1 << actor):
            updated = best_response_step(game, profile, actor)
            mask = remaining & ~(1 << actor)
            following.append((updated, mask or (1 << len(profile)) - 1))
    return following

