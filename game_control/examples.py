# Example 1: Applying an Intervention and Changing a Game Type
def add_payoff_terms(changes, payoffs, theta):
    return payoffs + changes(theta)


x, y = np.meshgrid(exact([-1, 1]), exact([-1, 1]), indexing="ij")
joint_score = x * y
zero_table = exact(np.zeros((2, 2), dtype=int))
binary_base = np.stack([joint_score, zero_table])
binary_coefficient = np.stack([zero_table, joint_score])
binary_changes = polynomial([((1,), binary_coefficient)])
binary_family = Game(binary_base, (2, 2)).with_payoff_rule(
    partial(add_payoff_terms, binary_changes)
)

binary_start = exact([-1])
binary_parameters = Map(lambda v: like(binary_start, v) + v)
binary_allowed = ge(Map(lambda v: v), 0) & ge(Map(lambda v: 2 - v), 0)
binary_control = Control(
    binary_parameters, binary_allowed, Map(lambda v: v.sum(axis=-1)),
    ("joint_weight_change",),
)
binary_measurements = gram((0, 1), 0, 1)
binary_desired_type = ge(Map(lambda q: q), sp.Rational(1, 4))
binary_command = exact([2])

binary_games = binary_control.parameters.then_apply(binary_family)
binary_types = binary_games.then_apply(binary_measurements)
binary_successful = binary_allowed & binary_desired_type.pullback(binary_types)
binary_before = binary_family(binary_start)
binary_after = binary_games(binary_command)
binary_before_measurement = binary_measurements(binary_before)
binary_after_measurement = binary_types(binary_command)
binary_intervention_valid = binary_successful.contains(binary_command)

binary_command_symbol = sp.symbols("v", real=True)
binary_symbolic_type = binary_types(exact([binary_command_symbol]))
binary_sensitivity = torch.func.jacfwd(binary_types)(
    torch.tensor([0.0], dtype=torch.float64)
)

project_shape = (2, 3, 2)
effort_menus = (exact([0, 1]), exact([0, 1, 2]), exact([0, 2]))
efforts = np.meshgrid(*effort_menus, indexing="ij")
project_costs = np.stack(efforts) ** 2 * sp.Rational(1, 4)
project_base = -project_costs
project_coefficients = []

for i, j in combinations(range(3), 2):
    output = efforts[i] * efforts[j]
    project_base[j] += output
    transfer = exact(np.zeros((3,) + project_shape, dtype=int))
    transfer[i], transfer[j] = output, -output
    project_coefficients.append(transfer)

joint_output = np.prod(efforts, axis=0)
project_base[2] += joint_output
for i in (0, 1):
    transfer = exact(np.zeros((3,) + project_shape, dtype=int))
    transfer[i], transfer[2] = joint_output, -joint_output
    project_coefficients.append(transfer)

project_terms = [
    (tuple(int(i == j) for j in range(5)), coefficient)
    for i, coefficient in enumerate(project_coefficients)
]
project_changes = polynomial(project_terms)
project_family = Game(project_base, project_shape).with_payoff_rule(
    partial(add_payoff_terms, project_changes)
)

project_start = exact([sp.Rational(1, 2)] * 3 + [1, 0])
project_actuator = exact([[0, 0], [0, 0], [0, 0], [-1, -1], [1, 0]])
project_parameters = Map(
    lambda command: like(project_start, command)
    + command @ like(project_actuator, command).T
)
project_command = exact([sp.Rational(1, 3), sp.Rational(1, 3)])

joint_grams = tuple(
    tuple(gram((0, 1, 2), i, j) for j in range(3))
    for i in range(3)
)
joint_alignment = Map(lambda game: stack([
    sum(entry(game) for entry in row) for row in joint_grams
], axis=-1))
collective_joint_energy = Map(lambda game: joint_alignment(game).sum(axis=-1))
project_measurements = joint_alignment
project_threshold = sp.Rational(1, 24)
project_desired_type = ge(Map(lambda z: z), project_threshold)
project_target = project_desired_type.pullback(project_measurements)
project_allowed = (
    ge(Map(lambda v: v), 0)
    & ge(Map(lambda v: 1 - v.sum(axis=-1)), 0)
)
project_control = Control(
    project_parameters,
    project_allowed,
    Map(lambda v: v.sum(axis=-1)),
    ("weight_to_second", "weight_to_third"),
)

project_games = project_control.parameters.then_apply(project_family)
project_types = project_games.then_apply(project_measurements)
project_successful = (
    project_allowed & project_desired_type.pullback(project_types)
)
project_before = project_family(project_start)
project_after = project_games(project_command)
before_measurements = project_measurements(project_before)
after_measurements = project_types(project_command)
project_initially_in_target = project_target.contains(project_before)
project_target_reached = project_target.contains(project_after)
project_intervention_valid = project_successful.contains(project_command)

numerical_project_command = torch.tensor([.2, .1], dtype=torch.float64)
project_sensitivity = torch.func.jacfwd(project_types)(numerical_project_command)


# Example 2: Inverse Design: Feasible Regions and Minimum-Cost Interventions
project_problem = design(project_family, project_control, project_target)
project_command_symbols = sp.symbols("v1 v2", real=True)
project_answer = solve_affine(project_problem, project_command_symbols)
project_solution = project_problem.select(exact(project_answer.data["point"]))
project_solution_valid = project_answer.verify() and project_solution.verify()

project_intervention = project_problem.select(project_command)
project_intervention_valid = project_intervention.verify()


# Example 3: Constrained Reachability: Safe Paths and Obstructions
project_safe_games = (
    ge(Map(lambda game: -project_measurements(game)[..., 1]), -sp.Rational(1, 48))
    | ge(Map(lambda game: project_measurements(game)[..., 2]), sp.Rational(1, 24))
)
project_safe_commands = project_allowed & project_safe_games.pullback(project_games)
route_start = exact([0, 0])
route_goal = project_solution.command
project_route = find_axis_route(project_safe_commands, route_start, route_goal)
project_direct_safe = segment_ok(project_safe_commands, route_start, route_goal)
project_route_valid = project_route is not None and project_target.contains(project_games(project_route[-1]))
matched_changes = Map(lambda t: stack([t[..., 0], t[..., 0]], axis=-1))
path_parameter = sp.Symbol("t", real=True)
matched_safe_set = univariate_region(
    project_safe_commands.pullback(matched_changes), path_parameter, sp.Interval(0, sp.Rational(1, 2))
)
matched_target_set = univariate_region(
    project_successful.pullback(matched_changes), path_parameter, sp.Interval(0, sp.Rational(1, 2))
)


# Example 4: Robust Control: Maintaining a Desired Game Type
shared_project_parameters = Map(lambda t:
    like(project_start, t)
    + t[..., :1] * like([0, 0, 0, -2, 1], t))
shared_project_games = shared_project_parameters.then_apply(project_family)
shared_target = project_target.pullback(shared_project_games)
drift_symbol = sp.Symbol("t", real=True)
drift_interval = univariate_region(shared_target, drift_symbol, sp.Interval(0, sp.Rational(1, 2)))
drift_radius = sp.Rational(1, 48)
drift_limit = sp.Rational(1, 24)


def drift_transition(state_command, disturbance):
    return stack([state_command[..., 0] + state_command[..., 1] + disturbance[..., 0]], axis=-1)


robust_moves = box_preimage(shared_target, drift_transition, ((-drift_radius, drift_radius),))
robust_moves = robust_moves & ge(Map(lambda tv: tv[..., 1]), -drift_limit)
robust_moves = robust_moves & ge(Map(lambda tv: -tv[..., 1]), -drift_limit)


def drift_pair(current, command):
    return stack([like(current, command) + command[..., 0] * 0, command[..., 0]], axis=-1)


def adjusted_parameter(current, command):
    return stack([like(current, command) + command[..., 0]], axis=-1)


absolute_above_positive = Map(lambda vc: vc[..., 1] - vc[..., 0])
absolute_above_negative = Map(lambda vc: vc[..., 1] + vc[..., 0])
absolute_cost = Map(lambda vc: vc[..., 1:].sum(axis=-1))


def drift_problem(current):
    parameters = Map(partial(adjusted_parameter, current))
    allowed = robust_moves.pullback(Map(partial(drift_pair, current)))
    allowed = allowed & ge(absolute_above_positive, 0)
    allowed = allowed & ge(absolute_above_negative, 0)
    control = Control(parameters, allowed, absolute_cost, ("adjustment", "absolute_cost"))
    return design(shared_project_games, control, project_target)


def drift_policy(current):
    problem = drift_problem(current)
    answer = solve_affine(problem, sp.symbols("v c", real=True))
    return answer, problem.select(exact(answer.data["point"]))


drift_answer, drift_selected = drift_policy(sp.Rational(1, 4))
drift_verified = drift_answer.verify() and drift_selected.verify()
drift_command_atoms = robust_moves.atoms(sp.symbols("t v", real=True))

pid_symbols = sp.symbols("t v", real=True)
pid_bounds = scalar_command_bounds(robust_moves, pid_symbols[:1], pid_symbols[1])
pid_measurement = Map(lambda game: sum(f(game) for f in joint_grams[1]))
pid_output = shared_project_games.then_apply(pid_measurement)
pid_reference_parameters = torch.tensor([.30] * 16 + [.35] * 16 + [.28] * 16, dtype=torch.float64)
pid_references = pid_output(pid_reference_parameters[:, None])
pid_disturbances = torch.tensor([-float(drift_radius)] * 20 + [float(drift_radius)] * 28, dtype=torch.float64)
pid_initial = torch.tensor([.25], dtype=torch.float64)
pid_initial_gains = torch.tensor([6., .6, 1.5], dtype=torch.float64)
pid_back_calculation = .5
pid_simulate = partial(pid_rollout, pid_output, pid_bounds, drift_transition,
                       pid_initial, pid_references, pid_disturbances,
                       back_calculation=pid_back_calculation)
pid_states = pid_simulate(pid_initial_gains)
pid_values = stack([pid_output(state.parameters) for state in pid_states])
pid_commands = stack([state.applied for state in pid_states[1:]])
pid_proposals = stack([state.proposed for state in pid_states[1:]])
pid_loss_before = pid_loss(pid_simulate, pid_output, pid_references, pid_initial_gains)

# The comparison controller respects only the physical actuator bounds.
pid_actuator_bounds = Map(lambda state: like([-drift_limit, drift_limit], state))
pid_unfiltered = pid_rollout(pid_output, pid_actuator_bounds, drift_transition,
                             pid_initial, pid_references, pid_disturbances,
                             pid_initial_gains, pid_back_calculation)

pid_gains = pid_initial_gains.clone().requires_grad_()
pid_optimizer = torch.optim.Adam([pid_gains], lr=.12)
for iteration in range(50):
    pid_optimizer.zero_grad()
    pid_objective = pid_loss(pid_simulate, pid_output, pid_references, pid_gains)
    pid_objective.backward()
    pid_optimizer.step()
    with torch.no_grad():
        pid_gains.clamp_(min=0)
pid_tuned_gains = pid_gains.detach()
pid_tuned_states = pid_simulate(pid_tuned_gains)
pid_loss_after = pid_loss(pid_simulate, pid_output, pid_references, pid_tuned_gains)
pid_tuned_values = stack([pid_output(state.parameters) for state in pid_tuned_states])

neural_shape = (2, 2, 2)
neural_signs = exact(np.meshgrid(*([[-1, 1]] * 3), indexing="ij"))
neural_observation = exact([1, 1])
neural_tolerances = exact([sp.Rational(1, 4)] * 2)


def policy_preactivations(theta, observation):
    coefficients = like(theta, observation)
    coefficients = coefficients.reshape((*coefficients.shape[:-1], 3, 3))
    return coefficients[..., :2] @ observation + coefficients[..., 2]


def policy_game_from_gains(gains):
    signals = gains[..., :, None, None, None] * like(neural_signs, gains)
    joint_output = prod(signals[..., i, :, :, :] for i in range(3))
    payoffs = joint_output[..., None, :, :, :] / 3 - signals**2 / 4
    return Game(payoffs, neural_shape)


def neural_policy_game(observation, theta):
    values = policy_preactivations(theta, like(observation, theta))
    if isinstance(values, torch.Tensor):
        gains = torch.relu(values)
    else:
        gains = exact([sp.Max(0, x) for x in values.flat]).reshape(values.shape)
    return policy_game_from_gains(gains)


neural_family = Map(partial(neural_policy_game, neural_observation))
neural_grams = tuple(gram((0, 1, 2), p, q) for p, q in combinations(range(3), 2))
neural_measurements = Map(lambda game: stack([f(game) for f in neural_grams], axis=-1))
neural_target = ge(neural_measurements, sp.Rational(1, 9))

neural_start = exact([
    1, 0, -sp.Rational(1, 2),
    sp.Rational(1, 2), sp.Rational(1, 2), -sp.Rational(1, 2),
    0, 1, -sp.Rational(1, 2),
])
neural_parameters = Map(lambda v: like(neural_start, v) + v)
neural_allowed = ge(Map(lambda v: v), -1) & ge(Map(lambda v: -v), -1)
neural_control = Control(
    neural_parameters, neural_allowed, Map(lambda v: (v**2).sum(axis=-1))
)
neural_problem = design(neural_family, neural_control, neural_target)
neural_command = exact([0, 0, 1] * 3)
neural_selected = neural_problem.select(neural_command)
neural_command_verified = neural_selected.verify()
neural_final = neural_parameters(neural_command)
neural_before = neural_measurements(neural_family(neural_start))
neural_after = neural_measurements(neural_selected.game())
neural_sensitivity = torch.func.jacfwd(
    neural_problem.games.then_apply(neural_measurements)
)(torch.tensor(np.asarray(neural_command, dtype=float), dtype=torch.float64))

def scaled_policy_observation(error, scale):
    return like(neural_observation, scale) + scale[0] * like(neural_tolerances * error, scale)


def linear_policy_game(theta, observation):
    return policy_game_from_gains(policy_preactivations(theta, observation))


neural_errors = sp.symbols("xi_0 xi_1", real=True)
neural_scale = sp.Symbol("lambda", real=True)
neural_uncertain_observation = Map(partial(scaled_policy_observation, exact(neural_errors)))
neural_activation_region = ge(Map(partial(policy_preactivations, neural_final)), 0)
neural_margin_control = Control(
    neural_uncertain_observation,
    ge(Map(lambda scale: scale), 0)
    & neural_activation_region.pullback(neural_uncertain_observation),
    Map(lambda scale: -scale[0]),
    ("disturbance_scale",),
)
neural_margin_problem = design(
    Map(partial(linear_policy_game, neural_final)),
    neural_margin_control,
    neural_target,
)
neural_margin_answer = solve_polynomial(
    neural_margin_problem,
    (neural_scale,),
    disturbances={error: (-1, 1) for error in neural_errors},
)
neural_safety_factor = neural_margin_answer.data["point"][0]
neural_margin_verified = neural_margin_answer.verify()


# Example 5: Control Authority: Impossibility Certificates and Missing Instruments
pair_actuator = exact([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
full_actuator = np.concatenate([pair_actuator, project_actuator], axis=1)
full_parameters = Map(lambda v: like(project_start, v) + v @ like(full_actuator, v).T)
full_bounds = (
    ge(Map(lambda v: v[..., :3]), -sp.Rational(1, 2))
    & ge(Map(lambda v: -v[..., :3]), -sp.Rational(1, 2))
    & ge(Map(lambda v: v[..., 3:]), 0)
    & ge(Map(lambda v: 1 - v[..., 3:].sum(axis=-1)), 0)
)
full_control = Control(full_parameters, full_bounds, Map(lambda v: v[..., 3:].sum(axis=-1)),
                       ("pair01", "pair02", "pair12", "joint_to_second", "joint_to_third"))
permission_symbols = sp.symbols("a01 a02 a12 v1 v2", real=True)
permission_solutions, permission_trials, permissions_minimal = find_minimal_permissions(
    project_family, full_control, project_target, (0, 1, 2), (3, 4), permission_symbols
)
added_permissions, repaired_problem, repaired_answer = permission_solutions[0]
repaired_intervention = repaired_problem.select(exact(repaired_answer.data["point"]))
permission_verified = permissions_minimal and repaired_answer.verify() and repaired_intervention.verify()
restricted_game = project_family(full_parameters(exact([*permission_symbols[:3], 0, 0])))
conserved_measurements = project_measurements(restricted_game)


# Example 6: Control Equivalence: Different Interventions with the Same Effect
product_transfers = Map(lambda command: command[..., :1] * command[..., 1:])
product_parameters = product_transfers.then_apply(project_parameters)
product_allowed = (
    ge(Map(lambda v: v), 0) & ge(Map(lambda v: 1 - v), 0)
    & project_allowed.pullback(product_transfers)
)
product_control = Control(product_parameters, product_allowed,
                          Map(lambda v: v.sum(axis=-1)), ("gate", "second_gain", "third_gain"))
product_problem = design(project_family, product_control, project_target)
product_values = product_problem.games.then_apply(project_measurements)
reference_command = exact([1, sp.Rational(1, 4), sp.Rational(1, 4)])
invariant_fiber = product_problem.equivalent_to(reference_command, project_measurements)
product_symbols = sp.symbols("m p1 p2", real=True)
fiber_equations = [expression for expression, relation in invariant_fiber.atoms(product_symbols) if relation == "eq"]
fiber_solutions = sp.solve(fiber_equations, product_symbols[1:], dict=True)
fiber_curve = rational_map([product_symbols[0], *[fiber_solutions[0][s] for s in product_symbols[1:]]], (product_symbols[0],))
fiber_minimum = rational_curve_minimum(
    product_problem, fiber_curve, product_symbols[0], sp.Interval(0, 1)
)
fiber_command = product_problem.select(fiber_minimum["command"])
fiber_verified = fiber_minimum["verified"] and fiber_command.verify()
local_command = torch.tensor([.5, .5, .5], dtype=torch.float64)
local_direction, local_nullspace, local_residual = local_controls(
    product_values, local_command, torch.zeros(3, dtype=torch.float64)
)
local_jacobian = torch.func.jacfwd(product_values)(local_command)
finite_null_step = local_command + .1 * local_nullspace[:, 0]
finite_measurement_change = product_values(finite_null_step) - product_values(local_command)
zero_product_jacobian = torch.func.jacfwd(product_values)(torch.zeros(3, dtype=torch.float64))


# Example 7: Equilibrium and Dynamics: Behavioral Consequences of Changing Payoffs
response_game = project_solution.game()
response_equilibria = pure_equilibria(response_game)
response_initials = ((0, 0, 0), (1, 0, 1))
response_traces = [
    rollout(profile, round_robin, partial(best_response_step, response_game), 6)
    for profile in response_initials
]
response_target_verified = project_target.contains(response_game)
response_coordinates = project_measurements(response_game)


# Example 8: Temporary Control: Reaching a State Where Intervention Can End
activation_effect = exact(np.zeros((3,) + project_shape, dtype=int))
activation_effect[0] = efforts[0]
activation_change = polynomial([((1,), activation_effect)])
activation_games = response_game.with_payoff_rule(partial(add_payoff_terms, activation_change))
activation_levels = (sp.S.Zero, sp.Rational(1, 2))
activation_states = tuple(product(product(*(range(size) for size in project_shape)), range(1, 8)))
activation_successors = partial(scheduled_responses, activation_games)


def baseline_choices(state):
    return (sp.S.Zero,)


def activation_choices(state):
    return activation_levels


activation_safe = set(activation_states)
activation_goal = {state for state in activation_states if state[0] == (1, 2, 1)}
baseline_layers, baseline_policy = winning_layers(
    activation_states, baseline_choices, activation_successors, activation_safe, activation_goal
)
release_profiles = {
    profile for profile in product(*(range(size) for size in project_shape))
    if all((profile, phase) in baseline_layers[-1] for phase in range(1, 8))
}
release_states = {state for state in activation_states if state[0] in release_profiles}
activation_layers, activation_policy = winning_layers(
    activation_states, activation_choices, activation_successors, activation_safe, release_states
)
activation_start = ((0, 0, 0), 7)
activation_rank = next(i for i, layer in enumerate(activation_layers) if activation_start in layer)
activation_preserves_target = all(project_target.contains(activation_games(exact([s]))) for s in activation_levels)
activation_measurements = [project_measurements(activation_games(exact([s]))) for s in activation_levels]


# Example 9: Distributed Control: Invariant Targets in Larger Games
network_shape = (2,) * 6
network_actions = np.meshgrid(*[exact([-1, 1])] * 6, indexing="ij")
network_edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5))
network_weights = exact([1, sp.Rational(4, 5), sp.Rational(1, 2), sp.Rational(9, 10), sp.Rational(1, 5), sp.Rational(3, 5)])
network_base = -np.stack(network_actions) * sp.Rational(1, 4)
network_terms = []
for edge_index, (i, j) in enumerate(network_edges):
    feature = network_actions[i] * network_actions[j]
    network_base[j] += feature
    coefficient = exact(np.zeros((6,) + network_shape, dtype=int))
    coefficient[i], coefficient[j] = feature, -feature
    powers = tuple(int(k == edge_index) for k in range(6))
    network_terms.append((powers, coefficient))
network_base += exact([1, -1, 0, 1, 0, -1])[:, None, None, None, None, None, None] * (
    network_actions[0] * network_actions[2] * network_actions[4] / 3
)
network_change = polynomial(network_terms)
network_family = Game(network_base, network_shape).with_payoff_rule(partial(add_payoff_terms, network_change))
network_rows = tuple(tuple(gram(edge, p, q) for q in range(6)) for edge in network_edges for p in edge)
network_measurements = Map(lambda game: stack([sum(f(game) for f in row) for row in network_rows], axis=-1))
network_target = ge(network_measurements, sp.Rational(1, 4))
network_parameters = Map(lambda v: like(network_weights, v) - v[..., :6])
network_bounds = (
    ge(network_parameters, 0) & ge(Map(lambda v: 1 - network_parameters(v)), 0)
    & ge(Map(lambda v: v[..., 6:] - v[..., :6]), 0)
    & ge(Map(lambda v: v[..., 6:] + v[..., :6]), 0)
)
network_control = Control(network_parameters, network_bounds, Map(lambda v: v[..., 6:].sum(axis=-1)),
                          tuple([f"edge_{i}" for i in range(6)] + [f"cost_{i}" for i in range(6)]))
network_problem = design(network_family, network_control, network_target)
network_symbols = sp.symbols("v0:6 c0:6", real=True)
network_answer = solve_affine(network_problem, network_symbols)
network_selected = network_problem.select(exact(network_answer.data["point"]))
uniform_parameters = Map(lambda v: like(network_weights, v) - v[..., :1])
uniform_bounds = ge(uniform_parameters, 0) & ge(Map(lambda v: 1 - uniform_parameters(v)), 0)
uniform_control = Control(uniform_parameters, uniform_bounds, Map(lambda v: v.sum() * 0), ("uniform_change",))
uniform_answer = solve_affine(design(network_family, uniform_control, network_target), (sp.Symbol("v", real=True),))
network_verified = network_answer.verify() and network_selected.verify() and uniform_answer.verify()
network_sensitivity = torch.func.jacfwd(network_problem.games.then_apply(network_measurements))(
    torch.zeros(12, dtype=torch.float64)
)

policy_shape = (3, 3, 3, 3)
operating_values = exact([1, sp.Rational(6, 5), sp.Rational(4, 5), sp.Rational(11, 10)])
maintenance_costs = exact([sp.Rational(2, 5), sp.Rational(3, 5), sp.Rational(4, 5), 1])


def policy_actions(profile, capacity):
    return tuple(int(policy == 0 or (policy == 2 and capacity >= 3)) for policy in profile)


def facility_transition(theta, time, profile):
    matrix = exact(np.zeros((5, 5), dtype=int))
    for capacity in range(5):
        active = sum(policy_actions(profile, capacity))
        for wear, probability in ((0, sp.Rational(3, 4)), (1, sp.Rational(1, 4))):
            following = max(0, min(4, capacity + 4 - 2 * active - wear))
            matrix[capacity, following] += probability
    return like(matrix, theta)


def facility_reward(theta, time, profile):
    base = exact(np.zeros((4, 5, 5), dtype=int))
    maintenance = exact(np.zeros((4, 5, 5), dtype=int))
    for capacity in range(5):
        actions = policy_actions(profile, capacity)
        for individual, action in enumerate(actions):
            if action:
                base[individual, capacity, :] = operating_values[individual] * sp.Rational(capacity, max(1, sum(actions)))
            else:
                base[individual, capacity, :] = -maintenance_costs[individual]
                maintenance[individual, capacity, :] = 1
    return like(base, theta) + theta[..., :, None, None] * like(maintenance, theta)


compiled_family = policy_game(policy_shape, exact([0, 0, 0, 0, 1]), facility_transition,
                              facility_reward, 5, sp.Rational(9, 10))
# Fixed transitions and affine rewards make this exact coefficient cache valid.
policy_base = compiled_family(exact([0, 0, 0, 0]))
policy_coefficients = [compiled_family(exact([int(i == j) for i in range(4)])).payoffs - policy_base.payoffs for j in range(4)]
policy_change = polynomial([(tuple(int(i == j) for i in range(4)), coefficient)
                            for j, coefficient in enumerate(policy_coefficients)])
policy_family = policy_base.with_payoff_rule(partial(add_payoff_terms, policy_change))
policy_parameters = Map(lambda r: r[..., :1] * like([1, 1, 1, 1], r))
policy_measurements_list = tuple(gram((0, 1, 2, 3), i, j) for i, j in combinations(range(4), 2))
policy_measurements = Map(lambda game: stack([f(game) for f in policy_measurements_list], axis=-1))
policy_target = ge(policy_measurements, sp.Rational(1, 20))
policy_control = Control(policy_parameters,
                         ge(Map(lambda r: r), 0) & ge(Map(lambda r: 4 - r), 0),
                         Map(lambda r: 4 * r[..., 0]), ("maintenance_rate",))
policy_problem = design(policy_family, policy_control, policy_target)
rate_symbol = sp.Symbol("r", real=True)
policy_rate_region = univariate_region(policy_problem.feasible, rate_symbol, sp.Interval(0, 4))
policy_command = exact([policy_rate_region.start])
policy_selected = policy_problem.select(policy_command)
policy_verified = policy_selected.verify()
policy_polynomials = [sp.factor(z) for z in policy_measurements(policy_problem.games(exact([rate_symbol])))]
policy_interior = exact([(policy_rate_region.start + policy_rate_region.end) / 2])
policy_interior_game = policy_problem.games(policy_interior)
policy_adaptive_equilibrium = ge(deviation_gaps((2, 2, 2, 2)), sp.Rational(1, 20)).contains(policy_interior_game)
policy_sensitivity = torch.func.jacfwd(policy_problem.games.then_apply(policy_measurements))(
    torch.tensor([float(policy_interior[0])], dtype=torch.float64)
)


# Example 10: Information Sufficiency
information_reference = project_solution.game()
information_joint = effect(information_reference, (0, 1, 2))
information_change = polynomial([((0,), -information_joint), ((1,), information_joint)])
information_family = information_reference.with_payoff_rule(partial(add_payoff_terms, information_change))
information_games = [information_family(exact([j])) for j in (1, -1)]
reference_witness = moment(((0,), 0), ((1, 2), 0), ((0, 1, 2), 0))
information_target = eq(reference_witness, reference_witness(information_reference))


def coupling_update(start, command):
    return like([start], command) + command


information_allowed = ge(Map(lambda v: v), 0) & ge(Map(lambda v: 2 - v), 0)
information_controls = [Control(Map(partial(coupling_update, j)), information_allowed,
                               Map(lambda v: v.sum(axis=-1)), ("joint_gain_change",)) for j in (1, -1)]
information_problems = [design(information_family, control, information_target) for control in information_controls]
information_symbols = (sp.Symbol("v", real=True),)
quadratic_observations = [gram(S, p, q) for S in subsets(3) for p in range(3) for q in range(p, 3)]
information_audit = audit_observations(information_games, information_problems, quadratic_observations, information_symbols)
refined_observations, refined_audit, information_status = refine_observations(
    information_games, information_problems, quadratic_observations,
    islice(moment_candidates(3, 3), 3000), information_symbols
)
information_commands = [answer.data["point"] for indices, answer in refined_audit]
information_separator_values = [refined_observations[-1](game) for game in information_games]

