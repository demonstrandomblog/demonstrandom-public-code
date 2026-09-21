# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Compose continuous-state games from policies, dynamics, and payoff rules.

StateSpace maps named agent variables to slices of the joint physical tensor.
Arena evaluates policies once per fixed time step and integrates dx/dt=f(x,u)
with Euler or RK4. Constraints project the endpoint sequentially. The normal-
form converter simulates each named policy combination to produce payoffs.
The bundled example contains two hunters and three moving prey agents.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
import matplotlib.pyplot as plt
import itertools

# ============================================================================
# LAYER 1: STATE SPACE
# ============================================================================

@dataclass
class StateSpec:
    """Symbolic state specification"""
    names: List[str]  # e.g., ['x', 'y', 'vx', 'vy']

    def dim(self) -> int:
        return len(self.names)

class StateSpace:
    """Joint state = agent states + optional shared state"""

    def __init__(self,
                 agents: List['Agent'],
                 shared_spec: Optional[StateSpec] = None):
        """
        Build state space from agent specifications.

        Args:
            agents: list of Agent objects with state_spec
            shared_spec: optional shared state specification
        """
        self.agents = agents
        self.agent_names = [a.name for a in agents]
        self.agent_dims = {a.name: a.state_spec.dim() for a in agents}
        self.shared_dim = shared_spec.dim() if shared_spec else 0

        self.slices, self.dim = self._build_state_indexing()

    def _build_state_indexing(self) -> Tuple[Dict[str, slice], int]:
        """
        Build slice indexing for joint state vector.

        Returns:
            (slices, total_dim)
        """
        slices = {}
        offset = 0

        for agent_name in self.agent_names:
            dim = self.agent_dims[agent_name]
            slices[agent_name] = slice(offset, offset + dim)
            offset += dim

        if self.shared_dim > 0:
            slices['shared'] = slice(offset, offset + self.shared_dim)
            offset += self.shared_dim

        return slices, offset

    def zero(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def get_state(self, state: torch.Tensor, agent: str) -> torch.Tensor:
        return state[self.slices[agent]]

    def get_shared(self, state: torch.Tensor) -> torch.Tensor:
        if self.shared_dim > 0:
            return state[self.slices['shared']]
        return torch.tensor([])

    def set_state(self, state: torch.Tensor, agent: str, value: torch.Tensor):
        state[self.slices[agent]] = value

@dataclass
class GameState:
    """Snapshot of game state; tensors and metadata remain mutable."""

    # Core state
    physical_state: torch.Tensor  # The flat state vector
    time: float

    # Accumulated info
    cumulative_payoffs: Dict[str, float]

    # Optional: for debugging/analysis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> 'GameState':
        """Copy physical state and top-level dictionaries for branching."""
        return GameState(
            physical_state=self.physical_state.clone(),
            time=self.time,
            cumulative_payoffs=self.cumulative_payoffs.copy(),
            metadata=self.metadata.copy()
        )

    def with_state(self, new_physical_state: torch.Tensor) -> 'GameState':
        """Return new GameState with updated physical state"""
        return GameState(
            physical_state=new_physical_state,
            time=self.time,
            cumulative_payoffs=self.cumulative_payoffs,
            metadata=self.metadata
        )

    def with_time(self, new_time: float) -> 'GameState':
        """Return new GameState with updated time"""
        return GameState(
            physical_state=self.physical_state,
            time=new_time,
            cumulative_payoffs=self.cumulative_payoffs,
            metadata=self.metadata
        )

    def add_payoffs(self, step_payoffs: Dict[str, float]) -> 'GameState':
        """Return new GameState with updated payoffs"""
        new_payoffs = self.cumulative_payoffs.copy()
        for agent, reward in step_payoffs.items():
            new_payoffs[agent] = new_payoffs.get(agent, 0.0) + reward

        return GameState(
            physical_state=self.physical_state,
            time=self.time,
            cumulative_payoffs=new_payoffs,
            metadata=self.metadata
        )

# ============================================================================
# LAYER 2: OBSERVATION MODEL
# ============================================================================

class ObservationModel(ABC):

    @abstractmethod
    def observe(self,
                state: torch.Tensor,
                agent: str,
                cumulative_payoff: Optional[float] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def obs_dim(self, agent: str) -> int:
        pass

# ============================================================================
# LAYER 3: DYNAMICS (Differentiable)
# ============================================================================

class Dynamics(ABC):
    """State evolution: dx/dt = f(x, u)"""

    @abstractmethod
    def derivative(self,
                   state: torch.Tensor,
                   controls: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns dstate/dt.
        Must be differentiable w.r.t. state and controls
        """
        pass

###############################
# LAYER 3.5: CONSTRAINTS (Between dynamics and integration)
###############################

class Constraint(ABC):
    """Abstract constraint on state"""

    @abstractmethod
    def violated(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns violation amount (0 = satisfied, >0 = violated).
        Must be differentiable for gradient-based optimization.
        """
        pass

    @abstractmethod
    def project(self, state: torch.Tensor) -> torch.Tensor:
        """Project state onto feasible set"""
        pass

class BoundaryConstraint(Constraint):
    """Box boundaries [x_min, x_max] × [y_min, y_max]"""

    def __init__(self, state_space: StateSpace, bounds: Dict[str, Tuple[float, float]]):
        """bounds = {'x': (min, max), 'y': (min, max)}"""
        self.state_space = state_space
        self.bounds = bounds

    def violated(self, state):
        # Soft violation for differentiability
        violation = 0.0
        for agent in self.state_space.agent_names:
            pos = self.state_space.get_state(state, agent)[:2]
            # Penalty grows quadratically outside bounds
            violation += torch.relu(self.bounds['x'][0] - pos[0])**2
            violation += torch.relu(pos[0] - self.bounds['x'][1])**2
            violation += torch.relu(self.bounds['y'][0] - pos[1])**2
            violation += torch.relu(pos[1] - self.bounds['y'][1])**2
        return violation

    def project(self, state):
        """Clamp positions and reflect velocities at box boundaries."""
        new_state = state.clone()
        for agent in self.state_space.agent_names:
            pos = self.state_space.get_state(state, agent)[:2]
            # Clamp position
            pos_clamped = torch.stack([
                torch.clamp(pos[0], self.bounds['x'][0], self.bounds['x'][1]),
                torch.clamp(pos[1], self.bounds['y'][0], self.bounds['y'][1])
            ])
            # Reflect velocity if hit boundary
            vel = self.state_space.get_state(state, agent)[2:4]
            vel_new = vel.clone()
            if pos[0] <= self.bounds['x'][0] or pos[0] >= self.bounds['x'][1]:
                vel_new[0] *= -0.8  # Bounce with damping
            if pos[1] <= self.bounds['y'][0] or pos[1] >= self.bounds['y'][1]:
                vel_new[1] *= -0.8

            self.state_space.set_state(new_state, agent,
                                torch.cat([pos_clamped, vel_new]))
        return new_state

# ============================================================================
# LAYER 3.75: PAYOFF MODEL
# ============================================================================

class PayoffModel(ABC):
    """Unified payoff computation"""

    @abstractmethod
    def agents(self) -> List[str]:
        """Return list of agent names"""
        pass

    def step(self,
             state: torch.Tensor,
             controls: Dict,
             dt: float) -> Dict[str, float]:
        """Running cost per timestep"""
        return {a: 0.0 for a in self.agents()}

    def terminal(self,
                 state: torch.Tensor) -> Dict[str, float]:
        """Terminal payoff"""
        return {a: 0.0 for a in self.agents()}

    def total(self,
              trajectory: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
              final_state: torch.Tensor,
              dt: float) -> Dict[str, float]:
        """
        Total payoff over trajectory.
        Default: sum step payoffs + terminal.
        Override for discounting, non-additive payoffs, etc.
        """
        total = {a: 0.0 for a in self.agents()}

        for state, controls in trajectory:
            step_payoff = self.step(state, controls, dt)
            for a in self.agents():
                total[a] += step_payoff[a]

        terminal_payoff = self.terminal(final_state)
        for a in self.agents():
            total[a] += terminal_payoff[a]

        return total

class Integrator(ABC):
    """Numerical integration with automatic constraint projection"""

    def step(self,
             dynamics: Dynamics,
             state: torch.Tensor,
             controls: Dict[str, torch.Tensor],
             dt: float,
             constraints: Optional[List[Constraint]] = None) -> torch.Tensor:
        """
        Integrate one timestep and project onto constraints.

        Args:
            dynamics: dynamics model
            state: current state
            controls: control inputs
            dt: timestep
            constraints: optional list of constraints to enforce

        Returns:
            new_state (after constraint projection if provided)
        """
        # Integration
        new_state = self._integrate(dynamics, state, controls, dt)

        # Constraint projection (automatic if constraints provided)
        if constraints:
            for constraint in constraints:
                new_state = constraint.project(new_state)

        return new_state

    @abstractmethod
    def _integrate(self,
                   dynamics: Dynamics,
                   state: torch.Tensor,
                   controls: Dict[str, torch.Tensor],
                   dt: float) -> torch.Tensor:
        """Actual integration scheme (implemented by subclasses)"""
        pass

class EulerIntegrator(Integrator):
    """Simple Euler (differentiable)"""

    def _integrate(self, dynamics, state, controls, dt):
        dstate = dynamics.derivative(state, controls)
        return state + dstate * dt

class RK4Integrator(Integrator):
    """4th order Runge-Kutta (differentiable, more accurate)"""

    def _integrate(self, dynamics, state, controls, dt):
        k1 = dynamics.derivative(state, controls)
        k2 = dynamics.derivative(state + 0.5 * dt * k1, controls)
        k3 = dynamics.derivative(state + 0.5 * dt * k2, controls)
        k4 = dynamics.derivative(state + dt * k3, controls)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# LAYER 4: POLICIES (Differentiable)
# ============================================================================

class Policy(ABC):
    """Observation to control"""

    @abstractmethod
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Must be differentiable for learning"""
        pass

    @abstractmethod
    def control_dim(self) -> int:
        pass

class NeuralPolicy(Policy, nn.Module):
    """Learnable policy"""

    def __init__(self, obs_dim: int, control_dim: int, hidden_dim: int = 64):
        Policy.__init__(self)
        nn.Module.__init__(self)
        self._control_dim = control_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, control_dim),
            nn.Tanh(),  # Bounded output
        )

        # Small init for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return nn.Module.__call__(self, obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def control_dim(self) -> int:
        return self._control_dim

class FunctionPolicy(Policy):
    """Hand-coded policy (can be differentiable or not)"""

    def __init__(self,
                 fn: Callable[[torch.Tensor], torch.Tensor],
                 control_dim: int,
                 differentiable: bool = False):
        self.fn = fn
        self._control_dim = control_dim
        self.differentiable = differentiable

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            return self.fn(obs)
        else:
            with torch.no_grad():
                return self.fn(obs)

    def control_dim(self) -> int:
        return self._control_dim

# ============================================================================
# LAYER 5: AGENTS (Hold state spec, policies, and strategy sets)
# ============================================================================

class Agent:
    """Agent with state specification and named strategies"""

    def __init__(self,
                 name: str,
                 state_spec: StateSpec,
                 strategy_set: Dict[str, Policy]):
        """
        Args:
            name: agent identifier
            state_spec: symbolic state specification
            strategy_set: dict of strategy_name -> Policy
        """
        self.name = name
        self.state_spec = state_spec
        self.strategy_set = strategy_set
        self.strategy_names = list(strategy_set.keys())

    def get_policy(self, strategy_name: str) -> Policy:
        return self.strategy_set[strategy_name]

# ============================================================================
# LAYER 6: DIFFERENTIAL GAME (Pure definition, no execution)
# ============================================================================

class DifferentialGame:
    """
    Game definition.

    Defines:
    - State space structure
    - Observation model
    - Dynamics
    - Payoff function
    - Initial conditions

    Does not execute - that's Arena's job.
    """

    def __init__(self,
                 state_space: StateSpace,
                 agents: List[Agent],
                 obs_model: ObservationModel,
                 dynamics: Dynamics,
                 payoff_model: PayoffModel,
                 initial_sampler: Callable[[], torch.Tensor],
                 name: str = "Differential Game"):
        """
        Combine state, observations, dynamics, payoff model, and initial sampler.
        """
        self.state_space = state_space
        self.agents = {a.name: a for a in agents}
        self.agent_names = [a.name for a in agents]
        self.obs_model = obs_model
        self.dynamics = dynamics
        self.payoff_model = payoff_model
        self.initial_sampler = initial_sampler
        self.name = name

    def get_strategy_sets(self) -> Dict[str, List[str]]:
        """Get available strategies per agent"""
        return {name: agent.strategy_names for name, agent in self.agents.items()}

    def __repr__(self):
        return f'<DifferentialGame "{self.name}" agents={self.agent_names}>'

# ============================================================================
# LAYER 7: ARENA (Executes games)
# ============================================================================

class Arena:

    def __init__(self,
                 game: DifferentialGame,
                 integrator: Integrator = None,
                 dt: float = 0.02,
                 max_time: float = 10.0):
        self.game = game
        self.integrator = integrator or EulerIntegrator()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(max_time) or max_time < 0:
            raise ValueError("max_time must be nonnegative and finite")
        self.dt = dt
        self.max_time = max_time

    def initial_state(self,
                      physical_state: Optional[torch.Tensor] = None) -> GameState:
        if physical_state is None:
            physical_state = self.game.initial_sampler()

        return GameState(
            physical_state=physical_state,
            time=0.0,
            cumulative_payoffs={agent: 0.0 for agent in self.game.agent_names}
        )

    def tick(self,
             state: GameState,
             policies: Dict[str, Policy],
             constraints: Optional[List[Constraint]] = None) -> GameState:

        # Observe
        observations = {
            agent: self.game.obs_model.observe(
                state.physical_state,
                agent,
                state.cumulative_payoffs.get(agent, 0.0)
            )
            for agent in self.game.agent_names
        }

        # Act
        controls = {
            agent: policies[agent](observations[agent])
            for agent in self.game.agent_names
        }

        # Compute step payoffs (before state changes)
        step_payoffs = self.game.payoff_model.step(
            state.physical_state,
            controls,
            self.dt
        )

        # Integrate physics
        new_physical = self.integrator.step(
            self.game.dynamics,
            state.physical_state,
            controls,
            self.dt,
            constraints
        )

        # Build new state
        new_state = (state
                     .with_state(new_physical)
                     .with_time(state.time + self.dt)
                     .add_payoffs(step_payoffs))

        return new_state

    def simulate(self,
                 policies: Dict[str, Policy],
                 initial: Optional[GameState] = None,
                 until: Optional[float] = None,
                 constraints: Optional[List[Constraint]] = None) -> List[GameState]:

        if initial is None:
            initial = self.initial_state()

        end_time = until if until is not None else self.max_time

        if not math.isfinite(end_time) or not math.isfinite(initial.time):
            raise ValueError("simulation times must be finite")
        if end_time < initial.time:
            raise ValueError("end time must not precede the initial time")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")

        trajectory = [initial]
        state = initial

        while state.time < end_time:
            next_state = self.tick(state, policies, constraints)
            if next_state.time <= state.time:
                raise ValueError("dt is too small to advance simulation time")
            state = next_state
            trajectory.append(state)

        return trajectory

    def play(self,
             strategy_profile: Dict[str, str],
             initial: Optional[GameState] = None,
             constraints: Optional[List[Constraint]] = None) -> Tuple[List[GameState], Dict[str, float]]:

        # Convert strategy names to policies
        policies = {
            agent: self.game.agents[agent].get_policy(strategy_profile[agent])
            for agent in self.game.agent_names
        }

        # Simulate
        trajectory = self.simulate(policies, initial, constraints=constraints)

        # Add terminal payoffs
        final_state = trajectory[-1]
        terminal_payoffs = self.game.payoff_model.terminal(final_state.physical_state)

        total_payoffs = final_state.cumulative_payoffs.copy()
        for agent, reward in terminal_payoffs.items():
            total_payoffs[agent] += reward

        return trajectory, total_payoffs

    def expected_payoffs(self,
                        strategy_profile: Dict[str, str],
                        n_samples: int = 100) -> Dict[str, float]:

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        total_payoffs = {agent: 0.0 for agent in self.game.agent_names}

        for _ in range(n_samples):
            _, payoffs = self.play(strategy_profile, initial=None)  # Random init each time
            for agent in self.game.agent_names:
                total_payoffs[agent] += payoffs[agent]

        return {agent: total_payoffs[agent] / n_samples for agent in self.game.agent_names}

# ============================================================================
# LAYER 8: STRATEGY PROFILE ITERATION
# ============================================================================

class StrategyProfileIterator:
    """
    Iterate over all strategy profiles.
    Essential for converting differential game to normal form.
    """

    def __init__(self, strategy_sets: Dict[str, List[str]]):
        """
        strategy_sets: dict of agent -> list of strategy names
        """
        self.strategy_sets = strategy_sets
        self.agents = list(strategy_sets.keys())

    def __iter__(self):
        """Yield all strategy profiles"""
        strategy_lists = [self.strategy_sets[agent] for agent in self.agents]
        for profile_tuple in itertools.product(*strategy_lists):
            yield dict(zip(self.agents, profile_tuple))

    def count(self) -> int:
        """Total number of profiles"""
        count = 1
        for strategies in self.strategy_sets.values():
            count *= len(strategies)
        return count

# ============================================================================
# LAYER 9: NORMAL FORM CONVERTER
# ============================================================================

class NormalFormConverter:
    """Convert differential game to normal form """

    @staticmethod
    def to_payoff_matrix(arena: Arena,
                        players: List[str],
                        n_samples: int = 1) -> torch.Tensor:
        """
        Convert to normal form payoff tensor.

        Args:
            arena: Arena with differential game
            players: list of player names to include (typically subset of 2)
            n_samples: number of samples per profile (for averaging)

        Returns:
            payoff tensor with shape (n_players, *strategy_dims)
        """
        # Get strategy sets for selected players
        all_sets = arena.game.get_strategy_sets()
        strategy_sets = {p: all_sets[p] for p in players}

        # Other agents use first strategy by default
        fixed_strategies = {
            agent: all_sets[agent][0]
            for agent in arena.game.agent_names
            if agent not in players
        }

        # Build tensor shape
        dims = [len(strategy_sets[p]) for p in players]
        payoff_shape = [len(players)] + dims
        payoffs = torch.zeros(payoff_shape)

        # Iterate over all profiles
        iterator = StrategyProfileIterator(strategy_sets)
        for profile in iterator:

            # Combine with fixed strategies
            full_profile = {**fixed_strategies, **profile}

            # Get indices for this profile
            indices = tuple(strategy_sets[p].index(profile[p]) for p in players)

            # Compute expected payoff
            if n_samples == 1:
                _, payoff_dict = arena.play(full_profile)
            else:
                payoff_dict = arena.expected_payoffs(full_profile, n_samples)

            # Store in tensor
            for i, player in enumerate(players):
                payoffs[(i,) + indices] = payoff_dict[player]

        return payoffs

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory(trajectory: List[GameState],
                   state_space: StateSpace,
                   title: str = ""):
    """Plot 2D trajectories (assumes first 2 dims are x, y)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {name: f'C{i}' for i, name in enumerate(state_space.agent_names)}

    for agent_name in state_space.agent_names:
        positions = []
        for game_state in trajectory:
            state = game_state.physical_state
            agent_state = state_space.get_state(state, agent_name)
            positions.append(agent_state[:2])

        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1],
               color=colors[agent_name], label=agent_name, alpha=0.7, linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1],
                  color=colors[agent_name], s=150, marker='o', edgecolor='black', linewidth=2)
        ax.scatter(positions[-1, 0], positions[-1, 1],
                  color=colors[agent_name], s=150, marker='X', edgecolor='black', linewidth=2)

    ax.set_aspect('equal')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_payoff_heatmap(payoff_tensor: torch.Tensor,
                       players: List[str],
                       strategy_sets: Dict[str, List[str]]):
    """Visualize 2-player payoff matrix"""
    if len(players) != 2:
        raise ValueError("Can only plot 2-player games")

    p1, p2 = players
    p1_strats = strategy_sets[p1]
    p2_strats = strategy_sets[p2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Player 1 payoffs
    matrix1 = payoff_tensor[0].numpy()
    im1 = ax1.imshow(matrix1, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(p2_strats)))
    ax1.set_yticks(range(len(p1_strats)))
    ax1.set_xticklabels(p2_strats)
    ax1.set_yticklabels(p1_strats)
    ax1.set_xlabel(f'{p2} strategy')
    ax1.set_ylabel(f'{p1} strategy')
    ax1.set_title(f'{p1} Payoffs')

    for i in range(len(p1_strats)):
        for j in range(len(p2_strats)):
            ax1.text(j, i, f'{matrix1[i, j]:.1f}',
                    ha='center', va='center', fontsize=12, weight='bold')

    plt.colorbar(im1, ax=ax1)

    # Player 2 payoffs
    matrix2 = payoff_tensor[1].numpy()
    im2 = ax2.imshow(matrix2, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(p2_strats)))
    ax2.set_yticks(range(len(p1_strats)))
    ax2.set_xticklabels(p2_strats)
    ax2.set_yticklabels(p1_strats)
    ax2.set_xlabel(f'{p2} strategy')
    ax2.set_ylabel(f'{p1} strategy')
    ax2.set_title(f'{p2} Payoffs')

    for i in range(len(p1_strats)):
        for j in range(len(p2_strats)):
            ax2.text(j, i, f'{matrix2[i, j]:.1f}',
                    ha='center', va='center', fontsize=12, weight='bold')

    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    return fig

# ============================================================================
# EXAMPLE: STAG HUNT
# ============================================================================

# Common state specs
POINT_MASS_2D = StateSpec(['x', 'y', 'vx', 'vy'])
POINT_MASS_3D = StateSpec(['x', 'y', 'z', 'vx', 'vy', 'vz'])

def build_stag_hunt():
    """Build stag hunt differential game"""

    # Build agents with strategies
    def make_pursuit(state_space, agent_name: str, targets: List[str], speed: float):
        def fn(obs):
            # obs is full state - extract this agent's position
            own_pos = state_space.get_state(obs, agent_name)[:2]
            target_pos = torch.stack([state_space.get_state(obs, t)[:2] for t in targets]).mean(dim=0)
            direction = target_pos - own_pos
            dist = torch.norm(direction) + 1e-6
            return (direction / dist) * speed
        return FunctionPolicy(fn, 2, differentiable=True)

    def make_flee(state_space, agent_name: str, threats: List[str], speed: float):
        def fn(obs):
            # obs is full state - extract this agent's position
            own_pos = state_space.get_state(obs, agent_name)[:2]
            threat_pos = torch.stack([state_space.get_state(obs, t)[:2] for t in threats]).mean(dim=0)
            direction = own_pos - threat_pos
            dist = torch.norm(direction) + 1e-6
            return (direction / dist) * speed
        return FunctionPolicy(fn, 2, differentiable=True)

    # Create agents with state specifications
    agents = [
        Agent('c1', POINT_MASS_2D, {}),  # Strategies added below
        Agent('c2', POINT_MASS_2D, {}),
        Agent('stag', POINT_MASS_2D, {}),
        Agent('hare1', POINT_MASS_2D, {}),
        Agent('hare2', POINT_MASS_2D, {}),
    ]

    # Build state space from agents
    state_space = StateSpace(agents, shared_spec=None)

    # Now add strategies (need state_space for closures)
    agents[0].strategy_set = {
        'ChaseStag': make_pursuit(state_space, 'c1', ['stag'], 1.5),
        'ChaseHare': make_pursuit(state_space, 'c1', ['hare1'], 1.5),
    }
    agents[0].strategy_names = list(agents[0].strategy_set.keys())

    agents[1].strategy_set = {
        'ChaseStag': make_pursuit(state_space, 'c2', ['stag'], 1.5),
        'ChaseHare': make_pursuit(state_space, 'c2', ['hare2'], 1.5),
    }
    agents[1].strategy_names = list(agents[1].strategy_set.keys())

    agents[2].strategy_set = {'Flee': make_flee(state_space, 'stag', ['c1', 'c2'], 1.1)}
    agents[2].strategy_names = list(agents[2].strategy_set.keys())

    agents[3].strategy_set = {'Flee': make_flee(state_space, 'hare1', ['c1', 'c2'], 0.6)}
    agents[3].strategy_names = list(agents[3].strategy_set.keys())

    agents[4].strategy_set = {'Flee': make_flee(state_space, 'hare2', ['c1', 'c2'], 0.6)}
    agents[4].strategy_names = list(agents[4].strategy_set.keys())

    # Observation: full state
    class FullObs(ObservationModel):
        def observe(self, state, agent, cumulative_payoff=None):
            return state
        def obs_dim(self, agent):
            return state_space.dim

    obs_model = FullObs()

    # Kinematic dynamics
    all_agents = ['c1', 'c2', 'stag', 'hare1', 'hare2']

    class StagHuntPayoff(PayoffModel):
        def __init__(self, state_space, agent_names):
            self.state_space = state_space
            self._agents = agent_names

        def agents(self):
            return self._agents

        def terminal(self, state):
            positions = {a: state_space.get_state(state, a)[:2] for a in all_agents}

            payoffs = {a: 0.0 for a in all_agents}
            radius = 0.5
            captured = {'c1': False, 'c2': False}

            # Stag (needs both)
            d1 = torch.norm(positions['c1'] - positions['stag']).item()
            d2 = torch.norm(positions['c2'] - positions['stag']).item()

            if d1 < radius and d2 < radius:
                payoffs['c1'] += 4.0
                payoffs['c2'] += 4.0
                captured['c1'] = True
                captured['c2'] = True

            # Hares (first come first served)
            if not captured['c1']:
                for hare in ['hare1', 'hare2']:
                    d = torch.norm(positions['c1'] - positions[hare]).item()
                    if d < radius:
                        payoffs['c1'] += 3.0
                        captured['c1'] = True
                        break

            if not captured['c2']:
                for hare in ['hare1', 'hare2']:
                    d = torch.norm(positions['c2'] - positions[hare]).item()
                    if d < radius:
                        payoffs['c2'] += 3.0
                        captured['c2'] = True
                        break

            return payoffs

    class KinematicDynamics(Dynamics):
        def __init__(self, max_speeds: Dict[str, float]):
            self.max_speeds = max_speeds

        def derivative(self, state, controls):
            dstate = []
            for agent in all_agents:
                agent_state = state_space.get_state(state, agent)
                control = controls[agent]

                # Soft clamping for differentiability
                vel_des = control
                speed = torch.norm(vel_des) + 1e-6
                max_speed = self.max_speeds[agent]

                # Smooth speed bound: max_speed * tanh(speed / max_speed)
                scale_factor = max_speed * torch.tanh(speed / max_speed) / speed
                vel = vel_des * scale_factor

                dpos = vel
                dvel = torch.zeros(2)
                dstate.append(torch.cat([dpos, dvel]))

            return torch.cat(dstate)

    dynamics = KinematicDynamics({
        'c1': 1.5, 'c2': 1.5,
        'stag': 1.1, 'hare1': 0.6, 'hare2': 0.6
    })

    # Initial state
    def initial():
        return torch.cat([
            torch.tensor([-2.0, -2.0, 0.0, 0.0]),  # c1
            torch.tensor([2.0, -2.0, 0.0, 0.0]),   # c2
            torch.tensor([0.0, 2.0, 0.0, 0.0]),    # stag
            torch.tensor([-1.5, 0.0, 0.0, 0.0]),   # hare1
            torch.tensor([1.5, 0.0, 0.0, 0.0]),    # hare2
        ])

    payoff_model = StagHuntPayoff(state_space, all_agents)
    game = DifferentialGame(state_space, agents, obs_model, dynamics, payoff_model, initial, "Stag Hunt")

    return game, state_space
