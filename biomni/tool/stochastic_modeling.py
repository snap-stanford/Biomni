"""
Stochastic Modeling and Simulation Tools.

This module provides functions for stochastic processes, Monte Carlo simulation,
Markov chains, random walks, and probabilistic modeling.
"""

import numpy as np
from scipy import stats
from scipy.linalg import eig
import matplotlib.pyplot as plt


def monte_carlo_simulation(function, parameter_distributions, n_samples=10000, seed=None):
    """
    Perform Monte Carlo simulation for uncertainty quantification.

    Parameters:
    -----------
    function : callable
        Function to evaluate (takes array of parameters)
    parameter_distributions : list of tuples
        List of (distribution_name, *params) for each parameter
        e.g., [('norm', 0, 1), ('uniform', 0, 10)]
    n_samples : int
        Number of Monte Carlo samples
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    str
        Monte Carlo simulation results and statistics
    """
    log = "# Monte Carlo Simulation\n\n"

    if seed is not None:
        np.random.seed(seed)

    n_params = len(parameter_distributions)
    log += f"## Configuration:\n"
    log += f"- Number of parameters: {n_params}\n"
    log += f"- Number of samples: {n_samples}\n"
    log += f"- Random seed: {seed}\n\n"

    log += "## Parameter Distributions:\n"
    for i, dist_spec in enumerate(parameter_distributions):
        log += f"- Parameter {i}: {dist_spec[0]}({', '.join(map(str, dist_spec[1:]))})\n"
    log += "\n"

    # Generate samples
    samples = np.zeros((n_samples, n_params))
    for i, dist_spec in enumerate(parameter_distributions):
        dist_name = dist_spec[0]
        dist_params = dist_spec[1:]

        dist = getattr(stats, dist_name)
        samples[:, i] = dist.rvs(*dist_params, size=n_samples)

    # Evaluate function for all samples
    log += "## Running Simulation:\n"
    results = np.array([function(sample) for sample in samples])

    log += f"✓ Completed {n_samples} evaluations\n\n"

    # Statistical analysis
    log += "## Results Statistics:\n"
    log += f"- Mean: {np.mean(results):.6f}\n"
    log += f"- Median: {np.median(results):.6f}\n"
    log += f"- Std: {np.std(results):.6f}\n"
    log += f"- Min: {np.min(results):.6f}\n"
    log += f"- Max: {np.max(results):.6f}\n\n"

    # Percentiles
    log += "## Percentiles:\n"
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(results, p)
        log += f"- {p}th: {value:.6f}\n"

    # Confidence intervals
    log += "\n## Confidence Intervals:\n"
    for confidence in [0.90, 0.95, 0.99]:
        alpha = 1 - confidence
        lower = np.percentile(results, 100 * alpha / 2)
        upper = np.percentile(results, 100 * (1 - alpha / 2))
        log += f"- {int(confidence*100)}% CI: [{lower:.6f}, {upper:.6f}]\n"

    return log


def simulate_markov_chain(transition_matrix, initial_state, n_steps=1000, state_names=None):
    """
    Simulate a discrete-time Markov chain.

    Parameters:
    -----------
    transition_matrix : array-like, shape (n, n)
        Transition probability matrix where P[i,j] = P(next_state=j | current_state=i)
    initial_state : int
        Initial state index
    n_steps : int
        Number of steps to simulate
    state_names : list of str, optional
        Names for the states

    Returns:
    --------
    str
        Markov chain simulation results and analysis
    """
    log = "# Markov Chain Simulation\n\n"

    P = np.array(transition_matrix)
    n_states = P.shape[0]

    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]

    log += f"## Configuration:\n"
    log += f"- Number of states: {n_states}\n"
    log += f"- Initial state: {state_names[initial_state]}\n"
    log += f"- Number of steps: {n_steps}\n\n"

    # Verify transition matrix
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        log += "⚠ WARNING: Transition matrix rows do not sum to 1\n\n"

    log += "## Transition Matrix:\n```\n"
    for i in range(n_states):
        log += f"{state_names[i]}: {P[i]}\n"
    log += "```\n\n"

    # Simulate chain
    states = [initial_state]
    current_state = initial_state

    for _ in range(n_steps - 1):
        # Sample next state based on transition probabilities
        next_state = np.random.choice(n_states, p=P[current_state])
        states.append(next_state)
        current_state = next_state

    states = np.array(states)

    # Analyze simulation
    log += "## Simulation Results:\n"

    # State frequencies
    log += "### State Frequencies:\n"
    for i in range(n_states):
        frequency = np.sum(states == i) / n_steps
        log += f"- {state_names[i]}: {frequency:.4f} ({np.sum(states == i)} occurrences)\n"

    # Compute stationary distribution
    log += "\n## Stationary Distribution Analysis:\n"
    eigenvalues, eigenvectors = eig(P.T)

    # Find eigenvector corresponding to eigenvalue 1
    stationary_idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-10)
    stationary_dist = np.real(eigenvectors[:, stationary_idx])
    stationary_dist = stationary_dist / stationary_dist.sum()

    log += "Theoretical stationary distribution:\n"
    for i in range(n_states):
        log += f"- {state_names[i]}: {stationary_dist[i]:.4f}\n"

    # Compare with observed
    log += "\n### Comparison (Observed vs Theoretical):\n"
    for i in range(n_states):
        observed = np.sum(states == i) / n_steps
        theoretical = stationary_dist[i]
        diff = abs(observed - theoretical)
        log += f"- {state_names[i]}: |{observed:.4f} - {theoretical:.4f}| = {diff:.4f}\n"

    return log


def random_walk_simulation(dimensions=1, n_steps=1000, step_distribution='normal', step_params=(0, 1)):
    """
    Simulate a random walk process.

    Parameters:
    -----------
    dimensions : int
        Number of dimensions (1D, 2D, or 3D walk)
    n_steps : int
        Number of steps
    step_distribution : str
        Distribution for step sizes: 'normal', 'uniform', 'exponential'
    step_params : tuple
        Parameters for the step distribution

    Returns:
    --------
    str
        Random walk analysis
    """
    log = f"# {dimensions}D Random Walk Simulation\n\n"

    log += f"## Configuration:\n"
    log += f"- Dimensions: {dimensions}\n"
    log += f"- Number of steps: {n_steps}\n"
    log += f"- Step distribution: {step_distribution}{step_params}\n\n"

    # Generate steps
    dist = getattr(stats, step_distribution)
    steps = dist.rvs(*step_params, size=(n_steps, dimensions))

    # Compute positions
    positions = np.cumsum(steps, axis=0)

    # Add initial position (0, 0, ...)
    positions = np.vstack([np.zeros(dimensions), positions])

    log += "## Walk Statistics:\n"

    # Final position
    final_position = positions[-1]
    log += f"- Final position: {final_position}\n"

    # Displacement from origin
    final_distance = np.linalg.norm(final_position)
    log += f"- Final distance from origin: {final_distance:.4f}\n"

    # Theoretical expected distance for Brownian motion: sqrt(n_steps * variance)
    if step_distribution == 'normal':
        variance = step_params[1]**2
        expected_distance = np.sqrt(n_steps * variance * dimensions)
        log += f"- Expected distance (theory): {expected_distance:.4f}\n"

    # Maximum distance reached
    distances = np.linalg.norm(positions, axis=1)
    max_distance = np.max(distances)
    log += f"- Maximum distance reached: {max_distance:.4f}\n"

    # Statistics per dimension
    if dimensions <= 3:
        log += "\n## Per-Dimension Statistics:\n"
        dim_names = ['x', 'y', 'z']
        for i in range(dimensions):
            log += f"### {dim_names[i]}-dimension:\n"
            log += f"- Final: {final_position[i]:.4f}\n"
            log += f"- Min: {positions[:, i].min():.4f}\n"
            log += f"- Max: {positions[:, i].max():.4f}\n"
            log += f"- Range: {positions[:, i].max() - positions[:, i].min():.4f}\n"

    # Persistence (directional correlation)
    log += "\n## Persistence Analysis:\n"
    if dimensions == 1:
        # Number of times walk changes direction
        direction_changes = np.sum(np.diff(np.sign(steps[:, 0])) != 0)
        log += f"- Direction changes: {direction_changes} ({100*direction_changes/n_steps:.1f}%)\n"

    return log


def gillespie_algorithm(reactions, initial_state, rate_constants, t_max=10.0, max_events=10000):
    """
    Simulate a stochastic chemical reaction system using Gillespie's algorithm.

    Parameters:
    -----------
    reactions : list of tuples
        Each reaction as (reactants_dict, products_dict)
        e.g., [({'A': 1, 'B': 1}, {'C': 1})]  for A + B -> C
    initial_state : dict
        Initial molecule counts {'A': 100, 'B': 50, ...}
    rate_constants : list
        Rate constant for each reaction
    t_max : float
        Maximum simulation time
    max_events : int
        Maximum number of events to simulate

    Returns:
    --------
    str
        Stochastic simulation results
    """
    log = "# Gillespie Algorithm - Stochastic Simulation\n\n"

    log += f"## Configuration:\n"
    log += f"- Number of reactions: {len(reactions)}\n"
    log += f"- Initial state: {initial_state}\n"
    log += f"- Maximum time: {t_max}\n"
    log += f"- Maximum events: {max_events}\n\n"

    log += "## Reactions:\n"
    for i, (reactants, products) in enumerate(reactions):
        react_str = ' + '.join([f"{v}{k}" if v > 1 else k for k, v in reactants.items()])
        prod_str = ' + '.join([f"{v}{k}" if v > 1 else k for k, v in products.items()])
        log += f"{i+1}. {react_str} → {prod_str} (k={rate_constants[i]})\n"
    log += "\n"

    # Initialize
    state = initial_state.copy()
    t = 0
    event_count = 0

    # Track trajectory
    times = [t]
    states = [state.copy()]

    log += "## Simulation Progress:\n"

    while t < t_max and event_count < max_events:
        # Calculate propensities
        propensities = []
        for i, (reactants, products) in enumerate(reactions):
            prop = rate_constants[i]
            for species, count in reactants.items():
                if species not in state or state[species] < count:
                    prop = 0
                    break
                # Compute binomial coefficient for multiple molecules
                for j in range(count):
                    prop *= (state[species] - j) / (j + 1)
            propensities.append(prop)

        propensities = np.array(propensities)
        total_propensity = np.sum(propensities)

        if total_propensity == 0:
            log += f"⚠ Simulation stopped at t={t:.4f}: No reactions possible\n\n"
            break

        # Sample time to next reaction
        tau = np.random.exponential(1.0 / total_propensity)
        t += tau

        if t > t_max:
            break

        # Sample which reaction occurs
        reaction_idx = np.random.choice(len(reactions), p=propensities / total_propensity)

        # Update state
        reactants, products = reactions[reaction_idx]
        for species, count in reactants.items():
            state[species] -= count
        for species, count in products.items():
            state[species] = state.get(species, 0) + count

        # Record
        times.append(t)
        states.append(state.copy())
        event_count += 1

        if event_count % (max_events // 10) == 0:
            log += f"  t = {t:.4f}, events = {event_count}\n"

    log += f"\n✓ Simulation completed\n"
    log += f"- Final time: {t:.4f}\n"
    log += f"- Total events: {event_count}\n"
    log += f"- Final state: {state}\n\n"

    # Analyze species trajectories
    log += "## Species Analysis:\n"
    species_list = list(initial_state.keys())
    for species in species_list:
        values = [s.get(species, 0) for s in states]
        log += f"### {species}:\n"
        log += f"- Initial: {values[0]}\n"
        log += f"- Final: {values[-1]}\n"
        log += f"- Min: {min(values)}\n"
        log += f"- Max: {max(values)}\n"
        log += f"- Mean: {np.mean(values):.2f}\n"

    return log


def wiener_process_simulation(t_max=1.0, n_steps=1000, n_paths=5):
    """
    Simulate Wiener process (Brownian motion) with multiple sample paths.

    Parameters:
    -----------
    t_max : float
        Maximum time
    n_steps : int
        Number of time steps
    n_paths : int
        Number of sample paths to generate

    Returns:
    --------
    str
        Wiener process simulation results
    """
    log = "# Wiener Process (Brownian Motion) Simulation\n\n"

    dt = t_max / n_steps
    t = np.linspace(0, t_max, n_steps + 1)

    log += f"## Configuration:\n"
    log += f"- Time span: [0, {t_max}]\n"
    log += f"- Number of steps: {n_steps}\n"
    log += f"- Time step: dt = {dt:.6f}\n"
    log += f"- Number of paths: {n_paths}\n\n"

    # Generate paths
    paths = np.zeros((n_paths, n_steps + 1))

    for i in range(n_paths):
        # Generate increments: dW ~ N(0, dt)
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        # Cumulative sum to get W(t)
        paths[i, 1:] = np.cumsum(dW)

    log += "## Simulation Results:\n"

    # Analyze paths
    for i in range(n_paths):
        final_value = paths[i, -1]
        max_value = paths[i].max()
        min_value = paths[i].min()
        log += f"### Path {i+1}:\n"
        log += f"- Final value W({t_max}) = {final_value:.4f}\n"
        log += f"- Range: [{min_value:.4f}, {max_value:.4f}]\n"

    log += "\n## Theoretical Properties:\n"
    log += f"- E[W(t)] = 0 (zero mean)\n"
    log += f"- Var[W(t)] = t\n"
    log += f"- For t={t_max}: Expected variance = {t_max:.4f}\n"
    log += f"- Observed variance across paths: {np.var(paths[:, -1]):.4f}\n"

    return log


def poisson_process_simulation(rate, t_max=10.0):
    """
    Simulate a Poisson process.

    Parameters:
    -----------
    rate : float
        Event rate (λ)
    t_max : float
        Maximum time

    Returns:
    --------
    str
        Poisson process simulation results
    """
    log = "# Poisson Process Simulation\n\n"

    log += f"## Configuration:\n"
    log += f"- Event rate λ = {rate}\n"
    log += f"- Time span: [0, {t_max}]\n\n"

    # Generate inter-arrival times (exponentially distributed)
    event_times = []
    t = 0

    while t < t_max:
        inter_arrival = np.random.exponential(1.0 / rate)
        t += inter_arrival
        if t <= t_max:
            event_times.append(t)

    n_events = len(event_times)

    log += "## Simulation Results:\n"
    log += f"- Number of events: {n_events}\n"
    log += f"- Expected number of events: λ×T = {rate * t_max:.2f}\n"
    log += f"- Difference: {abs(n_events - rate * t_max):.2f}\n\n"

    if n_events > 0:
        log += "## Event Times:\n"
        log += f"- First event: t = {event_times[0]:.4f}\n"
        log += f"- Last event: t = {event_times[-1]:.4f}\n\n"

        # Inter-arrival time statistics
        if n_events > 1:
            inter_arrivals = np.diff([0] + event_times)
            log += "## Inter-Arrival Times:\n"
            log += f"- Mean: {np.mean(inter_arrivals):.4f}\n"
            log += f"- Expected (1/λ): {1.0/rate:.4f}\n"
            log += f"- Std: {np.std(inter_arrivals):.4f}\n"
            log += f"- Min: {np.min(inter_arrivals):.4f}\n"
            log += f"- Max: {np.max(inter_arrivals):.4f}\n"

    return log
