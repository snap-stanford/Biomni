"""
Dynamical Systems Analysis Tools.

This module provides functions for analyzing dynamical systems including
phase portrait analysis, bifurcation diagrams, Lyapunov exponents, and chaos theory.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def generate_phase_portrait(
    system_func, x_range, y_range, parameters=(), n_trajectories=10, time_span=(0, 10), grid_density=20
):
    """
    Generate a phase portrait for a 2D dynamical system.

    Parameters:
    -----------
    system_func : callable
        Function defining dx/dt = f(t, [x, y], parameters)
    x_range, y_range : tuple
        (min, max) ranges for x and y axes
    parameters : tuple
        Additional parameters for the system
    n_trajectories : int
        Number of trajectories to plot
    time_span : tuple
        (t_start, t_end) for trajectory integration
    grid_density : int
        Density of the vector field grid

    Returns:
    --------
    str
        Phase portrait analysis summary
    """
    log = "# Phase Portrait Analysis\n\n"

    log += "## System Configuration:\n"
    log += f"- Phase space: x ∈ {x_range}, y ∈ {y_range}\n"
    log += f"- Number of trajectories: {n_trajectories}\n"
    log += f"- Time span: {time_span}\n"
    log += f"- Vector field grid: {grid_density} × {grid_density}\n\n"

    # Create vector field
    x = np.linspace(x_range[0], x_range[1], grid_density)
    y = np.linspace(y_range[0], y_range[1], grid_density)
    X, Y = np.meshgrid(x, y)

    # Compute derivatives at each grid point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(grid_density):
        for j in range(grid_density):
            state = [X[i, j], Y[i, j]]
            if parameters:
                derivatives = system_func(0, state, *parameters)
            else:
                derivatives = system_func(0, state)
            U[i, j] = derivatives[0]
            V[i, j] = derivatives[1]

    log += "## Vector Field Analysis:\n"
    max_speed = np.sqrt(U**2 + V**2).max()
    log += f"- Maximum flow speed: {max_speed:.4f}\n"
    log += f"- Mean flow speed: {np.sqrt(U**2 + V**2).mean():.4f}\n\n"

    # Generate trajectories
    log += "## Trajectory Analysis:\n"
    log += f"Generating {n_trajectories} trajectories...\n\n"

    trajectories = []
    for i in range(n_trajectories):
        # Random initial condition
        x0 = np.random.uniform(x_range[0], x_range[1])
        y0 = np.random.uniform(y_range[0], y_range[1])

        # Integrate
        def ode_wrapper(t, state):
            if parameters:
                return system_func(t, state, *parameters)
            else:
                return system_func(t, state)

        t_eval = np.linspace(time_span[0], time_span[1], 1000)
        solution = solve_ivp(ode_wrapper, time_span, [x0, y0], t_eval=t_eval)

        if solution.success:
            trajectories.append((solution.y[0], solution.y[1]))

    log += f"Successfully computed {len(trajectories)} trajectories\n"
    log += "Phase portrait data generated (use matplotlib to visualize)\n"

    return log


def bifurcation_diagram(
    system_func,
    parameter_range,
    initial_condition,
    n_points=500,
    transient_steps=1000,
    sample_steps=100,
    parameter_index=0,
):
    """
    Generate a bifurcation diagram by varying a system parameter.

    Parameters:
    -----------
    system_func : callable
        Discrete map function: x_{n+1} = f(x_n, parameter)
    parameter_range : tuple
        (min, max) range for the bifurcation parameter
    initial_condition : float or array
        Initial state
    n_points : int
        Number of parameter values to sample
    transient_steps : int
        Number of initial iterations to discard
    sample_steps : int
        Number of points to sample after transient

    Returns:
    --------
    str
        Bifurcation analysis summary
    """
    log = "# Bifurcation Diagram\n\n"

    log += "## Configuration:\n"
    log += f"- Parameter range: {parameter_range}\n"
    log += f"- Number of parameter values: {n_points}\n"
    log += f"- Transient steps: {transient_steps}\n"
    log += f"- Sample steps: {sample_steps}\n"
    log += f"- Initial condition: {initial_condition}\n\n"

    parameters = np.linspace(parameter_range[0], parameter_range[1], n_points)
    bifurcation_data = []

    log += "## Computing Bifurcation Diagram:\n"

    for i, param in enumerate(parameters):
        x = initial_condition

        # Transient phase
        for _ in range(transient_steps):
            x = system_func(x, param)

        # Sampling phase
        samples = []
        for _ in range(sample_steps):
            x = system_func(x, param)
            samples.append(x)

        bifurcation_data.append((param, samples))

        if i % (n_points // 10) == 0:
            log += f"  Progress: {100 * i // n_points}%\n"

    log += "\n✓ Bifurcation diagram computed\n\n"

    # Analyze attractors
    log += "## Attractor Analysis:\n"

    # Sample a few parameter values
    sample_indices = [0, n_points // 4, n_points // 2, 3 * n_points // 4, n_points - 1]
    for idx in sample_indices:
        param, samples = bifurcation_data[idx]
        unique_points = len(set(np.round(samples, 6)))
        log += f"- Parameter = {param:.4f}: {unique_points} distinct points"

        if unique_points == 1:
            log += " (fixed point)\n"
        elif unique_points <= 10:
            log += f" (period-{unique_points} orbit)\n"
        else:
            log += " (chaotic or quasi-periodic)\n"

    log += "\nBifurcation data available for visualization.\n"

    return log


def compute_lyapunov_exponent(
    system_func, initial_condition, parameters=(), n_iterations=10000, dt=0.01, perturbation=1e-8
):
    """
    Compute the largest Lyapunov exponent for a dynamical system.

    A positive Lyapunov exponent indicates chaos.

    Parameters:
    -----------
    system_func : callable
        System function defining the dynamics
    initial_condition : array
        Initial state
    parameters : tuple
        System parameters
    n_iterations : int
        Number of iterations
    dt : float
        Time step
    perturbation : float
        Initial perturbation size

    Returns:
    --------
    str
        Lyapunov exponent estimation
    """
    log = "# Lyapunov Exponent Computation\n\n"

    initial_condition = np.array(initial_condition)
    n_dims = len(initial_condition)

    log += "## Configuration:\n"
    log += f"- System dimension: {n_dims}\n"
    log += f"- Initial condition: {initial_condition}\n"
    log += f"- Number of iterations: {n_iterations}\n"
    log += f"- Time step: {dt}\n"
    log += f"- Initial perturbation: {perturbation}\n\n"

    # Initialize
    x = initial_condition.copy()
    x_perturbed = x + perturbation

    lyapunov_sum = 0.0

    log += "## Computing Lyapunov Exponent:\n"

    for i in range(n_iterations):
        # Integrate both trajectories for one step
        if parameters:
            dx = system_func(0, x, *parameters) * dt
            dx_pert = system_func(0, x_perturbed, *parameters) * dt
        else:
            dx = system_func(0, x) * dt
            dx_pert = system_func(0, x_perturbed) * dt

        x = x + dx
        x_perturbed = x_perturbed + dx_pert

        # Compute separation
        separation = np.linalg.norm(x_perturbed - x)

        # Add to Lyapunov sum
        if separation > 0:
            lyapunov_sum += np.log(separation / perturbation)

            # Renormalize perturbation
            x_perturbed = x + (x_perturbed - x) * (perturbation / separation)

        if i % (n_iterations // 10) == 0 and i > 0:
            log += f"  Iteration {i}: λ ≈ {lyapunov_sum / (i * dt):.6f}\n"

    # Final Lyapunov exponent
    lyapunov_exponent = lyapunov_sum / (n_iterations * dt)

    log += "\n## Results:\n"
    log += f"**Largest Lyapunov Exponent: λ = {lyapunov_exponent:.6f}**\n\n"

    log += "## Interpretation:\n"
    if lyapunov_exponent > 0.01:
        log += "✓ **CHAOTIC** - Positive Lyapunov exponent indicates sensitive dependence on initial conditions\n"
    elif lyapunov_exponent > -0.01:
        log += "⚠ **MARGINAL** - Near-zero exponent, may be at edge of chaos or require longer integration\n"
    else:
        log += "✓ **STABLE** - Negative Lyapunov exponent indicates convergence to attractor\n"

    return log


def poincare_section(
    system_func, initial_conditions, parameters=(), plane_axis=2, plane_value=0, time_span=(0, 1000), n_trajectories=5
):
    """
    Compute a Poincaré section for a 3D dynamical system.

    Parameters:
    -----------
    system_func : callable
        System function defining dx/dt
    initial_conditions : list of arrays
        List of initial conditions
    parameters : tuple
        System parameters
    plane_axis : int
        Axis perpendicular to the Poincaré plane (0=x, 1=y, 2=z)
    plane_value : float
        Value of the plane (e.g., z=0)
    time_span : tuple
        Integration time span
    n_trajectories : int
        Number of trajectories to compute

    Returns:
    --------
    str
        Poincaré section analysis
    """
    log = "# Poincaré Section Analysis\n\n"

    axis_names = ["x", "y", "z"]
    log += "## Configuration:\n"
    log += f"- Poincaré plane: {axis_names[plane_axis]} = {plane_value}\n"
    log += f"- Number of trajectories: {n_trajectories}\n"
    log += f"- Time span: {time_span}\n\n"

    all_intersections = []

    log += "## Computing Trajectories:\n"

    for i, ic in enumerate(initial_conditions[:n_trajectories]):
        # Integrate
        def ode_wrapper(t, state):
            if parameters:
                return system_func(t, state, *parameters)
            else:
                return system_func(t, state)

        t_eval = np.linspace(time_span[0], time_span[1], 10000)
        solution = solve_ivp(ode_wrapper, time_span, ic, t_eval=t_eval, dense_output=True)

        if not solution.success:
            log += f"⚠ Trajectory {i + 1} failed to integrate\n"
            continue

        # Find intersections with Poincaré plane
        trajectory = solution.y.T
        intersections = []

        for j in range(len(trajectory) - 1):
            # Check if trajectory crosses the plane
            if (trajectory[j, plane_axis] - plane_value) * (trajectory[j + 1, plane_axis] - plane_value) < 0:
                # Linear interpolation to find exact crossing point
                alpha = (plane_value - trajectory[j, plane_axis]) / (
                    trajectory[j + 1, plane_axis] - trajectory[j, plane_axis]
                )

                intersection = trajectory[j] + alpha * (trajectory[j + 1] - trajectory[j])

                # Only count upward crossings (to get unique points)
                if trajectory[j + 1, plane_axis] > trajectory[j, plane_axis]:
                    intersections.append(intersection)

        all_intersections.extend(intersections)
        log += f"  Trajectory {i + 1}: {len(intersections)} intersections found\n"

    log += "\n## Results:\n"
    log += f"- Total intersections: {len(all_intersections)}\n"

    if len(all_intersections) > 0:
        intersections_array = np.array(all_intersections)

        # Get the two coordinates orthogonal to the plane axis
        other_axes = [k for k in range(3) if k != plane_axis]

        log += "\n## Intersection Statistics:\n"
        log += f"- {axis_names[other_axes[0]]} range: [{intersections_array[:, other_axes[0]].min():.4f}, "
        log += f"{intersections_array[:, other_axes[0]].max():.4f}]\n"
        log += f"- {axis_names[other_axes[1]]} range: [{intersections_array[:, other_axes[1]].min():.4f}, "
        log += f"{intersections_array[:, other_axes[1]].max():.4f}]\n"

        log += "\nPoincaré section data available for visualization.\n"
    else:
        log += "\n⚠ No intersections found. Try adjusting plane position or time span.\n"

    return log


def find_limit_cycle(system_func, initial_guess, parameters=(), period_guess=10.0, tol=1e-6):
    """
    Find a limit cycle (periodic orbit) in a dynamical system.

    Parameters:
    -----------
    system_func : callable
        System defining the dynamics
    initial_guess : array
        Initial guess for a point on the limit cycle
    parameters : tuple
        System parameters
    period_guess : float
        Guess for the period
    tol : float
        Tolerance for convergence

    Returns:
    --------
    str
        Limit cycle analysis
    """
    log = "# Limit Cycle Detection\n\n"

    initial_guess = np.array(initial_guess)
    n_dims = len(initial_guess)

    log += "## Configuration:\n"
    log += f"- System dimension: {n_dims}\n"
    log += f"- Initial guess: {initial_guess}\n"
    log += f"- Period guess: {period_guess}\n"
    log += f"- Tolerance: {tol}\n\n"

    log += "## Searching for Limit Cycle:\n"

    # Use shooting method to find periodic orbit
    def shooting_function(x_and_T):
        x0 = x_and_T[:-1]
        T = x_and_T[-1]

        # Integrate for one period
        def ode_wrapper(t, state):
            if parameters:
                return system_func(t, state, *parameters)
            else:
                return system_func(t, state)

        solution = solve_ivp(ode_wrapper, (0, T), x0, dense_output=True)

        if not solution.success:
            return np.ones(n_dims + 1) * 1e10

        x_final = solution.y[:, -1]

        # Residual: x(T) - x(0) should be zero for periodic orbit
        residual = x_final - x0

        # Add phase condition to fix period (prevents trivial solution)
        phase_condition = np.dot(x0, system_func(0, x0) if not parameters else system_func(0, x0, *parameters))

        return np.append(residual, phase_condition)

    try:
        # Initial guess: state + period
        x0_full = np.append(initial_guess, period_guess)

        result = fsolve(shooting_function, x0_full, full_output=True)
        solution, info, ier, msg = result

        if ier == 1:
            limit_cycle_point = solution[:-1]
            period = solution[-1]

            log += "✓ Limit cycle found!\n\n"
            log += "## Limit Cycle Properties:\n"
            log += f"- Period T = {period:.6f}\n"
            log += f"- Point on cycle: {limit_cycle_point}\n"
            log += f"- Residual norm: {np.linalg.norm(info['fvec']):.2e}\n"
        else:
            log += "✗ Search did not converge\n"
            log += f"- Message: {msg}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def analyze_attractor(system_func, initial_condition, parameters=(), time_span=(0, 100), n_points=10000):
    """
    Analyze the attractor of a dynamical system.

    Parameters:
    -----------
    system_func : callable
        System dynamics
    initial_condition : array
        Initial state
    parameters : tuple
        System parameters
    time_span : tuple
        Time range for integration
    n_points : int
        Number of time points

    Returns:
    --------
    str
        Attractor analysis
    """
    log = "# Attractor Analysis\n\n"

    initial_condition = np.array(initial_condition)
    n_dims = len(initial_condition)

    log += "## Configuration:\n"
    log += f"- System dimension: {n_dims}\n"
    log += f"- Initial condition: {initial_condition}\n"
    log += f"- Time span: {time_span}\n"
    log += f"- Number of points: {n_points}\n\n"

    # Integrate system
    def ode_wrapper(t, state):
        if parameters:
            return system_func(t, state, *parameters)
        else:
            return system_func(t, state)

    t_eval = np.linspace(time_span[0], time_span[1], n_points)
    solution = solve_ivp(ode_wrapper, time_span, initial_condition, t_eval=t_eval)

    if not solution.success:
        return "✗ Integration failed"

    trajectory = solution.y.T

    log += "## Trajectory Statistics:\n"
    for i in range(n_dims):
        log += f"- Dimension {i}: min={trajectory[:, i].min():.4f}, "
        log += f"max={trajectory[:, i].max():.4f}, "
        log += f"mean={trajectory[:, i].mean():.4f}, "
        log += f"std={trajectory[:, i].std():.4f}\n"

    # Estimate attractor dimension (correlation dimension approximation)
    log += "\n## Attractor Characterization:\n"

    # Compute trajectory length
    lengths = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    total_length = np.sum(lengths)
    log += f"- Total trajectory length: {total_length:.4f}\n"

    # Check if trajectory returns to near initial condition (periodicity test)
    final_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    log += f"- Distance from initial to final state: {final_distance:.4f}\n"

    if final_distance < 0.1 * np.std(trajectory):
        log += "  → Trajectory appears to be periodic or quasi-periodic\n"
    else:
        log += "  → Trajectory does not return to initial state (non-periodic)\n"

    return log
