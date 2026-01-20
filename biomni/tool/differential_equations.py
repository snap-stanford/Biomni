"""
Differential Equations and Dynamical Systems Modeling Tools.

This module provides functions for solving and analyzing differential equations,
including ODEs, PDEs, and systems of differential equations.
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve


def solve_ode_system(
    func, initial_conditions, time_span, parameters=None, method="RK45", output_file="ode_solution.csv"
):
    """
    Solve a system of ordinary differential equations (ODEs).

    Parameters:
    -----------
    func : callable
        Function defining the system of ODEs. Should have signature func(t, y, *parameters)
        where t is time, y is the state vector, and parameters are additional arguments.
    initial_conditions : list or array
        Initial conditions for the state variables
    time_span : tuple or array
        Either (t_start, t_end) or an array of time points
    parameters : tuple, optional
        Additional parameters to pass to the ODE function
    method : str, optional
        Integration method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    output_file : str, optional
        File to save the solution

    Returns:
    --------
    str
        Summary of the ODE solution process and results

    Example:
    --------
    >>> # Solve the Lorenz system
    >>> def lorenz(t, y, sigma, rho, beta):
    ...     x, y, z = y
    ...     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    >>> result = solve_ode_system(lorenz, [1, 1, 1], (0, 50), parameters=(10, 28, 8 / 3))
    """
    import pandas as pd

    log = "# ODE System Solution Log\n\n"

    # Prepare time points
    if isinstance(time_span, tuple) and len(time_span) == 2:
        t_eval = np.linspace(time_span[0], time_span[1], 1000)
        log += f"## Time span: {time_span[0]} to {time_span[1]} with 1000 evaluation points\n\n"
    else:
        t_eval = np.array(time_span)
        log += f"## Using {len(t_eval)} specified time points\n\n"

    # Wrap function to handle parameters
    if parameters:

        def ode_func(t, y):
            return func(t, y, *parameters)
    else:
        ode_func = func

    log += f"## Initial conditions: {initial_conditions}\n"
    log += f"## Integration method: {method}\n\n"

    # Solve the ODE system
    try:
        solution = solve_ivp(
            ode_func, (t_eval[0], t_eval[-1]), initial_conditions, method=method, t_eval=t_eval, dense_output=True
        )

        if solution.success:
            log += "✓ ODE system solved successfully\n\n"

            # Save solution to file
            df = pd.DataFrame(solution.y.T, columns=[f"y{i}" for i in range(len(initial_conditions))])
            df.insert(0, "t", solution.t)
            df.to_csv(output_file, index=False)

            log += f"## Solution saved to {output_file}\n"
            log += f"- Number of state variables: {len(initial_conditions)}\n"
            log += f"- Number of time points: {len(solution.t)}\n"
            log += f"- Final state: {solution.y[:, -1]}\n\n"

            # Provide basic statistics
            log += "## State variable statistics:\n"
            for i in range(len(initial_conditions)):
                log += f"- Variable y{i}: min={solution.y[i].min():.4f}, max={solution.y[i].max():.4f}, "
                log += f"mean={solution.y[i].mean():.4f}, std={solution.y[i].std():.4f}\n"
        else:
            log += f"✗ ODE solver failed: {solution.message}\n"

    except Exception as e:
        log += f"✗ Error solving ODE system: {str(e)}\n"

    return log


def solve_pde_heat_equation(
    initial_condition, boundary_conditions, domain, time_span, diffusivity=1.0, output_file="pde_solution.csv"
):
    """
    Solve the 1D heat equation (a prototypical parabolic PDE).

    ∂u/∂t = α * ∂²u/∂x²

    Parameters:
    -----------
    initial_condition : callable or array
        Initial temperature distribution u(x, 0)
    boundary_conditions : tuple
        (left_bc, right_bc) where each is either a value (Dirichlet) or callable
    domain : tuple
        (x_min, x_max, n_points) defining the spatial domain
    time_span : tuple
        (t_start, t_end, n_steps) defining the time domain
    diffusivity : float, optional
        Thermal diffusivity coefficient α
    output_file : str, optional
        File to save the solution

    Returns:
    --------
    str
        Summary of the PDE solution
    """
    import pandas as pd

    log = "# Heat Equation PDE Solution Log\n\n"

    x_min, x_max, nx = domain
    t_start, t_end, nt = time_span

    dx = (x_max - x_min) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)

    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_start, t_end, nt)

    # Stability check for explicit method
    stability_criterion = diffusivity * dt / dx**2
    log += f"## Domain: x ∈ [{x_min}, {x_max}] with {nx} points (dx={dx:.4f})\n"
    log += f"## Time: t ∈ [{t_start}, {t_end}] with {nt} steps (dt={dt:.4f})\n"
    log += f"## Diffusivity α = {diffusivity}\n"
    log += f"## Stability criterion: α*dt/dx² = {stability_criterion:.4f} (must be ≤ 0.5 for stability)\n\n"

    if stability_criterion > 0.5:
        log += "⚠ WARNING: Stability criterion violated! Solution may be unstable.\n\n"

    # Initialize solution array
    u = np.zeros((nt, nx))

    # Set initial condition
    if callable(initial_condition):
        u[0, :] = initial_condition(x)
    else:
        u[0, :] = initial_condition

    # Set boundary conditions
    left_bc, right_bc = boundary_conditions

    # Time stepping using explicit finite difference
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + diffusivity * dt / dx**2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

        # Apply boundary conditions
        u[n + 1, 0] = left_bc(t[n + 1]) if callable(left_bc) else left_bc
        u[n + 1, -1] = right_bc(t[n + 1]) if callable(right_bc) else right_bc

    # Save solution
    df = pd.DataFrame(u, columns=[f"x_{i}" for i in range(nx)])
    df.insert(0, "t", t)
    df.to_csv(output_file, index=False)

    log += "✓ PDE solution computed successfully\n"
    log += f"## Solution saved to {output_file}\n"
    log += f"- Solution shape: {nt} time steps × {nx} spatial points\n"
    log += f"- Final temperature range: [{u[-1].min():.4f}, {u[-1].max():.4f}]\n"

    return log


def find_equilibrium_points(func, search_region, parameters=None):
    """
    Find equilibrium (steady-state) points of a dynamical system.

    Equilibrium points are where df/dt = 0.

    Parameters:
    -----------
    func : callable
        Function defining the system dynamics
    search_region : list of tuples
        List of (min, max) ranges for each variable to search
    parameters : tuple, optional
        Additional parameters for the function

    Returns:
    --------
    str
        Summary of found equilibrium points
    """
    log = "# Equilibrium Point Analysis\n\n"

    # Create wrapper for fsolve
    if parameters:

        def system(y):
            return func(0, y, *parameters)
    else:

        def system(y):
            return func(0, y)

    equilibria = []

    # Try multiple initial guesses in the search region
    n_attempts = 20
    for i in range(n_attempts):
        # Random initial guess within search region
        guess = [np.random.uniform(low, high) for low, high in search_region]

        try:
            eq = fsolve(system, guess, full_output=True)
            solution, info, ier, msg = eq

            if ier == 1:  # Solution found
                # Check if this is a new equilibrium (not already found)
                is_new = True
                for existing_eq in equilibria:
                    if np.allclose(solution, existing_eq, atol=1e-6):
                        is_new = False
                        break

                if is_new:
                    equilibria.append(solution)
        except:
            pass

    log += f"## Search conducted with {n_attempts} random initial guesses\n"
    log += f"## Found {len(equilibria)} unique equilibrium point(s):\n\n"

    for i, eq in enumerate(equilibria, 1):
        log += f"### Equilibrium {i}:\n"
        log += f"```\n{eq}\n```\n"

        # Evaluate the function at this point to verify
        residual = system(eq)
        log += f"Residual norm: {np.linalg.norm(residual):.2e}\n\n"

    if len(equilibria) == 0:
        log += "No equilibrium points found in the search region.\n"

    return log


def analyze_linear_stability(jacobian_func, equilibrium_point, parameters=None):
    """
    Perform linear stability analysis at an equilibrium point.

    Parameters:
    -----------
    jacobian_func : callable
        Function that returns the Jacobian matrix at a given point
    equilibrium_point : array-like
        The equilibrium point to analyze
    parameters : tuple, optional
        Additional parameters for the Jacobian function

    Returns:
    --------
    str
        Stability analysis results including eigenvalues and classification
    """
    log = "# Linear Stability Analysis\n\n"

    log += f"## Equilibrium point: {equilibrium_point}\n\n"

    # Compute Jacobian at equilibrium
    if parameters:
        J = jacobian_func(equilibrium_point, *parameters)
    else:
        J = jacobian_func(equilibrium_point)

    log += "## Jacobian matrix:\n```\n"
    log += str(J) + "\n```\n\n"

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    log += "## Eigenvalues:\n"
    for i, eig in enumerate(eigenvalues, 1):
        if np.isreal(eig):
            log += f"{i}. λ_{i} = {eig.real:.4f}\n"
        else:
            log += f"{i}. λ_{i} = {eig.real:.4f} + {eig.imag:.4f}i\n"

    log += "\n## Stability Classification:\n"

    # Classify stability
    max_real = max(eig.real for eig in eigenvalues)

    if max_real < 0:
        log += "✓ **STABLE** - All eigenvalues have negative real parts\n"
        log += "  The system will return to equilibrium after small perturbations.\n\n"
    elif max_real > 0:
        log += "✗ **UNSTABLE** - At least one eigenvalue has positive real part\n"
        log += "  The system will diverge from equilibrium after small perturbations.\n\n"
    else:
        log += "⚠ **MARGINALLY STABLE** - Eigenvalues on imaginary axis\n"
        log += "  Further analysis required (center, non-hyperbolic point).\n\n"

    # Additional classification for 2D systems
    if len(eigenvalues) == 2:
        log += "## 2D System Classification:\n"
        eig1, eig2 = eigenvalues

        if np.isreal(eig1) and np.isreal(eig2):
            if eig1.real < 0 and eig2.real < 0:
                log += "- Type: **Stable Node**\n"
            elif eig1.real > 0 and eig2.real > 0:
                log += "- Type: **Unstable Node**\n"
            else:
                log += "- Type: **Saddle Point** (one stable, one unstable direction)\n"
        else:
            if eig1.real < 0:
                log += "- Type: **Stable Spiral/Focus**\n"
            elif eig1.real > 0:
                log += "- Type: **Unstable Spiral/Focus**\n"
            else:
                log += "- Type: **Center** (neutrally stable, periodic orbits)\n"

    return log


def solve_boundary_value_problem(ode_func, boundary_conditions, domain, initial_guess=None, parameters=None):
    """
    Solve a boundary value problem (BVP) for ODEs.

    Parameters:
    -----------
    ode_func : callable
        Function defining dy/dx = f(x, y, parameters)
    boundary_conditions : callable
        Function that returns boundary condition residuals
    domain : tuple
        (x_start, x_end) defining the domain
    initial_guess : array, optional
        Initial guess for the solution
    parameters : tuple, optional
        Additional parameters for the ODE

    Returns:
    --------
    str
        Summary of the BVP solution
    """
    log = "# Boundary Value Problem Solution\n\n"

    x_start, x_end = domain
    x = np.linspace(x_start, x_end, 100)

    log += f"## Domain: x ∈ [{x_start}, {x_end}]\n\n"

    # Create initial guess if not provided
    if initial_guess is None:
        # Simple linear guess
        n_vars = 2  # Assume 2 variables if not specified
        initial_guess = np.zeros((n_vars, len(x)))
        log += "Using zero initial guess\n\n"

    # Wrap ODE function
    if parameters:

        def ode_wrapper(x, y):
            return ode_func(x, y, *parameters)
    else:
        ode_wrapper = ode_func

    try:
        # Solve BVP
        solution = solve_bvp(ode_wrapper, boundary_conditions, x, initial_guess)

        if solution.success:
            log += "✓ BVP solved successfully\n\n"
            log += "## Solution Statistics:\n"
            log += f"- Number of variables: {solution.y.shape[0]}\n"
            log += f"- Number of mesh points: {solution.x.shape[0]}\n"
            log += f"- RMS residual: {solution.rms_residuals.max():.2e}\n"
        else:
            log += f"✗ BVP solver failed: {solution.message}\n"

    except Exception as e:
        log += f"✗ Error solving BVP: {str(e)}\n"

    return log


def simulate_predator_prey_model(
    prey_initial,
    predator_initial,
    time_span,
    alpha=1.0,
    beta=0.1,
    gamma=1.5,
    delta=0.075,
    output_file="predator_prey.csv",
):
    """
    Simulate the Lotka-Volterra predator-prey model.

    dx/dt = αx - βxy  (prey)
    dy/dt = δxy - γy  (predator)

    Parameters:
    -----------
    prey_initial : float
        Initial prey population
    predator_initial : float
        Initial predator population
    time_span : tuple
        (t_start, t_end) time range
    alpha : float
        Prey growth rate
    beta : float
        Predation rate
    gamma : float
        Predator death rate
    delta : float
        Predator reproduction rate per prey eaten
    output_file : str
        File to save results

    Returns:
    --------
    str
        Summary of the simulation
    """

    def lotka_volterra(t, y, alpha, beta, gamma, delta):
        x, y_pred = y
        dx_dt = alpha * x - beta * x * y_pred
        dy_dt = delta * x * y_pred - gamma * y_pred
        return [dx_dt, dy_dt]

    result = solve_ode_system(
        lotka_volterra,
        [prey_initial, predator_initial],
        time_span,
        parameters=(alpha, beta, gamma, delta),
        output_file=output_file,
    )

    log = "# Lotka-Volterra Predator-Prey Model\n\n"
    log += "## Parameters:\n"
    log += f"- α (prey growth rate) = {alpha}\n"
    log += f"- β (predation rate) = {beta}\n"
    log += f"- γ (predator death rate) = {gamma}\n"
    log += f"- δ (predator efficiency) = {delta}\n\n"
    log += "## Initial Conditions:\n"
    log += f"- Prey population: {prey_initial}\n"
    log += f"- Predator population: {predator_initial}\n\n"
    log += result

    return log
