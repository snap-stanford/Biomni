"""
Optimization and Mathematical Programming Tools.

This module provides functions for various optimization problems including
linear programming, nonlinear optimization, constrained optimization,
and metaheuristic algorithms.
"""

import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution, minimize_scalar
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds
import matplotlib.pyplot as plt


def solve_linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs'):
    """
    Solve a linear programming problem.

    Minimize: c^T @ x
    Subject to: A_ub @ x <= b_ub
                A_eq @ x == b_eq
                bounds[i][0] <= x[i] <= bounds[i][1]

    Parameters:
    -----------
    c : array-like
        Coefficients of the linear objective function
    A_ub : array-like, optional
        2D array for inequality constraint matrix
    b_ub : array-like, optional
        1D array for inequality constraint bounds
    A_eq : array-like, optional
        2D array for equality constraint matrix
    b_eq : array-like, optional
        1D array for equality constraint bounds
    bounds : sequence of tuples, optional
        Bounds for each variable (min, max)
    method : str
        Solver method: 'highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex'

    Returns:
    --------
    str
        Optimization results and summary
    """
    log = "# Linear Programming Solution\n\n"

    log += f"## Problem Setup:\n"
    log += f"- Objective function: minimize c^T @ x\n"
    log += f"- Number of variables: {len(c)}\n"
    log += f"- Objective coefficients: {c}\n"

    if A_ub is not None:
        log += f"- Inequality constraints: {np.array(A_ub).shape[0]}\n"
    if A_eq is not None:
        log += f"- Equality constraints: {np.array(A_eq).shape[0]}\n"

    log += f"- Method: {method}\n\n"

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method=method)

        log += "## Results:\n"
        if result.success:
            log += "✓ Optimization successful\n\n"
            log += f"**Optimal objective value:** {result.fun:.6f}\n\n"
            log += f"**Optimal solution:**\n```\n"
            for i, x in enumerate(result.x):
                log += f"x[{i}] = {x:.6f}\n"
            log += f"```\n\n"
            log += f"- Number of iterations: {result.nit}\n"
            log += f"- Status: {result.message}\n"
        else:
            log += f"✗ Optimization failed\n"
            log += f"- Status: {result.message}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def solve_nonlinear_optimization(objective, x0, constraints=None, bounds=None, method='SLSQP'):
    """
    Solve a nonlinear optimization problem.

    Parameters:
    -----------
    objective : callable
        Objective function to minimize f(x)
    x0 : array-like
        Initial guess
    constraints : list of dict, optional
        Constraints in the format expected by scipy.optimize.minimize
    bounds : sequence of tuples, optional
        Bounds for each variable
    method : str
        Optimization algorithm: 'SLSQP', 'COBYLA', 'trust-constr', 'L-BFGS-B', etc.

    Returns:
    --------
    str
        Optimization results

    Example:
    --------
    >>> # Minimize f(x,y) = (x-1)^2 + (y-2)^2 subject to x + y = 3
    >>> def objective(x):
    ...     return (x[0]-1)**2 + (x[1]-2)**2
    >>> constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] - 3}]
    >>> result = solve_nonlinear_optimization(objective, [0, 0], constraints=constraints)
    """
    log = "# Nonlinear Optimization Solution\n\n"

    log += f"## Problem Setup:\n"
    log += f"- Number of variables: {len(x0)}\n"
    log += f"- Initial guess: {x0}\n"
    log += f"- Method: {method}\n"

    if constraints:
        log += f"- Number of constraints: {len(constraints)}\n"
    if bounds:
        log += f"- Variable bounds specified: Yes\n"

    log += "\n"

    try:
        result = minimize(objective, x0, method=method, constraints=constraints, bounds=bounds)

        log += "## Results:\n"
        if result.success:
            log += "✓ Optimization converged successfully\n\n"
            log += f"**Optimal objective value:** {result.fun:.6f}\n\n"
            log += f"**Optimal solution:**\n```\n"
            for i, x in enumerate(result.x):
                log += f"x[{i}] = {x:.6f}\n"
            log += f"```\n\n"
            log += f"- Number of iterations: {result.nit}\n"
            log += f"- Number of function evaluations: {result.nfev}\n"
            log += f"- Status: {result.message}\n"
        else:
            log += f"⚠ Optimization did not converge\n"
            log += f"- Status: {result.message}\n"
            log += f"- Best objective found: {result.fun:.6f}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def global_optimization_differential_evolution(objective, bounds, maxiter=1000, popsize=15, seed=None):
    """
    Perform global optimization using differential evolution.

    This is useful for non-convex optimization problems with multiple local minima.

    Parameters:
    -----------
    objective : callable
        Objective function to minimize
    bounds : sequence of tuples
        Bounds for each variable (min, max)
    maxiter : int
        Maximum number of generations
    popsize : int
        Population size multiplier (total population = popsize * n_variables)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    str
        Optimization results
    """
    log = "# Global Optimization - Differential Evolution\n\n"

    n_vars = len(bounds)
    log += f"## Problem Setup:\n"
    log += f"- Number of variables: {n_vars}\n"
    log += f"- Variable bounds:\n"
    for i, (lb, ub) in enumerate(bounds):
        log += f"  - x[{i}]: [{lb}, {ub}]\n"
    log += f"- Maximum iterations: {maxiter}\n"
    log += f"- Population size: {popsize * n_vars}\n\n"

    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter,
                                       popsize=popsize, seed=seed, disp=False)

        log += "## Results:\n"
        if result.success:
            log += "✓ Optimization successful\n\n"
        else:
            log += "⚠ Maximum iterations reached\n\n"

        log += f"**Best objective value:** {result.fun:.6f}\n\n"
        log += f"**Best solution:**\n```\n"
        for i, x in enumerate(result.x):
            log += f"x[{i}] = {x:.6f}\n"
        log += f"```\n\n"
        log += f"- Number of iterations: {result.nit}\n"
        log += f"- Number of function evaluations: {result.nfev}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def gradient_descent_optimizer(objective, gradient, x0, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """
    Implement gradient descent optimization.

    Parameters:
    -----------
    objective : callable
        Objective function to minimize
    gradient : callable
        Gradient of the objective function
    x0 : array-like
        Initial point
    learning_rate : float
        Step size for gradient descent
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance

    Returns:
    --------
    str
        Optimization trajectory and results
    """
    log = "# Gradient Descent Optimization\n\n"

    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    obj_values = [objective(x)]

    log += f"## Setup:\n"
    log += f"- Initial point: {x0}\n"
    log += f"- Learning rate: {learning_rate}\n"
    log += f"- Maximum iterations: {max_iter}\n"
    log += f"- Convergence tolerance: {tolerance}\n\n"

    log += "## Optimization Progress:\n"

    for iteration in range(max_iter):
        grad = gradient(x)
        x_new = x - learning_rate * grad

        obj_new = objective(x_new)
        trajectory.append(x_new.copy())
        obj_values.append(obj_new)

        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance:
            log += f"✓ Converged at iteration {iteration}\n\n"
            x = x_new
            break

        x = x_new

        # Log progress every 100 iterations
        if iteration % 100 == 0:
            log += f"Iteration {iteration}: f(x) = {obj_new:.6f}, ||grad|| = {np.linalg.norm(grad):.6f}\n"

    else:
        log += f"⚠ Reached maximum iterations ({max_iter})\n\n"

    log += f"\n## Final Results:\n"
    log += f"**Optimal objective value:** {obj_values[-1]:.6f}\n\n"
    log += f"**Optimal solution:**\n```\n{x}\n```\n\n"
    log += f"- Total iterations: {len(trajectory) - 1}\n"
    log += f"- Objective improvement: {obj_values[0]:.6f} → {obj_values[-1]:.6f}\n"

    return log


def solve_least_squares_optimization(residual_func, x0, bounds=None, method='trf'):
    """
    Solve a nonlinear least squares problem.

    Minimize: 0.5 * sum(residual_func(x)^2)

    Parameters:
    -----------
    residual_func : callable
        Function that returns residual vector
    x0 : array-like
        Initial guess
    bounds : tuple of array-like, optional
        Lower and upper bounds on parameters
    method : str
        Algorithm: 'trf' (Trust Region Reflective), 'dogbox', or 'lm' (Levenberg-Marquardt)

    Returns:
    --------
    str
        Optimization results
    """
    from scipy.optimize import least_squares

    log = "# Nonlinear Least Squares Optimization\n\n"

    log += f"## Setup:\n"
    log += f"- Number of parameters: {len(x0)}\n"
    log += f"- Initial guess: {x0}\n"
    log += f"- Method: {method}\n\n"

    try:
        result = least_squares(residual_func, x0, bounds=bounds, method=method)

        log += "## Results:\n"
        if result.success:
            log += "✓ Optimization successful\n\n"
        else:
            log += "⚠ " + result.message + "\n\n"

        log += f"**Optimal cost (0.5 * ||residual||^2):** {result.cost:.6f}\n\n"
        log += f"**Optimal parameters:**\n```\n"
        for i, x in enumerate(result.x):
            log += f"x[{i}] = {x:.6f}\n"
        log += f"```\n\n"
        log += f"- Number of function evaluations: {result.nfev}\n"
        log += f"- Number of Jacobian evaluations: {result.njev}\n"
        log += f"- Optimality (gradient norm): {result.optimality:.2e}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def particle_swarm_optimization(objective, bounds, n_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    """
    Implement particle swarm optimization for global optimization.

    Parameters:
    -----------
    objective : callable
        Objective function to minimize
    bounds : list of tuples
        Bounds for each dimension (min, max)
    n_particles : int
        Number of particles in the swarm
    max_iter : int
        Maximum number of iterations
    w : float
        Inertia weight
    c1 : float
        Cognitive parameter (attraction to personal best)
    c2 : float
        Social parameter (attraction to global best)

    Returns:
    --------
    str
        Optimization results
    """
    log = "# Particle Swarm Optimization\n\n"

    n_dims = len(bounds)
    log += f"## Setup:\n"
    log += f"- Number of dimensions: {n_dims}\n"
    log += f"- Number of particles: {n_particles}\n"
    log += f"- Maximum iterations: {max_iter}\n"
    log += f"- Parameters: w={w}, c1={c1}, c2={c2}\n\n"

    # Initialize particles
    particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds],
                                 (n_particles, n_dims))
    velocities = np.zeros((n_particles, n_dims))

    # Initialize best positions
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([objective(p) for p in particles])

    global_best_position = personal_best_positions[personal_best_scores.argmin()]
    global_best_score = personal_best_scores.min()

    log += f"## Initial best score: {global_best_score:.6f}\n\n"
    log += "## Optimization Progress:\n"

    for iteration in range(max_iter):
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.random(2)
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best_positions[i] - particles[i]) +
                           c2 * r2 * (global_best_position - particles[i]))

            # Update position
            particles[i] += velocities[i]

            # Apply bounds
            particles[i] = np.clip(particles[i], [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate
            score = objective(particles[i])

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()

        if iteration % 10 == 0:
            log += f"Iteration {iteration}: Best score = {global_best_score:.6f}\n"

    log += f"\n## Final Results:\n"
    log += f"**Best objective value:** {global_best_score:.6f}\n\n"
    log += f"**Best solution:**\n```\n"
    for i, x in enumerate(global_best_position):
        log += f"x[{i}] = {x:.6f}\n"
    log += f"```\n"

    return log


def multi_objective_pareto_front(objective_funcs, bounds, n_samples=1000):
    """
    Approximate the Pareto front for multi-objective optimization.

    Parameters:
    -----------
    objective_funcs : list of callables
        List of objective functions to minimize
    bounds : list of tuples
        Bounds for each decision variable
    n_samples : int
        Number of random samples to generate

    Returns:
    --------
    str
        Description of the Pareto front
    """
    log = "# Multi-Objective Optimization - Pareto Front\n\n"

    n_objectives = len(objective_funcs)
    n_dims = len(bounds)

    log += f"## Setup:\n"
    log += f"- Number of objectives: {n_objectives}\n"
    log += f"- Number of decision variables: {n_dims}\n"
    log += f"- Number of samples: {n_samples}\n\n"

    # Generate random samples
    samples = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds],
                               (n_samples, n_dims))

    # Evaluate objectives
    objectives = np.array([[f(x) for f in objective_funcs] for x in samples])

    # Find Pareto-optimal solutions
    is_pareto = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # Check if j dominates i
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_pareto[i] = False
                    break

    pareto_solutions = samples[is_pareto]
    pareto_objectives = objectives[is_pareto]

    log += f"## Results:\n"
    log += f"- Number of Pareto-optimal solutions found: {len(pareto_solutions)}\n\n"

    log += "## Sample Pareto-Optimal Solutions:\n"
    for i in range(min(5, len(pareto_solutions))):
        log += f"\n### Solution {i+1}:\n"
        log += f"Decision variables: {pareto_solutions[i]}\n"
        log += f"Objectives: {pareto_objectives[i]}\n"

    return log
