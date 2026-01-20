"""
Numerical Methods and Computational Techniques.

This module provides numerical algorithms for integration, differentiation,
root finding, interpolation, and other computational mathematics tasks.
"""

import numpy as np
from scipy import integrate, interpolate, optimize
from scipy.misc import derivative


def numerical_integration(func, a, b, method="quad", n_points=100, params=()):
    """
    Numerically integrate a function over an interval.

    Parameters:
    -----------
    func : callable
        Function to integrate
    a, b : float
        Integration bounds
    method : str
        Integration method: 'quad' (adaptive), 'trapz' (trapezoidal), 'simps' (Simpson's)
    n_points : int
        Number of evaluation points (for trapz and simps)
    params : tuple
        Additional parameters for the function

    Returns:
    --------
    str
        Integration results
    """
    log = "# Numerical Integration\n\n"

    log += "## Setup:\n"
    log += f"- Integration bounds: [{a}, {b}]\n"
    log += f"- Method: {method}\n"

    if params:
        func_with_params = lambda x: func(x, *params)
    else:
        func_with_params = func

    try:
        if method == "quad":
            result, error = integrate.quad(func_with_params, a, b)
            log += "\n## Results (Adaptive Quadrature):\n"
            log += f"- Integral value: {result:.10f}\n"
            log += f"- Estimated error: {error:.2e}\n"

        elif method == "trapz":
            x = np.linspace(a, b, n_points)
            y = np.array([func_with_params(xi) for xi in x])
            result = integrate.trapz(y, x)
            log += f"- Number of points: {n_points}\n"
            log += "\n## Results (Trapezoidal Rule):\n"
            log += f"- Integral value: {result:.10f}\n"

        elif method == "simps":
            x = np.linspace(a, b, n_points if n_points % 2 == 1 else n_points + 1)
            y = np.array([func_with_params(xi) for xi in x])
            result = integrate.simpson(y, x)
            log += f"- Number of points: {len(x)}\n"
            log += "\n## Results (Simpson's Rule):\n"
            log += f"- Integral value: {result:.10f}\n"

        else:
            return f"Error: Unknown method '{method}'"

    except Exception as e:
        log += f"\n✗ Error: {str(e)}\n"

    return log


def numerical_derivative(func, x0, method="central", h=1e-5, order=1):
    """
    Compute numerical derivative of a function at a point.

    Parameters:
    -----------
    func : callable
        Function to differentiate
    x0 : float
        Point at which to compute derivative
    method : str
        Method: 'forward', 'backward', 'central'
    h : float
        Step size
    order : int
        Order of derivative (1, 2, 3, or 4)

    Returns:
    --------
    str
        Derivative results
    """
    log = "# Numerical Differentiation\n\n"

    log += "## Setup:\n"
    log += f"- Point: x = {x0}\n"
    log += f"- Method: {method} difference\n"
    log += f"- Step size: h = {h}\n"
    log += f"- Order: {order}\n\n"

    try:
        # Use scipy's derivative function
        result = derivative(func, x0, dx=h, n=order, order=3 if method == "central" else 1)

        log += "## Result:\n"
        log += f"- f^({order})(x0) ≈ {result:.10f}\n"

        # Also compute using finite difference formulas for comparison
        if order == 1:
            if method == "forward":
                manual = (func(x0 + h) - func(x0)) / h
            elif method == "backward":
                manual = (func(x0) - func(x0 - h)) / h
            else:  # central
                manual = (func(x0 + h) - func(x0 - h)) / (2 * h)

            log += f"- Manual {method} difference: {manual:.10f}\n"
            log += f"- Difference: {abs(result - manual):.2e}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def find_roots(func, bracket=None, x0=None, method="brentq", params=()):
    """
    Find roots (zeros) of a function.

    Parameters:
    -----------
    func : callable
        Function for which to find roots
    bracket : tuple, optional
        Bracketing interval [a, b] where func(a) and func(b) have opposite signs
    x0 : float, optional
        Initial guess (for methods that don't use bracketing)
    method : str
        Root-finding method: 'brentq', 'newton', 'fsolve', 'bisect'
    params : tuple
        Additional parameters for the function

    Returns:
    --------
    str
        Root-finding results
    """
    log = "# Root Finding\n\n"

    log += f"## Method: {method}\n"

    if params:
        func_with_params = lambda x: func(x, *params)
    else:
        func_with_params = func

    try:
        if method == "brentq":
            if bracket is None:
                return "Error: brentq method requires a bracket [a, b]"
            a, b = bracket
            log += f"- Bracket: [{a}, {b}]\n"
            log += f"- f(a) = {func_with_params(a):.6f}\n"
            log += f"- f(b) = {func_with_params(b):.6f}\n\n"

            root, result = optimize.brentq(func_with_params, a, b, full_output=True)

            log += "## Result:\n"
            log += f"- Root: x = {root:.10f}\n"
            log += f"- f(root) = {func_with_params(root):.2e}\n"
            log += f"- Iterations: {result.iterations}\n"
            log += f"- Function calls: {result.function_calls}\n"

        elif method == "bisect":
            if bracket is None:
                return "Error: bisect method requires a bracket [a, b]"
            a, b = bracket
            log += f"- Bracket: [{a}, {b}]\n\n"

            root = optimize.bisect(func_with_params, a, b)

            log += "## Result:\n"
            log += f"- Root: x = {root:.10f}\n"
            log += f"- f(root) = {func_with_params(root):.2e}\n"

        elif method == "newton":
            if x0 is None:
                return "Error: newton method requires an initial guess x0"
            log += f"- Initial guess: x0 = {x0}\n\n"

            root = optimize.newton(func_with_params, x0)

            log += "## Result:\n"
            log += f"- Root: x = {root:.10f}\n"
            log += f"- f(root) = {func_with_params(root):.2e}\n"

        elif method == "fsolve":
            if x0 is None:
                return "Error: fsolve method requires an initial guess x0"
            log += f"- Initial guess: x0 = {x0}\n\n"

            root, info, ier, msg = optimize.fsolve(func_with_params, x0, full_output=True)
            root = root[0] if len(root) == 1 else root

            log += "## Result:\n"
            log += f"- Root: x = {root:.10f}\n"
            log += f"- f(root) = {func_with_params(root):.2e}\n"
            log += f"- Message: {msg}\n"

        else:
            return f"Error: Unknown method '{method}'"

    except Exception as e:
        log += f"\n✗ Error: {str(e)}\n"

    return log


def interpolate_data(x_data, y_data, x_eval, method="cubic"):
    """
    Interpolate data points to evaluate at new points.

    Parameters:
    -----------
    x_data, y_data : array-like
        Known data points
    x_eval : array-like
        Points at which to evaluate the interpolant
    method : str
        Interpolation method: 'linear', 'cubic', 'quadratic', 'nearest'

    Returns:
    --------
    str
        Interpolation results
    """
    log = "# Data Interpolation\n\n"

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_eval = np.array(x_eval)

    log += "## Setup:\n"
    log += f"- Number of data points: {len(x_data)}\n"
    log += f"- Number of evaluation points: {len(x_eval)}\n"
    log += f"- Method: {method}\n"
    log += f"- Data range: x ∈ [{x_data.min():.4f}, {x_data.max():.4f}]\n"
    log += f"- Evaluation range: x ∈ [{x_eval.min():.4f}, {x_eval.max():.4f}]\n\n"

    try:
        if method == "linear":
            interp_func = interpolate.interp1d(x_data, y_data, kind="linear")
        elif method == "cubic":
            interp_func = interpolate.interp1d(x_data, y_data, kind="cubic")
        elif method == "quadratic":
            interp_func = interpolate.interp1d(x_data, y_data, kind="quadratic")
        elif method == "nearest":
            interp_func = interpolate.interp1d(x_data, y_data, kind="nearest")
        else:
            return f"Error: Unknown method '{method}'"

        y_eval = interp_func(x_eval)

        log += "## Results:\n"
        log += "✓ Interpolation successful\n\n"
        log += "## Sample interpolated values:\n"
        n_samples = min(10, len(x_eval))
        for i in range(n_samples):
            log += f"- f({x_eval[i]:.4f}) = {y_eval[i]:.4f}\n"

        log += "\n## Interpolated value statistics:\n"
        log += f"- Min: {y_eval.min():.4f}\n"
        log += f"- Max: {y_eval.max():.4f}\n"
        log += f"- Mean: {y_eval.mean():.4f}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def solve_nonlinear_system(equations, initial_guess, jacobian=None):
    """
    Solve a system of nonlinear equations.

    Parameters:
    -----------
    equations : callable
        Function that returns the system of equations F(x) = 0
    initial_guess : array-like
        Initial guess for the solution
    jacobian : callable, optional
        Jacobian matrix function (if not provided, will be approximated)

    Returns:
    --------
    str
        Solution results
    """
    log = "# Nonlinear System Solution\n\n"

    initial_guess = np.array(initial_guess)
    n = len(initial_guess)

    log += "## Setup:\n"
    log += f"- Number of equations: {n}\n"
    log += f"- Initial guess: {initial_guess}\n"
    log += f"- Jacobian provided: {'Yes' if jacobian else 'No (will be approximated)'}\n\n"

    try:
        if jacobian:
            solution = optimize.fsolve(equations, initial_guess, fprime=jacobian, full_output=True)
        else:
            solution = optimize.fsolve(equations, initial_guess, full_output=True)

        x, info, ier, msg = solution

        log += "## Results:\n"
        if ier == 1:
            log += "✓ Solution converged\n\n"
        else:
            log += "⚠ Solution may not have converged\n\n"

        log += "**Solution:**\n```\n"
        for i, val in enumerate(x):
            log += f"x[{i}] = {val:.10f}\n"
        log += "```\n\n"

        # Verify solution
        residual = equations(x)
        residual_norm = np.linalg.norm(residual)

        log += "## Verification:\n"
        log += f"- Residual norm: {residual_norm:.2e}\n"
        log += f"- Number of function calls: {info['nfev']}\n"
        log += f"- Message: {msg}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def richardson_extrapolation(func, x, h, order=2, n_levels=4):
    """
    Use Richardson extrapolation to improve numerical derivative estimate.

    Parameters:
    -----------
    func : callable
        Function to differentiate
    x : float
        Point at which to compute derivative
    h : float
        Initial step size
    order : int
        Order of derivative
    n_levels : int
        Number of extrapolation levels

    Returns:
    --------
    str
        Extrapolated derivative estimate
    """
    log = "# Richardson Extrapolation\n\n"

    log += "## Setup:\n"
    log += f"- Point: x = {x}\n"
    log += f"- Initial step size: h = {h}\n"
    log += f"- Derivative order: {order}\n"
    log += f"- Extrapolation levels: {n_levels}\n\n"

    # Create Richardson extrapolation table
    D = np.zeros((n_levels, n_levels))

    log += "## Extrapolation Table:\n"

    # First column: central difference approximations with decreasing h
    for i in range(n_levels):
        h_i = h / (2**i)
        if order == 1:
            D[i, 0] = (func(x + h_i) - func(x - h_i)) / (2 * h_i)
        elif order == 2:
            D[i, 0] = (func(x + h_i) - 2 * func(x) + func(x - h_i)) / h_i**2

        log += f"D[{i},0] (h/{2**i}): {D[i, 0]:.10f}\n"

    # Richardson extrapolation
    for j in range(1, n_levels):
        for i in range(n_levels - j):
            D[i, j] = D[i + 1, j - 1] + (D[i + 1, j - 1] - D[i, j - 1]) / (4**j - 1)
        log += f"\nD[{i},{j}]: {D[0, j]:.10f}\n"

    best_estimate = D[0, n_levels - 1]

    log += "\n## Best Estimate:\n"
    log += f"- f^({order})(x) ≈ {best_estimate:.10f}\n"

    return log


def adaptive_integration(func, a, b, tol=1e-6, max_depth=20):
    """
    Adaptive integration using recursive subdivision.

    Parameters:
    -----------
    func : callable
        Function to integrate
    a, b : float
        Integration bounds
    tol : float
        Error tolerance
    max_depth : int
        Maximum recursion depth

    Returns:
    --------
    str
        Integration results with adaptive mesh information
    """
    log = "# Adaptive Integration\n\n"

    log += "## Setup:\n"
    log += f"- Bounds: [{a}, {b}]\n"
    log += f"- Tolerance: {tol}\n"
    log += f"- Maximum depth: {max_depth}\n\n"

    intervals = []

    def adaptive_quad(f, a, b, tol, depth):
        """Recursive adaptive quadrature."""
        # Compute Simpson's rule estimate
        c = (a + b) / 2
        fa, fb, fc = f(a), f(b), f(c)
        h = b - a

        S = h / 6 * (fa + 4 * fc + fb)

        # Subdivide
        d = (a + c) / 2
        e = (c + b) / 2
        fd, fe = f(d), f(e)

        S_left = (c - a) / 6 * (fa + 4 * fd + fc)
        S_right = (b - c) / 6 * (fc + 4 * fe + fb)
        S2 = S_left + S_right

        error = abs(S2 - S) / 15

        if error < tol or depth >= max_depth:
            intervals.append((a, b, depth, error))
            return S2

        return adaptive_quad(f, a, c, tol / 2, depth + 1) + adaptive_quad(f, c, b, tol / 2, depth + 1)

    try:
        result = adaptive_quad(func, a, b, tol, 0)

        log += "## Results:\n"
        log += f"- Integral value: {result:.10f}\n"
        log += f"- Number of intervals: {len(intervals)}\n"
        log += f"- Maximum depth reached: {max([iv[2] for iv in intervals])}\n"
        log += f"- Estimated total error: {sum([iv[3] for iv in intervals]):.2e}\n\n"

        log += "## Interval Statistics:\n"
        depths = [iv[2] for iv in intervals]
        log += "- Depth distribution:\n"
        for d in range(max(depths) + 1):
            count = depths.count(d)
            if count > 0:
                log += f"  - Depth {d}: {count} intervals\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log
