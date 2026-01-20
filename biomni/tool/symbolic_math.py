"""
Symbolic Mathematics and Computer Algebra.

This module provides symbolic computation capabilities including
algebraic manipulation, calculus, equation solving, and symbolic analysis.
Requires sympy library.
"""

import numpy as np


def symbolic_differentiation(expression_str, variable, order=1):
    """
    Compute symbolic derivative of an expression.

    Parameters:
    -----------
    expression_str : str
        Mathematical expression as string (e.g., "x**2 + sin(x)")
    variable : str
        Variable to differentiate with respect to
    order : int
        Order of derivative

    Returns:
    --------
    str
        Symbolic derivative result
    """
    try:
        from sympy import symbols, sympify, diff, latex
    except ImportError:
        return "Error: sympy library not available. Install with: pip install sympy"

    log = "# Symbolic Differentiation\n\n"

    log += f"## Expression: {expression_str}\n"
    log += f"## Variable: {variable}\n"
    log += f"## Order: {order}\n\n"

    try:
        # Create symbol and parse expression
        var = symbols(variable)
        expr = sympify(expression_str)

        log += f"Parsed expression: {expr}\n\n"

        # Compute derivative
        derivative = diff(expr, var, order)

        log += "## Result:\n"
        log += f"```\nd^{order}/d{variable}^{order} [{expr}] = {derivative}\n```\n\n"

        # Simplify
        simplified = derivative.simplify()
        if simplified != derivative:
            log += f"## Simplified:\n```\n{simplified}\n```\n\n"

        # LaTeX representation
        log += f"## LaTeX:\n```latex\n{latex(derivative)}\n```\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def symbolic_integration(expression_str, variable, definite=False, limits=None):
    """
    Compute symbolic integral of an expression.

    Parameters:
    -----------
    expression_str : str
        Expression to integrate
    variable : str
        Integration variable
    definite : bool
        Whether to compute definite integral
    limits : tuple, optional
        (lower, upper) limits for definite integral

    Returns:
    --------
    str
        Symbolic integral result
    """
    try:
        from sympy import symbols, sympify, integrate, latex, oo
    except ImportError:
        return "Error: sympy library not available"

    log = "# Symbolic Integration\n\n"

    log += f"## Expression: {expression_str}\n"
    log += f"## Variable: {variable}\n"

    try:
        var = symbols(variable)
        expr = sympify(expression_str)

        log += f"Parsed expression: {expr}\n\n"

        if definite and limits:
            # Parse limits (handle infinity)
            lower = sympify(str(limits[0])) if limits[0] != 'oo' else oo
            upper = sympify(str(limits[1])) if limits[1] != 'oo' else oo

            log += f"## Definite Integral from {limits[0]} to {limits[1]}:\n"

            result = integrate(expr, (var, lower, upper))

            log += f"```\n∫[{limits[0]} to {limits[1]}] {expr} d{variable} = {result}\n```\n\n"

            # Try to evaluate numerically if possible
            try:
                numerical_value = float(result.evalf())
                log += f"Numerical value: {numerical_value:.6f}\n\n"
            except:
                pass

        else:
            log += "## Indefinite Integral:\n"
            result = integrate(expr, var)

            log += f"```\n∫ {expr} d{variable} = {result} + C\n```\n\n"

        # LaTeX
        log += f"## LaTeX:\n```latex\n{latex(result)}\n```\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def solve_symbolic_equation(equation_str, variable):
    """
    Solve a symbolic equation.

    Parameters:
    -----------
    equation_str : str
        Equation to solve (e.g., "x**2 - 4 = 0" or "x**2 - 4")
    variable : str
        Variable to solve for

    Returns:
    --------
    str
        Solutions
    """
    try:
        from sympy import symbols, sympify, solve, latex
    except ImportError:
        return "Error: sympy library not available"

    log = "# Symbolic Equation Solving\n\n"

    log += f"## Equation: {equation_str}\n"
    log += f"## Solve for: {variable}\n\n"

    try:
        var = symbols(variable)

        # Handle equation format
        if '=' in equation_str:
            lhs, rhs = equation_str.split('=')
            expr = sympify(lhs) - sympify(rhs)
        else:
            expr = sympify(equation_str)

        log += f"Solving: {expr} = 0\n\n"

        # Solve
        solutions = solve(expr, var)

        log += f"## Solutions ({len(solutions)} found):\n"

        if len(solutions) == 0:
            log += "No solutions found.\n"
        else:
            for i, sol in enumerate(solutions, 1):
                log += f"{i}. {variable} = {sol}\n"

                # Try to evaluate numerically
                try:
                    numerical = complex(sol.evalf())
                    if abs(numerical.imag) < 1e-10:
                        log += f"   ≈ {numerical.real:.6f}\n"
                    else:
                        log += f"   ≈ {numerical:.6f}\n"
                except:
                    pass

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def symbolic_series_expansion(expression_str, variable, point=0, order=6):
    """
    Compute Taylor/Laurent series expansion of an expression.

    Parameters:
    -----------
    expression_str : str
        Expression to expand
    variable : str
        Expansion variable
    point : float or str
        Point around which to expand
    order : int
        Order of expansion

    Returns:
    --------
    str
        Series expansion
    """
    try:
        from sympy import symbols, sympify, series, latex
    except ImportError:
        return "Error: sympy library not available"

    log = "# Series Expansion\n\n"

    log += f"## Expression: {expression_str}\n"
    log += f"## Variable: {variable}\n"
    log += f"## Expansion point: {point}\n"
    log += f"## Order: {order}\n\n"

    try:
        var = symbols(variable)
        expr = sympify(expression_str)
        point_val = sympify(str(point))

        log += f"Computing Taylor series of {expr} around {variable}={point}...\n\n"

        # Compute series
        series_expansion = series(expr, var, point_val, order)

        log += "## Series Expansion:\n"
        log += f"```\n{series_expansion}\n```\n\n"

        # Extract polynomial part (remove O() term)
        polynomial = series_expansion.removeO()

        log += "## Polynomial Approximation:\n"
        log += f"```\n{polynomial}\n```\n\n"

        log += f"## LaTeX:\n```latex\n{latex(polynomial)}\n```\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def symbolic_limit(expression_str, variable, point, direction='+-'):
    """
    Compute symbolic limit of an expression.

    Parameters:
    -----------
    expression_str : str
        Expression
    variable : str
        Variable
    point : str
        Limit point (can be 'oo' for infinity)
    direction : str
        Direction: '+-' (both sides), '+' (from right), '-' (from left)

    Returns:
    --------
    str
        Limit result
    """
    try:
        from sympy import symbols, sympify, limit, latex, oo
    except ImportError:
        return "Error: sympy library not available"

    log = "# Symbolic Limit\n\n"

    log += f"## Expression: {expression_str}\n"
    log += f"## Variable: {variable} → {point}\n"
    log += f"## Direction: {direction}\n\n"

    try:
        var = symbols(variable)
        expr = sympify(expression_str)

        # Parse point
        if point == 'oo':
            point_val = oo
        elif point == '-oo':
            point_val = -oo
        else:
            point_val = sympify(str(point))

        log += f"Computing: lim[{variable}→{point}] {expr}\n\n"

        # Compute limit
        result = limit(expr, var, point_val, dir=direction)

        log += "## Result:\n"
        log += f"```\nlim[{variable}→{point}] {expr} = {result}\n```\n\n"

        # LaTeX
        log += f"## LaTeX:\n```latex\n{latex(result)}\n```\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def symbolic_matrix_operations(matrix_str, operation='det'):
    """
    Perform symbolic matrix operations.

    Parameters:
    -----------
    matrix_str : str
        Matrix as string (e.g., "[[a, b], [c, d]]")
    operation : str
        Operation: 'det' (determinant), 'inverse', 'eigenvals', 'eigenvects'

    Returns:
    --------
    str
        Matrix operation result
    """
    try:
        from sympy import Matrix, symbols, latex
    except ImportError:
        return "Error: sympy library not available"

    log = f"# Symbolic Matrix Operation: {operation}\n\n"

    log += f"## Matrix: {matrix_str}\n\n"

    try:
        # Parse matrix
        M = Matrix(eval(matrix_str))

        log += f"Parsed matrix ({M.shape[0]}×{M.shape[1]}):\n```\n{M}\n```\n\n"

        if operation == 'det':
            result = M.det()
            log += f"## Determinant:\n```\ndet(M) = {result}\n```\n\n"

        elif operation == 'inverse':
            result = M.inv()
            log += f"## Inverse Matrix:\n```\n{result}\n```\n\n"

        elif operation == 'eigenvals':
            eigenvals = M.eigenvals()
            log += f"## Eigenvalues:\n"
            for eigenval, multiplicity in eigenvals.items():
                log += f"- λ = {eigenval} (multiplicity: {multiplicity})\n"

        elif operation == 'eigenvects':
            eigenvects = M.eigenvects()
            log += f"## Eigenvectors:\n"
            for eigenval, multiplicity, vects in eigenvects:
                log += f"\n### Eigenvalue λ = {eigenval}:\n"
                for i, vect in enumerate(vects, 1):
                    log += f"Eigenvector {i}: {vect}\n"

        else:
            return f"Error: Unknown operation '{operation}'"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def simplify_expression(expression_str):
    """
    Simplify a symbolic expression.

    Parameters:
    -----------
    expression_str : str
        Expression to simplify

    Returns:
    --------
    str
        Simplified expression
    """
    try:
        from sympy import sympify, simplify, latex
    except ImportError:
        return "Error: sympy library not available"

    log = "# Expression Simplification\n\n"

    log += f"## Original: {expression_str}\n\n"

    try:
        expr = sympify(expression_str)
        log += f"Parsed: {expr}\n\n"

        # Simplify
        simplified = simplify(expr)

        log += f"## Simplified:\n```\n{simplified}\n```\n\n"

        log += f"## LaTeX:\n```latex\n{latex(simplified)}\n```\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def solve_system_of_equations(equations_list, variables_list):
    """
    Solve a system of symbolic equations.

    Parameters:
    -----------
    equations_list : list of str
        List of equations
    variables_list : list of str
        List of variables to solve for

    Returns:
    --------
    str
        Solutions
    """
    try:
        from sympy import symbols, sympify, solve
    except ImportError:
        return "Error: sympy library not available"

    log = "# System of Equations\n\n"

    log += f"## Equations:\n"
    for eq in equations_list:
        log += f"- {eq}\n"
    log += f"\n## Variables: {', '.join(variables_list)}\n\n"

    try:
        # Parse variables
        vars_symbols = symbols(' '.join(variables_list))
        if not isinstance(vars_symbols, tuple):
            vars_symbols = (vars_symbols,)

        # Parse equations
        equations = []
        for eq_str in equations_list:
            if '=' in eq_str:
                lhs, rhs = eq_str.split('=')
                equations.append(sympify(lhs) - sympify(rhs))
            else:
                equations.append(sympify(eq_str))

        # Solve
        solutions = solve(equations, vars_symbols, dict=True)

        log += f"## Solutions ({len(solutions)} found):\n\n"

        if len(solutions) == 0:
            log += "No solutions found.\n"
        else:
            for i, sol in enumerate(solutions, 1):
                log += f"### Solution {i}:\n"
                for var, val in sol.items():
                    log += f"- {var} = {val}\n"
                    try:
                        numerical = complex(val.evalf())
                        if abs(numerical.imag) < 1e-10:
                            log += f"  ≈ {numerical.real:.6f}\n"
                    except:
                        pass
                log += "\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log
