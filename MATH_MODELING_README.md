# MathMind: Mathematical Modeling AI Agent

## Overview

MathMind (formerly Biomni) has been transformed into a **general-purpose mathematical modeling AI agent** that can autonomously solve complex mathematical problems across diverse domains.

## Capabilities

### Core Mathematical Domains

1. **Differential Equations & Dynamical Systems**
   - Solve ODEs, PDEs, and systems of differential equations
   - Phase portrait analysis and bifurcation diagrams
   - Lyapunov exponents and chaos analysis
   - Find equilibrium points and analyze stability
   - Poincaré sections for 3D systems

2. **Optimization**
   - Linear programming and convex optimization
   - Nonlinear and constrained optimization
   - Global optimization (differential evolution, particle swarm)
   - Multi-objective optimization with Pareto fronts
   - Gradient descent and adaptive methods

3. **Numerical Methods**
   - Numerical integration (adaptive quadrature, Monte Carlo)
   - Numerical differentiation (finite differences, Richardson extrapolation)
   - Root finding (Newton, bisection, Brent's method)
   - Interpolation (linear, cubic splines, polynomial)
   - Solving nonlinear systems

4. **Linear Algebra**
   - Matrix decompositions (SVD, QR, LU, Cholesky, Schur)
   - Eigenvalue problems and eigenvector computation
   - Solving linear systems (direct and iterative methods)
   - Matrix norms and condition numbers
   - Pseudoinverse and matrix exponentials

5. **Statistical Modeling**
   - Linear and nonlinear regression
   - Time series analysis (ARIMA, seasonality, stationarity tests)
   - Hypothesis testing (t-tests, ANOVA, Mann-Whitney)
   - Distribution fitting and goodness-of-fit tests
   - Bootstrap confidence intervals

6. **Stochastic Modeling**
   - Monte Carlo simulation for uncertainty quantification
   - Markov chain simulation and stationary distributions
   - Random walk processes (1D, 2D, 3D)
   - Gillespie algorithm for stochastic chemical reactions
   - Wiener processes and Poisson processes

7. **Symbolic Mathematics**
   - Symbolic differentiation and integration
   - Solving algebraic equations symbolically
   - Taylor/Laurent series expansions
   - Limits and asymptotic analysis
   - Matrix operations with symbolic elements

8. **Network Analysis**
   - Graph theory and network topology
   - Community detection and clustering
   - Shortest paths and network flows
   - Centrality measures
   - Random graph models

## Quick Start Example

```python
from biomni import A1

# Initialize the mathematical modeling agent
agent = A1(
    model_name="claude-sonnet-4-5-20250929",
    commercial_mode=False
)

# Example 1: Solve an ODE system (Lorenz attractor)
prompt = """
Solve the Lorenz system of differential equations:
- dx/dt = σ(y - x)
- dy/dt = x(ρ - z) - y
- dz/dt = xy - βz

Use parameters σ=10, ρ=28, β=8/3, initial conditions [1, 1, 1],
and integrate from t=0 to t=50. Analyze the chaotic behavior and
compute the largest Lyapunov exponent.
"""

result = agent.go(prompt)
print(result)

# Example 2: Optimization problem
prompt = """
Minimize the Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2
using both gradient-based and global optimization methods.
Compare the results and analyze convergence.
"""

result = agent.go(prompt)
print(result)

# Example 3: Statistical analysis
prompt = """
Given two samples from different populations, perform:
1. Descriptive statistics
2. Test for normality
3. Two-sample t-test
4. Bootstrap confidence intervals for the mean difference
Interpret the results at α=0.05 significance level.
"""

result = agent.go(prompt)
print(result)
```

## Available Tool Modules

The agent has access to the following mathematical modeling tool modules:

- **differential_equations.py**: ODE/PDE solvers, equilibrium analysis, stability
- **optimization.py**: Linear/nonlinear optimization, global search, multi-objective
- **statistical_modeling.py**: Regression, time series, hypothesis testing
- **numerical_methods.py**: Integration, differentiation, root finding, interpolation
- **linear_algebra.py**: Matrix operations, decompositions, eigenvalues
- **dynamical_systems.py**: Phase portraits, bifurcation, Lyapunov exponents
- **stochastic_modeling.py**: Monte Carlo, Markov chains, stochastic processes
- **symbolic_math.py**: Symbolic calculus, equation solving, series expansions
- **support_tools.py**: Python REPL, general utilities

## Mathematical Datasets

The agent has access to a comprehensive data lake of mathematical resources:

- Optimization test functions and benchmarks
- Standard ODE/PDE problems with known solutions
- Matrix collections for linear algebra testing
- Time series and statistical datasets
- Network and graph theory examples
- Numerical methods test cases
- Special function values and mathematical constants

## Key Features

1. **Autonomous Problem Solving**: Give the agent a mathematical problem and it will:
   - Select appropriate numerical methods
   - Write and execute code
   - Analyze results
   - Provide interpretations and visualizations

2. **Multi-Language Support**: Can execute Python, R, and command-line tools

3. **Retrieval-Augmented Planning**: Automatically selects relevant tools and datasets

4. **Comprehensive Reporting**: Generates detailed logs with:
   - Method descriptions
   - Numerical results
   - Statistical analysis
   - Convergence diagnostics
   - Interpretations

5. **Error Handling**: Robust error checking and numerical stability analysis

## Example Use Cases

### 1. Engineering Design
```python
prompt = """
Optimize the dimensions of a cylindrical pressure vessel to minimize
material cost while maintaining a minimum volume of 1000 L and
maximum pressure of 10 MPa. Material properties: yield strength 250 MPa,
density 7850 kg/m³, cost $5/kg.
"""
```

### 2. Physics Simulation
```python
prompt = """
Simulate a double pendulum with masses m1=1kg, m2=1kg, lengths l1=1m, l2=1m.
Initial angles: θ1=π/2, θ2=π/4. Compute the trajectory for 20 seconds
and analyze whether the motion is chaotic.
"""
```

### 3. Financial Modeling
```python
prompt = """
Implement the Black-Scholes model for European option pricing.
Stock price S0=$100, strike K=$105, volatility σ=0.2, risk-free rate r=0.05,
time to maturity T=1 year. Compute option price and Greeks (delta, gamma, vega).
"""
```

### 4. Data Analysis
```python
prompt = """
Analyze this time series data for trends and seasonality.
Fit an ARIMA model, perform residual diagnostics, and forecast
the next 12 periods with 95% confidence intervals.
"""
```

## Differences from Original Biomni

This mathematical modeling version replaces:

- ✗ Biomedical databases → ✓ Mathematical datasets
- ✗ Genomics tools → ✓ Numerical methods
- ✗ Protein analysis → ✓ Optimization algorithms
- ✗ Drug discovery → ✓ Dynamical systems analysis
- ✗ Clinical data → ✓ Statistical test problems

## Configuration

The agent supports all the same configuration options as Biomni:

```python
agent = A1(
    model_name="claude-sonnet-4-5-20250929",  # Or GPT-4, Gemini, etc.
    commercial_mode=False,
    timeout=600,  # Execution timeout in seconds
    tool_retrieval=True,  # Automatic tool selection
)
```

## Advanced Features

### Custom Tool Registration
```python
# Add your own mathematical functions
def my_custom_solver(problem_parameters):
    # Your implementation
    pass

agent.add_tool(my_custom_solver)
```

### Streaming Execution
```python
# Stream results in real-time
for chunk in agent.go_stream(prompt):
    print(chunk, end='', flush=True)
```

### Conversation History
```python
# Save detailed execution logs
agent.save_conversation_history("my_analysis.pdf")
```

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- SymPy (for symbolic math)
- NetworkX (for graph analysis)
- Scikit-learn (for ML-based modeling)
- See full requirements in `pyproject.toml`

## Citation

If you use MathMind in your research, please cite the original Biomni paper:

```bibtex
@article{biomni2025,
  title={Biomni: A General-Purpose Biomedical AI Agent},
  author={...},
  journal={bioRxiv},
  year={2025}
}
```

## License

Apache 2.0 License (same as original Biomni)

## Contributing

Contributions are welcome! Please see CONTRIBUTION.md for guidelines.

## Support

For questions and discussions, join our Slack community or open an issue on GitHub.
