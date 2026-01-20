# Data lake dictionary with mathematical modeling datasets
data_lake_dict = {
    # Benchmark Optimization Problems
    "optimization_test_functions.csv": "Classical optimization test functions (Rosenbrock, Rastrigin, Ackley, etc.) with global minima.",
    "constrained_optimization_benchmarks.parquet": "Constrained optimization benchmark problems with known solutions.",
    "multi_objective_test_problems.parquet": "Multi-objective optimization test problems and Pareto fronts.",
    # Differential Equations and Dynamical Systems
    "ode_benchmark_problems.json": "Standard ODE test problems from the literature (Van der Pol, Lorenz, Rössler, etc.).",
    "stiff_ode_problems.json": "Stiff ordinary differential equation test cases.",
    "pde_benchmark_solutions.parquet": "Benchmark solutions for heat equation, wave equation, and Laplace equation.",
    "dynamical_systems_examples.json": "Examples of dynamical systems with known behavior (fixed points, limit cycles, chaos).",
    # Linear Algebra Datasets
    "sparse_matrix_collection.parquet": "Suite Sparse Matrix Collection - real-world sparse matrices from various applications.",
    "symmetric_matrices_test.parquet": "Collection of symmetric and positive definite test matrices.",
    "ill_conditioned_matrices.parquet": "Test matrices with various condition numbers for numerical stability testing.",
    "eigenvalue_problems.parquet": "Standard eigenvalue test problems with known eigenvalues/eigenvectors.",
    # Statistical and Time Series Data
    "time_series_benchmarks.parquet": "Standard time series datasets (airline passengers, sunspots, economic indicators).",
    "regression_datasets.parquet": "Classic regression datasets (Boston housing, diabetes, California housing).",
    "classification_datasets.parquet": "Standard classification datasets (iris, wine, breast cancer).",
    "statistical_distributions.json": "Parameters for standard probability distributions.",
    # Stochastic Process Data
    "markov_chain_examples.json": "Example Markov chain transition matrices from various applications.",
    "random_walk_datasets.parquet": "Historical random walk data (stock prices, Brownian motion).",
    "queuing_system_data.json": "Queueing theory examples and arrival/service time distributions.",
    "monte_carlo_benchmarks.json": "Monte Carlo integration test problems with known solutions.",
    # Network and Graph Data
    "graph_theory_datasets.parquet": "Standard graphs (Petersen, K-partite, Erdős–Rényi) for network analysis.",
    "network_flow_problems.json": "Network optimization problems (max flow, min cost flow).",
    "social_network_datasets.parquet": "Real-world social network structures (Zachary's karate club, etc.).",
    "complex_networks.parquet": "Scale-free and small-world network examples.",
    # Numerical Methods Test Cases
    "integration_test_integrals.json": "Definite integrals with known analytical solutions for numerical integration testing.",
    "root_finding_test_functions.json": "Functions with known roots for testing root-finding algorithms.",
    "interpolation_test_data.parquet": "Test datasets for interpolation and approximation methods.",
    "numerical_differentiation_tests.json": "Functions with known derivatives for testing numerical differentiation.",
    # Mathematical Models
    "population_dynamics_models.json": "Predator-prey, SIR, SEIR, logistic growth model parameters.",
    "physics_simulation_parameters.json": "Parameters for pendulum, spring-mass, planetary motion simulations.",
    "chemical_kinetics_models.json": "Chemical reaction system parameters and rate constants.",
    "financial_models.json": "Black-Scholes, Vasicek, CIR model parameters and market data.",
    # Computational Geometry
    "convex_hull_test_data.parquet": "Point clouds for convex hull algorithms.",
    "voronoi_tesselation_data.parquet": "Point sets for Voronoi diagram generation.",
    "triangulation_datasets.parquet": "2D and 3D point sets for triangulation algorithms.",
    # Function Approximation
    "fourier_series_examples.parquet": "Signals and their Fourier decompositions.",
    "wavelet_test_signals.parquet": "Test signals for wavelet analysis.",
    "spline_fitting_data.parquet": "Datasets for spline interpolation and smoothing.",
    # Special Functions and Constants
    "special_functions_values.parquet": "Tabulated values of Bessel, Legendre, and other special functions.",
    "mathematical_constants.json": "High-precision values of π, e, φ, γ, and other constants.",
    # Machine Learning Test Data
    "ml_regression_benchmarks.parquet": "Standard ML regression datasets with ground truth.",
    "ml_classification_benchmarks.parquet": "Standard ML classification datasets.",
    "dimensionality_reduction_data.parquet": "High-dimensional datasets for PCA, t-SNE, UMAP testing.",
    # Fractals and Chaos
    "fractal_dimension_datasets.parquet": "Datasets with known fractal dimensions.",
    "chaotic_time_series.parquet": "Time series from chaotic systems (Lorenz, logistic map).",
    "strange_attractor_data.parquet": "Coordinates of strange attractors from various systems.",
}

# Updated library_content as a dictionary focused on mathematical modeling
library_content_dict = {
    # === PYTHON PACKAGES ===
    # Core Scientific Computing
    "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
    "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, statistics, ODEs, and more.",
    "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
    "sympy": "[Python Package] A Python library for symbolic mathematics including algebra, calculus, equation solving, and discrete math.",
    # Optimization
    "cvxpy": "[Python Package] A Python-embedded modeling language for convex optimization problems.",
    "pyomo": "[Python Package] A Python-based, open-source optimization modeling language for linear, nonlinear, and mixed-integer programming.",
    "scipy.optimize": "[Python Module] Optimization algorithms including minimization, root finding, curve fitting, and linear programming (part of SciPy).",
    "optuna": "[Python Package] An automatic hyperparameter optimization framework, particularly useful for ML model tuning.",
    "pygmo": "[Python Package] A scientific library for massively parallel optimization providing bio-inspired and evolutionary algorithms.",
    "nlopt": "[Python Package] A library for nonlinear optimization providing various local and global optimization algorithms.",
    "gekko": "[Python Package] A library for machine learning and optimization of mixed-integer and differential algebraic equations.",
    "deap": "[Python Package] A novel evolutionary computation framework for rapid prototyping and testing of ideas in genetic algorithms.",
    # Differential Equations and Dynamical Systems
    "scipy.integrate": "[Python Module] Numerical integration of ODEs and PDEs (odeint, solve_ivp, solve_bvp) - part of SciPy.",
    "diffeqpy": "[Python Package] A Python interface to DifferentialEquations.jl from Julia for high-performance ODE/PDE solving.",
    "assimulo": "[Python Package] A package for solving ordinary differential equations with various advanced solvers.",
    "pydy": "[Python Package] A tool kit for modeling and simulating multibody dynamics problems.",
    # Linear Algebra and Matrix Computations
    "scipy.linalg": "[Python Module] Linear algebra routines including decompositions, eigenvalue problems, and matrix functions.",
    "numpy.linalg": "[Python Module] Basic linear algebra operations (part of NumPy).",
    "sparse": "[Python Package] Sparse multidimensional arrays for Python matching NumPy interface.",
    "scipy.sparse": "[Python Module] Sparse matrix and sparse linear algebra routines.",
    # Numerical Methods
    "numdifftools": "[Python Package] Tools for numerical differentiation with automatic differentiation capabilities.",
    "findiff": "[Python Package] A Python package for finite difference numerical derivatives in any number of dimensions.",
    "mpmath": "[Python Package] A library for arbitrary-precision floating-point arithmetic.",
    # Statistics and Probability
    "statsmodels": "[Python Package] Statistical modeling including regression, time series analysis, and hypothesis testing.",
    "scipy.stats": "[Python Module] Probability distributions, statistical functions, and tests (part of SciPy).",
    "pingouin": "[Python Package] An open-source statistical package written in Python 3 and based mostly on Pandas and NumPy.",
    "lifelines": "[Python Package] A complete survival analysis library for Python.",
    # Stochastic Processes and Monte Carlo
    "pymc": "[Python Package] Probabilistic programming for Bayesian statistical modeling and probabilistic machine learning.",
    "emcee": "[Python Package] A Python implementation of the affine-invariant ensemble sampler for MCMC.",
    "pyDOE": "[Python Package] Design of experiments for Python (Latin hypercube, factorial designs).",
    "SALib": "[Python Package] Sensitivity Analysis Library in Python for running sensitivity analysis.",
    # Symbolic and Computer Algebra
    "sympy": "[Python Package] A Python library for symbolic mathematics.",
    "sage": "[Python Package] Open-source mathematics software system - alternative to Mathematica/Maple/Matlab.",
    # Visualization
    "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
    "seaborn": "[Python Package] Statistical data visualization library based on matplotlib.",
    "plotly": "[Python Package] An interactive, open-source plotting library supporting 40+ chart types.",
    "mayavi": "[Python Package] A 3D scientific data visualization and plotting library in Python.",
    "vispy": "[Python Package] A high-performance interactive 2D/3D data visualization library.",
    # Machine Learning (for mathematical modeling)
    "scikit-learn": "[Python Package] Machine learning library with classification, regression, clustering, and dimensionality reduction.",
    "tensorflow": "[Python Package] An end-to-end open source platform for machine learning.",
    "pytorch": "[Python Package] An open source machine learning framework for deep learning.",
    "jax": "[Python Package] Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU.",
    # Network Analysis
    "networkx": "[Python Package] A Python package for creating, manipulating, and studying complex networks and graphs.",
    "igraph": "[Python Package] A library for creating, manipulating, and studying the structure of complex networks.",
    "graph-tool": "[Python Package] An efficient Python module for manipulation and statistical analysis of graphs.",
    # Computational Geometry
    "shapely": "[Python Package] Manipulation and analysis of geometric objects in the Cartesian plane.",
    "scipy.spatial": "[Python Module] Spatial algorithms and data structures (Delaunay, Voronoi, KDTree, etc.).",
    "trimesh": "[Python Package] A pure Python library for loading and using triangular meshes.",
    # Signal Processing
    "scipy.signal": "[Python Module] Signal processing tools including filtering, spectral analysis, and wavelets.",
    "pywavelets": "[Python Package] Wavelet transform module for Python.",
    # Special Functions
    "scipy.special": "[Python Module] Special mathematical functions (Bessel, error, gamma, beta, etc.).",
    # Parallel Computing
    "dask": "[Python Package] Parallel computing library that scales Python workflows.",
    "joblib": "[Python Package] A set of tools to provide lightweight pipelining and parallel computing.",
    "numba": "[Python Package] A JIT compiler that translates Python and NumPy code into fast machine code.",
    # Utilities
    "tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
    "h5py": "[Python Package] A Python interface to the HDF5 binary data format for storing large numerical arrays.",
    "tables": "[Python Package] PyTables for managing hierarchical datasets designed to efficiently handle extremely large amounts of data.",
    # === R PACKAGES ===
    # Core R Statistical Packages
    "stats": "[R Package] Base R statistics functions for probability distributions, statistical tests, and modeling.",
    "MASS": "[R Package] Functions and datasets from the book Modern Applied Statistics with S.",
    # Optimization in R
    "optim": "[R Package] General-purpose optimization based on Nelder-Mead, quasi-Newton and conjugate-gradient algorithms.",
    "optimx": "[R Package] A unified interface to optimization algorithms in R.",
    "nloptr": "[R Package] R interface to NLopt library for nonlinear optimization.",
    "lpSolve": "[R Package] Interface to Lp_solve library for solving linear and integer programming problems.",
    # Differential Equations in R
    "deSolve": "[R Package] General solvers for ordinary differential equations (ODE), partial differential equations (PDE), and delay differential equations (DDE).",
    "odesolve": "[R Package] Solvers for ordinary differential equations.",
    # Time Series Analysis
    "forecast": "[R Package] Methods and tools for displaying and analyzing univariate time series forecasts.",
    "tseries": "[R Package] Time series analysis and computational finance.",
    # Data Manipulation and Visualization
    "ggplot2": "[R Package] A system for declaratively creating graphics based on The Grammar of Graphics.",
    "dplyr": "[R Package] A grammar of data manipulation for working with data frames.",
    "tidyr": "[R Package] Tools for creating tidy data.",
    # Matrix Computations
    "Matrix": "[R Package] Classes and methods for dense and sparse matrices.",
    "matrixcalc": "[R Package] Collection of functions for matrix calculations.",
    # === CLI TOOLS ===
    # Mathematical Software
    "octave": "[CLI Tool] GNU Octave - scientific programming language largely compatible with MATLAB. Use with subprocess.run(['octave', '--eval', 'command']).",
    "bc": "[CLI Tool] An arbitrary precision calculator language. Use with subprocess.run(['bc']).",
    "gnuplot": "[CLI Tool] A portable command-line driven graphing utility. Use with subprocess.run(['gnuplot', '-e', 'plot command']).",
    # Symbolic Mathematics
    "maxima": "[CLI Tool] A system for the manipulation of symbolic and numerical expressions. Use with subprocess.run(['maxima', '-q']).",
    # Computational Tools
    "julia": "[CLI Tool] A high-level, high-performance programming language for technical computing.",
}
