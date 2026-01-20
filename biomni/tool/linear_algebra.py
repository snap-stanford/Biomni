"""
Linear Algebra and Matrix Computations.

This module provides functions for linear algebra operations including
matrix decompositions, eigenvalue problems, linear systems, and matrix analysis.
"""

import numpy as np
from scipy import linalg


def solve_linear_system(A, b, method="direct"):
    """
    Solve a linear system Ax = b.

    Parameters:
    -----------
    A : array-like, shape (n, n)
        Coefficient matrix
    b : array-like, shape (n,) or (n, k)
        Right-hand side vector(s)
    method : str
        Solution method: 'direct' (LU), 'lstsq' (least squares), 'cholesky' (for SPD matrices)

    Returns:
    --------
    str
        Solution and system analysis
    """
    log = "# Linear System Solution\n\n"

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    m, n = A.shape
    log += "## System Properties:\n"
    log += f"- Matrix size: {m} × {n}\n"
    log += f"- System: {'Overdetermined' if m > n else 'Underdetermined' if m < n else 'Square'}\n"

    # Compute condition number
    if m == n:
        cond = np.linalg.cond(A)
        log += f"- Condition number: {cond:.2e}\n"
        if cond > 1e10:
            log += "  ⚠ Matrix is ill-conditioned\n"

    log += f"- Method: {method}\n\n"

    try:
        if method == "direct" and m == n:
            x = linalg.solve(A, b)
            log += "## Solution (Direct Method):\n"

        elif method == "cholesky":
            # For symmetric positive definite matrices
            x = linalg.solve(A, b, assume_a="pos")
            log += "## Solution (Cholesky Decomposition):\n"

        elif method == "lstsq":
            x, residuals, rank, s = linalg.lstsq(A, b)
            log += "## Solution (Least Squares):\n"
            log += f"- Matrix rank: {rank}\n"
            if len(residuals) > 0:
                log += f"- Sum of squared residuals: {residuals[0]:.2e}\n"

        else:
            return f"Error: Unknown method '{method}'"

        log += f"```\n{x}\n```\n\n"

        # Verify solution
        residual = A @ x - b
        residual_norm = np.linalg.norm(residual)

        log += "## Verification:\n"
        log += f"- Residual norm ||Ax - b||: {residual_norm:.2e}\n"

        if residual_norm < 1e-10:
            log += "✓ Solution verified (residual < 1e-10)\n"
        elif residual_norm < 1e-6:
            log += "✓ Solution accurate (residual < 1e-6)\n"
        else:
            log += "⚠ Solution may be inaccurate\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def eigenvalue_analysis(A, compute_vectors=True):
    """
    Compute eigenvalues and eigenvectors of a matrix.

    Parameters:
    -----------
    A : array-like, shape (n, n)
        Square matrix
    compute_vectors : bool
        Whether to compute eigenvectors

    Returns:
    --------
    str
        Eigenvalue analysis results
    """
    log = "# Eigenvalue Analysis\n\n"

    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        return "Error: Matrix must be square for eigenvalue analysis"

    log += "## Matrix Properties:\n"
    log += f"- Size: {n} × {n}\n"

    # Check if matrix is symmetric
    is_symmetric = np.allclose(A, A.T)
    log += f"- Symmetric: {is_symmetric}\n"

    # Check if matrix is positive definite (if symmetric)
    if is_symmetric:
        try:
            np.linalg.cholesky(A)
            log += "- Positive definite: Yes\n"
        except:
            log += "- Positive definite: No\n"

    log += "\n"

    try:
        if compute_vectors:
            if is_symmetric:
                # Use specialized routine for symmetric matrices
                eigenvalues, eigenvectors = linalg.eigh(A)
            else:
                eigenvalues, eigenvectors = linalg.eig(A)

            log += "## Eigenvalues:\n"
            for i, eig in enumerate(eigenvalues):
                if np.isreal(eig):
                    log += f"λ{i + 1} = {eig.real:.6f}\n"
                else:
                    log += f"λ{i + 1} = {eig.real:.6f} + {eig.imag:.6f}i\n"

            log += "\n## Eigenvalue Statistics:\n"
            log += f"- Largest eigenvalue: {np.max(np.abs(eigenvalues)):.6f}\n"
            log += f"- Smallest eigenvalue: {np.min(np.abs(eigenvalues)):.6f}\n"
            log += f"- Trace (sum of eigenvalues): {np.sum(eigenvalues):.6f}\n"
            log += f"- Determinant (product of eigenvalues): {np.prod(eigenvalues):.6f}\n"

            log += "\n## Eigenvectors:\n"
            log += f"Eigenvector matrix shape: {eigenvectors.shape}\n"

            # Show first eigenvector as example
            if n <= 10:
                log += f"\nFirst eigenvector (v1):\n```\n{eigenvectors[:, 0]}\n```\n"

            # Verify orthogonality for symmetric matrices
            if is_symmetric:
                ortho_error = np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(n))
                log += f"\nOrthogonality check ||V^T V - I||: {ortho_error:.2e}\n"

        else:
            eigenvalues = linalg.eigvals(A)
            log += "## Eigenvalues:\n"
            for i, eig in enumerate(eigenvalues):
                if np.isreal(eig):
                    log += f"λ{i + 1} = {eig.real:.6f}\n"
                else:
                    log += f"λ{i + 1} = {eig.real:.6f} + {eig.imag:.6f}i\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def matrix_decomposition(A, decomposition="svd"):
    """
    Perform matrix decomposition.

    Parameters:
    -----------
    A : array-like
        Matrix to decompose
    decomposition : str
        Type: 'svd' (Singular Value), 'qr' (QR), 'lu' (LU), 'cholesky', 'schur'

    Returns:
    --------
    str
        Decomposition results
    """
    log = f"# Matrix Decomposition: {decomposition.upper()}\n\n"

    A = np.array(A, dtype=float)
    m, n = A.shape

    log += "## Matrix Properties:\n"
    log += f"- Size: {m} × {n}\n"
    log += f"- Rank: {np.linalg.matrix_rank(A)}\n\n"

    try:
        if decomposition == "svd":
            # Singular Value Decomposition
            U, s, Vt = linalg.svd(A)

            log += "## Singular Value Decomposition: A = U Σ V^T\n\n"
            log += f"- U shape: {U.shape}\n"
            log += f"- Singular values: {s.shape[0]}\n"
            log += f"- V^T shape: {Vt.shape}\n\n"

            log += "## Singular Values:\n"
            for i, sv in enumerate(s):
                log += f"σ{i + 1} = {sv:.6f}\n"

            log += "\n## Condition Number:\n"
            log += f"- κ(A) = σ_max / σ_min = {s[0] / s[-1]:.2e}\n"

            # Reconstruction error
            A_reconstructed = U @ np.diag(s) @ Vt if m == n else U[:, : len(s)] @ np.diag(s) @ Vt
            recon_error = np.linalg.norm(A - A_reconstructed)
            log += "\n## Reconstruction Error:\n"
            log += f"- ||A - UΣV^T||: {recon_error:.2e}\n"

        elif decomposition == "qr":
            # QR Decomposition
            Q, R = linalg.qr(A)

            log += "## QR Decomposition: A = QR\n\n"
            log += f"- Q shape: {Q.shape} (orthogonal matrix)\n"
            log += f"- R shape: {R.shape} (upper triangular)\n\n"

            # Check orthogonality of Q
            ortho_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
            log += "## Q Orthogonality Check:\n"
            log += f"- ||Q^T Q - I||: {ortho_error:.2e}\n"

            # Reconstruction error
            recon_error = np.linalg.norm(A - Q @ R)
            log += "\n## Reconstruction Error:\n"
            log += f"- ||A - QR||: {recon_error:.2e}\n"

        elif decomposition == "lu":
            # LU Decomposition
            P, L, U = linalg.lu(A)

            log += "## LU Decomposition: PA = LU\n\n"
            log += f"- P shape: {P.shape} (permutation matrix)\n"
            log += f"- L shape: {L.shape} (lower triangular)\n"
            log += f"- U shape: {U.shape} (upper triangular)\n\n"

            # Reconstruction error
            recon_error = np.linalg.norm(P @ A - L @ U)
            log += "## Reconstruction Error:\n"
            log += f"- ||PA - LU||: {recon_error:.2e}\n"

        elif decomposition == "cholesky":
            # Cholesky Decomposition (for positive definite matrices)
            if m != n:
                return "Error: Cholesky decomposition requires square matrix"

            L = linalg.cholesky(A, lower=True)

            log += "## Cholesky Decomposition: A = LL^T\n\n"
            log += f"- L shape: {L.shape} (lower triangular)\n\n"

            # Reconstruction error
            recon_error = np.linalg.norm(A - L @ L.T)
            log += "## Reconstruction Error:\n"
            log += f"- ||A - LL^T||: {recon_error:.2e}\n"

        elif decomposition == "schur":
            # Schur Decomposition
            if m != n:
                return "Error: Schur decomposition requires square matrix"

            T, Z = linalg.schur(A)

            log += "## Schur Decomposition: A = ZTZ^T\n\n"
            log += f"- T shape: {T.shape} (upper triangular/quasi-triangular)\n"
            log += f"- Z shape: {Z.shape} (orthogonal matrix)\n\n"

            # Check orthogonality of Z
            ortho_error = np.linalg.norm(Z.T @ Z - np.eye(n))
            log += "## Z Orthogonality Check:\n"
            log += f"- ||Z^T Z - I||: {ortho_error:.2e}\n"

            # Reconstruction error
            recon_error = np.linalg.norm(A - Z @ T @ Z.T)
            log += "\n## Reconstruction Error:\n"
            log += f"- ||A - ZTZ^T||: {recon_error:.2e}\n"

        else:
            return f"Error: Unknown decomposition '{decomposition}'"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def matrix_norms(A):
    """
    Compute various matrix norms.

    Parameters:
    -----------
    A : array-like
        Matrix to analyze

    Returns:
    --------
    str
        Matrix norms and properties
    """
    log = "# Matrix Norms and Properties\n\n"

    A = np.array(A, dtype=float)
    m, n = A.shape

    log += f"## Matrix Dimensions: {m} × {n}\n\n"

    log += "## Matrix Norms:\n"
    log += f"- Frobenius norm (||A||_F): {np.linalg.norm(A, 'fro'):.6f}\n"
    log += f"- Spectral norm (||A||_2): {np.linalg.norm(A, 2):.6f}\n"
    log += f"- 1-norm (||A||_1): {np.linalg.norm(A, 1):.6f}\n"
    log += f"- ∞-norm (||A||_∞): {np.linalg.norm(A, np.inf):.6f}\n\n"

    log += "## Matrix Properties:\n"
    log += f"- Trace: {np.trace(A) if m == n else 'N/A (not square)'}\n"
    log += f"- Rank: {np.linalg.matrix_rank(A)}\n"

    if m == n:
        det = np.linalg.det(A)
        log += f"- Determinant: {det:.6f}\n"
        cond = np.linalg.cond(A)
        log += f"- Condition number: {cond:.2e}\n"

        if abs(det) > 1e-10:
            log += "- Matrix is invertible: Yes\n"
        else:
            log += "- Matrix is invertible: No (singular)\n"

    return log


def compute_pseudoinverse(A, method="svd", rcond=1e-15):
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.

    Parameters:
    -----------
    A : array-like
        Matrix to invert
    method : str
        Method: 'svd' or 'lstsq'
    rcond : float
        Cutoff for small singular values

    Returns:
    --------
    str
        Pseudoinverse computation results
    """
    log = "# Moore-Penrose Pseudoinverse\n\n"

    A = np.array(A, dtype=float)
    m, n = A.shape

    log += "## Input Matrix:\n"
    log += f"- Size: {m} × {n}\n"
    log += f"- Rank: {np.linalg.matrix_rank(A)}\n"
    log += f"- Method: {method}\n"
    log += f"- rcond: {rcond}\n\n"

    try:
        A_pinv = np.linalg.pinv(A, rcond=rcond)

        log += "## Pseudoinverse A⁺:\n"
        log += f"- Size: {A_pinv.shape[0]} × {A_pinv.shape[1]}\n\n"

        # Verify properties of pseudoinverse
        log += "## Verification of Pseudoinverse Properties:\n"

        # Property 1: A * A⁺ * A = A
        prop1_error = np.linalg.norm(A @ A_pinv @ A - A)
        log += f"1. AA⁺A = A: ||AA⁺A - A|| = {prop1_error:.2e}\n"

        # Property 2: A⁺ * A * A⁺ = A⁺
        prop2_error = np.linalg.norm(A_pinv @ A @ A_pinv - A_pinv)
        log += f"2. A⁺AA⁺ = A⁺: ||A⁺AA⁺ - A⁺|| = {prop2_error:.2e}\n"

        # Property 3: (A * A⁺)^T = A * A⁺
        AA_pinv = A @ A_pinv
        prop3_error = np.linalg.norm(AA_pinv.T - AA_pinv)
        log += f"3. (AA⁺)^T = AA⁺: ||(AA⁺)^T - AA⁺|| = {prop3_error:.2e}\n"

        # Property 4: (A⁺ * A)^T = A⁺ * A
        A_pinv_A = A_pinv @ A
        prop4_error = np.linalg.norm(A_pinv_A.T - A_pinv_A)
        log += f"4. (A⁺A)^T = A⁺A: ||(A⁺A)^T - A⁺A|| = {prop4_error:.2e}\n"

        log += (
            "\n✓ All properties satisfied (errors < 1e-10)\n"
            if max(prop1_error, prop2_error, prop3_error, prop4_error) < 1e-10
            else "\n"
        )

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log


def matrix_exponential(A, method="pade"):
    """
    Compute the matrix exponential exp(A).

    Parameters:
    -----------
    A : array-like, shape (n, n)
        Square matrix
    method : str
        Computation method: 'pade', 'eigen'

    Returns:
    --------
    str
        Matrix exponential results
    """
    log = "# Matrix Exponential\n\n"

    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        return "Error: Matrix must be square"

    log += "## Input Matrix A:\n"
    log += f"- Size: {n} × {n}\n"
    log += f"- Method: {method}\n\n"

    try:
        expA = linalg.expm(A)

        log += "## Matrix Exponential exp(A):\n"
        log += f"- Size: {expA.shape[0]} × {expA.shape[1]}\n\n"

        if n <= 5:
            log += f"exp(A) =\n```\n{expA}\n```\n\n"

        log += "## Properties:\n"
        log += f"- Norm ||exp(A)||: {np.linalg.norm(expA):.6f}\n"
        log += f"- Determinant det(exp(A)): {np.linalg.det(expA):.6f}\n"
        log += f"- Trace of exp(A): {np.trace(expA):.6f}\n"
        log += f"- exp(trace(A)): {np.exp(np.trace(A)):.6f}\n\n"

        # Verify: det(exp(A)) should equal exp(trace(A))
        det_expA = np.linalg.det(expA)
        exp_trA = np.exp(np.trace(A))
        log += "Verification: det(exp(A)) ≈ exp(tr(A))?\n"
        log += f"- Difference: {abs(det_expA - exp_trA):.2e}\n"

    except Exception as e:
        log += f"✗ Error: {str(e)}\n"

    return log
