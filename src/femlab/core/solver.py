"""Linear solver wrapper for the global FEM system.

Wraps scipy.sparse.linalg.spsolve with basic logging and
failure detection.
"""

import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def solve_linear(
    K: sparse.csr_matrix,
    f: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Solve K u = f using a sparse direct solver.

    Args:
        K: (n, n) sparse stiffness matrix (CSR).
        f: (n,) force vector.
        verbose: If True, print solver diagnostics.

    Returns:
        u: (n,) displacement solution vector.

    Raises:
        RuntimeError: If the solver detects a singular or ill-conditioned system.
    """
    n = K.shape[0]
    if verbose:
        nnz = K.nnz
        print(f"[solver] System size: {n} DOFs, {nnz} non-zeros")

    t0 = time.perf_counter()
    try:
        u = spla.spsolve(K, f)
    except Exception as exc:
        raise RuntimeError(f"Sparse solve failed: {exc}") from exc
    elapsed = time.perf_counter() - t0

    if verbose:
        residual = np.linalg.norm(K @ u - f)
        print(f"[solver] Solve time: {elapsed:.4f} s")
        print(f"[solver] Residual ||Ku - f||: {residual:.6e}")

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Solution contains NaN/Inf — system is likely singular.")

    return u
