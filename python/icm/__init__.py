"""
ICM (Independent Chip Model) equity computation for poker tournaments.

High-performance Python bindings using ctypes for the ICM C library.
Computes tournament placement equities using generating-function quadrature.

Example usage:
    import icm
    icm.init()
    equities = icm.equity(stacks=[1000, 2000, 3000], payouts=[0.5, 0.3, 0.2])
"""

import ctypes
import ctypes.util
import os
import sys
import platform

import numpy as np

__all__ = ["init", "equity", "equity_subset", "ICMError"]

# ── Exceptions ──────────────────────────────────────────────────


class ICMError(Exception):
    """Raised when the ICM library cannot be loaded or a computation fails."""
    pass


# ── Library loading ─────────────────────────────────────────────

_lib = None


def _find_library():
    """Locate the ICM shared library, searching common paths."""
    if platform.system() == "Darwin":
        libname = "libicm.dylib"
    else:
        libname = "libicm.so"

    # Search order:
    # 1. ICM_LIB_PATH environment variable (explicit override)
    # 2. Build directory relative to this package (../../../build/)
    # 3. Project root relative to this package (../../../)
    # 4. System library paths via ctypes.util.find_library

    env_path = os.environ.get("ICM_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Relative to this file: python/icm/__init__.py -> project root
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(pkg_dir, "..", ".."))

    candidates = [
        os.path.join(project_root, "build", libname),
        os.path.join(project_root, libname),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fall back to system search
    found = ctypes.util.find_library("icm")
    if found:
        return found

    return None


def _load_library():
    """Load the ICM shared library and set up function signatures."""
    global _lib

    path = _find_library()
    if path is None:
        raise ICMError(
            "Cannot find ICM shared library. Build it with 'make libicm' "
            "or set ICM_LIB_PATH to the full path of libicm.so/libicm.dylib."
        )

    try:
        _lib = ctypes.CDLL(path)
    except OSError as e:
        raise ICMError(f"Failed to load ICM library from {path}: {e}") from e

    # void icm_init(const char *wisdom_path)
    _lib.icm_init.argtypes = [ctypes.c_char_p]
    _lib.icm_init.restype = None

    # double icm_equity(int n, const double *S, int Q,
    #                   const double *payout, int k, double *equity)
    _lib.icm_equity.argtypes = [
        ctypes.c_int,                              # n
        ctypes.POINTER(ctypes.c_double),           # S
        ctypes.c_int,                              # Q
        ctypes.POINTER(ctypes.c_double),           # payout
        ctypes.c_int,                              # k
        ctypes.POINTER(ctypes.c_double),           # equity
    ]
    _lib.icm_equity.restype = ctypes.c_double

    # double icm_equity_subset(int n, const double *S, int Q,
    #                          const double *payout, int k, double *equity,
    #                          const int *targets, int n_targets)
    _lib.icm_equity_subset.argtypes = [
        ctypes.c_int,                              # n
        ctypes.POINTER(ctypes.c_double),           # S
        ctypes.c_int,                              # Q
        ctypes.POINTER(ctypes.c_double),           # payout
        ctypes.c_int,                              # k
        ctypes.POINTER(ctypes.c_double),           # equity
        ctypes.POINTER(ctypes.c_int),              # targets
        ctypes.c_int,                              # n_targets
    ]
    _lib.icm_equity_subset.restype = ctypes.c_double


def _ensure_loaded():
    """Ensure the library is loaded (lazy loading on first use)."""
    if _lib is None:
        _load_library()


# ── Helper: convert input to contiguous C-double array ──────────

def _to_double_array(data, name):
    """Convert a list or numpy array to a contiguous C-double numpy array."""
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _to_int_array(data, name):
    """Convert a list or numpy array to a contiguous C-int numpy array."""
    arr = np.ascontiguousarray(data, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _ptr(arr, ctype=ctypes.c_double):
    """Get a ctypes pointer from a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


# ── Public API ──────────────────────────────────────────────────

_initialized = False


def init(wisdom_path=None):
    """Initialize the ICM library.

    Call this once before any equity computation. Loads FFTW wisdom
    and builds internal lookup tables.

    Args:
        wisdom_path: Path to fftw_wisdom.dat file. If None, the library
            uses its default search ("fftw_wisdom.dat" in the current
            working directory).
    """
    global _initialized
    _ensure_loaded()

    if wisdom_path is not None:
        if isinstance(wisdom_path, str):
            wisdom_path = wisdom_path.encode("utf-8")
        _lib.icm_init(wisdom_path)
    else:
        _lib.icm_init(None)

    _initialized = True


def _auto_init():
    """Auto-initialize if init() hasn't been called yet."""
    global _initialized
    if not _initialized:
        init()


def equity(stacks, payouts, Q=256):
    """Compute ICM equities for all players.

    Args:
        stacks: Array-like of chip stacks (positive doubles), length n.
        payouts: Array-like of payout coefficients, length k (1 <= k <= n).
            Standard ICM payouts (e.g., [0.5, 0.3, 0.2] for a 50/30/20 split).
        Q: Number of quadrature points (default 256). Higher values give
            more precision at the cost of speed.

    Returns:
        numpy.ndarray: Equity values for each player (length n).
            Equities sum to sum(payouts).

    Example:
        >>> import icm
        >>> icm.init()
        >>> eq = icm.equity([1000, 2000, 3000], [0.5, 0.3, 0.2])
        >>> print(eq)  # Player with more chips gets higher equity
    """
    _auto_init()

    stacks_arr = _to_double_array(stacks, "stacks")
    payouts_arr = _to_double_array(payouts, "payouts")

    n = len(stacks_arr)
    k = len(payouts_arr)

    if k < 1 or k > n:
        raise ValueError(f"payouts length k={k} must satisfy 1 <= k <= n={n}")
    if Q < 1:
        raise ValueError(f"Q must be positive, got {Q}")

    equity_out = np.zeros(n, dtype=np.float64)

    elapsed_ns = _lib.icm_equity(
        ctypes.c_int(n),
        _ptr(stacks_arr),
        ctypes.c_int(Q),
        _ptr(payouts_arr),
        ctypes.c_int(k),
        _ptr(equity_out),
    )

    return equity_out


def equity_subset(stacks, payouts, targets, Q=256):
    """Compute ICM equities for a subset of players.

    Only computes equities for the specified target players, which can be
    faster than computing all equities when you only need a few.

    Args:
        stacks: Array-like of chip stacks (positive doubles), length n.
        payouts: Array-like of payout coefficients, length k.
        targets: Array-like of 0-based player indices to compute.
        Q: Number of quadrature points (default 256).

    Returns:
        numpy.ndarray: Equity values for each player (length n).
            Only the entries at target indices are filled; others are 0.

    Example:
        >>> import icm
        >>> icm.init()
        >>> eq = icm.equity_subset([1000, 2000, 3000], [0.5, 0.3, 0.2], targets=[0, 2])
        >>> print(eq[0], eq[2])  # Only players 0 and 2 computed
    """
    _auto_init()

    stacks_arr = _to_double_array(stacks, "stacks")
    payouts_arr = _to_double_array(payouts, "payouts")
    targets_arr = _to_int_array(targets, "targets")

    n = len(stacks_arr)
    k = len(payouts_arr)
    n_targets = len(targets_arr)

    if k < 1 or k > n:
        raise ValueError(f"payouts length k={k} must satisfy 1 <= k <= n={n}")
    if Q < 1:
        raise ValueError(f"Q must be positive, got {Q}")
    if n_targets < 1:
        raise ValueError("targets must not be empty")
    if np.any(targets_arr < 0) or np.any(targets_arr >= n):
        raise ValueError(f"target indices must be in [0, {n-1}]")

    equity_out = np.zeros(n, dtype=np.float64)

    elapsed_ns = _lib.icm_equity_subset(
        ctypes.c_int(n),
        _ptr(stacks_arr),
        ctypes.c_int(Q),
        _ptr(payouts_arr),
        ctypes.c_int(k),
        _ptr(equity_out),
        _ptr(targets_arr, ctypes.c_int),
        ctypes.c_int(n_targets),
    )

    return equity_out
