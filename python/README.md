# ICM Python Bindings

Python bindings for the ICM (Independent Chip Model) equity computation library.
Uses ctypes to call the high-performance C implementation directly.

## Prerequisites

- Python 3.8+
- NumPy
- FFTW3 (`brew install fftw` on macOS, `libfftw3-dev` on Linux)
- The ICM shared library (built automatically or manually)

## Quick Start

### Option 1: Build the shared library, then use the package

From the project root:

```bash
# Build the shared library
make libicm

# Use the Python package (no install needed)
cd python
python -c "
import icm
icm.init()
eq = icm.equity([1000, 2000, 3000], [0.5, 0.3, 0.2])
print('Equities:', eq)
print('Sum:', eq.sum())
"
```

### Option 2: Install with pip

```bash
cd python
pip install .           # standard install
pip install -e .        # editable install (for development)
pip install -e .[test]  # with test dependencies
```

The `pip install` step builds the shared library automatically (requires a C
compiler and FFTW3; see Prerequisites).

## Usage

```python
import icm

# Initialize (loads FFTW wisdom, builds lookup tables)
icm.init()

# Or with a custom wisdom file:
# icm.init("/path/to/fftw_wisdom.dat")

# Compute equities for all players
stacks = [1000, 2000, 3000, 4000, 5000]
payouts = [0.50, 0.30, 0.20]
equities = icm.equity(stacks, payouts)
print(equities)  # Larger stacks get higher equity

# Compute equities for a subset of players (faster for large n)
equities = icm.equity_subset(stacks, payouts, targets=[0, 2, 4])
print(equities[0], equities[2], equities[4])

# Works with numpy arrays too
import numpy as np
stacks = np.array([1000.0, 2000.0, 3000.0])
payouts = np.array([0.5, 0.3, 0.2])
equities = icm.equity(stacks, payouts)
```

## API Reference

### `icm.init(wisdom_path=None)`

Initialize the library. Call once before any computation.

- `wisdom_path`: Path to `fftw_wisdom.dat`. If `None`, uses the default.

### `icm.equity(stacks, payouts, Q=256)`

Compute ICM equities for all players.

- `stacks`: List or numpy array of chip stacks (length n).
- `payouts`: List or numpy array of payout coefficients (length k, 1 <= k <= n).
- `Q`: Number of quadrature points (default 256).
- Returns: numpy array of equities (length n), summing to `sum(payouts)`.

### `icm.equity_subset(stacks, payouts, targets, Q=256)`

Compute equities for a subset of players.

- `targets`: List or numpy array of 0-based player indices to compute.
- Returns: numpy array (length n), only target indices filled.

## Running Tests

```bash
cd python
pip install -e .[test]
pytest tests/ -v
```

## Environment Variables

- `ICM_LIB_PATH`: Override the shared library path (e.g., `/custom/path/libicm.dylib`).
- `ICM_DEVICE`: Device profile for building (`m3_max`, `zen4`, etc.).
  Defaults to `m3_max` on macOS, `zen4` on Linux.
