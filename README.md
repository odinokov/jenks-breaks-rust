# jenks-breaks

[![CI](https://github.com/odinokov/jenks-breaks-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/odinokov/jenks-breaks-rust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Built with maturin](https://img.shields.io/badge/built%20with-maturin%20%2B%20PyO3-000000.svg)](https://www.maturin.rs/)

**Fast, exact [Fisher–Jenks natural breaks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization) for Python, implemented in Rust.**

Given sorted 1-D data, `jenks_breaks` partitions it into `k` classes that minimize the
total within-class sum of squared deviations — the optimal Jenks classification used in
cartography, choropleth mapping, and data binning — in **O(k·n)** time.

```python
import numpy as np
from jenks_breaks import jenks_breaks

data = np.sort(np.random.default_rng(0).uniform(0, 10, 100_000))
jenks_breaks(data, 5)   # -> [0.0, 2.01, 4.02, 6.00, 7.99, 10.0]  (class boundaries)
```

---

## Features

- **Asymptotically optimal.** SMAWK row minima give **O(k·n)** — no `O(n²)` scan, no
  `O(n·log n)` divide-and-conquer. One million points classify in ~0.3 s.
- **Numerically stable.** Costs use mean-centered prefix sums, so large-offset data
  (`1e9 + noise`) keeps full precision where the textbook `Σx² − (Σx)²/n` formula
  collapses to noise.
- **Exact & verified.** Checked against an independent brute-force DP (exhaustive for
  small `n`, randomized for large) and cross-validated against
  [`jenkspy`](https://github.com/mthh/jenkspy).
- **Drop-in & typed.** Accepts `float32`/`float64` arrays, releases the GIL during
  compute, ships `py.typed` stubs, and offers a `jenkspy`-compatible API.
- **Portable wheels.** A single `abi3` wheel per platform runs on CPython 3.9+.

## Installation

**From PyPI** (recommended):

```bash
pip install jenks_breaks
```

**Pre-built wheel from GitHub Releases** — grab the wheel for your platform from the
[latest release](https://github.com/odinokov/jenks-breaks-rust/releases/latest) and:

```bash
pip install jenks_breaks-<version>-cp39-abi3-<platform>.whl
```

**From source** (requires a [Rust toolchain](https://rustup.rs/)):

```bash
pip install git+https://github.com/odinokov/jenks-breaks-rust.git
```

## Usage

```python
import numpy as np
from jenks_breaks import jenks_breaks, jenks_break_indices

data = np.sort(np.random.default_rng(0).uniform(0, 10, 10_000))

# jenkspy-compatible: k + 1 class boundaries [min, ...upper bounds..., max]
boundaries = jenks_breaks(data, 5)

# Or the raw zero-based end index of each of the first k-1 classes
idx = jenks_break_indices(data, 5)          # len == k - 1
assert list(boundaries) == [data[0], *data[idx], data[-1]]
```

**Input contract.** `data` must be a 1-D, ascending-sorted `float32`/`float64` NumPy
array. Sorting and finiteness are validated (`ValueError` otherwise); pass
`assume_sorted=True` to skip the O(n) sortedness check on trusted input.

## API

| Function | Returns | Description |
|---|---|---|
| `jenks_breaks(data, num_classes, *, assume_sorted=False)` | `list[float]` | `num_classes + 1` class boundaries `[min, …, max]` (jenkspy-compatible). |
| `jenks_break_indices(data, num_classes, *, assume_sorted=False)` | `list[int]` | Zero-based end index of each of the first `num_classes − 1` classes. |
| `jenks_breaks_optimized(data, num_classes)` | `list[int]` | Deprecated alias for `jenks_break_indices`. |

> **Precision note.** Costs are computed from prefix sums over midpoint-centered data.
> This is exact for typical single-scale inputs; data spanning an *extreme* dynamic range
> within a single call may lose precision to floating-point cancellation — an inherent
> trade-off of any O(k·n) prefix-sum Jenks implementation.

## Benchmarks

`num_classes = 5`, random uniform data, best-of-many on one machine
(`python bench/benchmark.py`). Both libraries are **compiled native extensions** —
[`jenkspy`](https://github.com/mthh/jenkspy) is a Cython/C build — so this is *not* a
"compiled vs interpreted" comparison. The gap is **algorithmic**: jenks-breaks is
**O(k·n)** (SMAWK), while jenkspy is roughly **O(k·n²)** and becomes impractical past
~10⁴ points. jenkspy timings vary run-to-run; the Rust column is the stable side.

| n | jenks-breaks (Rust · O(k·n)) | jenkspy (Cython · ~O(k·n²)) | speedup |
|---|---:|---:|---:|
| 100 | 0.009 ms | 0.081 ms | ~9× |
| 1 000 | 0.097 ms | 2.23 ms | ~23× |
| 10 000 | 1.66 ms | ~243 ms | ~150× |
| 100 000 | 20.5 ms | (infeasible) | — |
| 1 000 000 | 0.31 s | (infeasible) | — |

> A naive **pure-Python** Fisher–Jenks DP (also O(k·n²)) ranges from ~750× slower at
> n=100 to ~27,000× at n=5,000 — but most of that is interpreter overhead. jenkspy is the
> fair baseline, where the gap is purely the SMAWK algorithm.

Beyond speed, jenks-breaks is also **more numerically robust**: jenkspy and the naive
Python DP compute segment cost with the raw `Σx² − (Σx)²/n` formula, which loses precision
on large-offset data, whereas jenks-breaks centers first and stays correct.

## How it works

The dynamic program `dp[j][i] = min_k dp[j-1][k] + cost(k+1, i)` has a totally monotone
cost matrix, because the SSE cost satisfies the concave quadrangle inequality. That lets
[SMAWK](https://en.wikipedia.org/wiki/SMAWK_algorithm) find every layer's row minima in
linear time, giving O(k·n) overall with an allocation-free hot path. The algorithm lives
in `src/core.rs` (pure Rust, unit-testable) and `src/smawk.rs`; `src/lib.rs` is the thin
PyO3 wrapper.

## Development

```bash
cargo test                                   # Rust unit + differential tests
cargo clippy --all-targets -- -D warnings
cargo fmt --all
cargo bench                                  # Criterion microbenchmarks
maturin develop --release && pytest tests_py # Python tests (no network; vendored fixture)
```

Releases are cut by pushing a `vX.Y.Z` tag: CI builds `abi3` wheels for Linux, macOS, and
Windows plus an sdist, and attaches them to the GitHub Release.

## License

[MIT](LICENSE)
