"""Benchmark jenks_breaks (Rust/SMAWK) against jenkspy.

Usage: python bench/benchmark.py
Requires: numpy, jenkspy, and an installed jenks_breaks (maturin develop --release).

jenkspy is ~O(k*n^2), so it is only run up to JENKSPY_MAX_N; the Rust SMAWK core
(O(k*n)) is benchmarked at every size to show how far it scales past jenkspy.
"""

import time

import numpy as np

from jenks_breaks import jenks_breaks

try:
    from jenkspy import JenksNaturalBreaks

    HAVE_JENKSPY = True
except ImportError:
    HAVE_JENKSPY = False

JENKSPY_MAX_N = 10_000  # above this jenkspy takes tens of seconds / huge memory


def timeit(fn, *, min_reps=5, min_time=0.5):
    """Return best-of seconds per call, auto-scaling repetitions."""
    fn()  # warm up
    reps, elapsed, best = 0, 0.0, float("inf")
    while reps < min_reps or elapsed < min_time:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        elapsed += dt
        reps += 1
    return best


def fmt_ms(seconds):
    return f"{seconds * 1e3:.3f} ms"


def main():
    rng = np.random.default_rng(42)
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    k = 5

    print(f"num_classes = {k}   (jenkspy capped at n = {JENKSPY_MAX_N})\n", flush=True)
    header = f"{'n':>10} | {'jenks_breaks (Rust)':>20} | {'jenkspy':>16} | {'speedup':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for n in sizes:
        data = np.sort(rng.uniform(-1000, 1000, n))

        best_rust = timeit(lambda d=data: jenks_breaks(d, k))

        if HAVE_JENKSPY and n <= JENKSPY_MAX_N:
            jnb = JenksNaturalBreaks(k)
            best_py = timeit(lambda d=data: jnb.fit(d), min_reps=3, min_time=0.3)
            py_str = fmt_ms(best_py)
            speedup = f"{best_py / best_rust:7.1f}x"
        else:
            py_str = "skipped" if HAVE_JENKSPY else "n/a"
            speedup = "-"

        print(f"{n:>10} | {fmt_ms(best_rust):>20} | {py_str:>16} | {speedup:>9}", flush=True)


if __name__ == "__main__":
    main()
