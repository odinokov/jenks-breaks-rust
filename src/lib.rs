use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyReadonlyArray1};

#[inline(always)]
fn sse_cost(a: usize, b: usize, s: &[f64], s2: &[f64]) -> f64 {
    // segment is 1-based inclusive [a..b], assume a<=b
    let w = (b - a + 1) as f64;
    let sum = s[b] - s[a - 1];
    let sq  = s2[b] - s2[a - 1];
    sq - (sum * sum) / w
}

// Divide & Conquer DP for monotone opt:
// dp_j[i] = min_{k in [j-1..i-1]} dp_{j-1}[k] + cost(k+1, i)
// argmin_j[i] is the optimizer k for backtracking.
// We assume optimal k is monotone in i (true for SSE segmentation / Jenks).
fn compute_layer(
    j: usize,
    i_left: usize,
    i_right: usize,
    k_left: usize,
    k_right: usize,
    dp_prev: &[f64],
    dp_cur: &mut [f64],
    argmin_cur: &mut [usize],
    s: &[f64],
    s2: &[f64],
) {
    if i_left > i_right { return; }
    let mid = (i_left + i_right) / 2;

    let mut best_k = k_left;
    let mut best = f64::INFINITY;
    // k must satisfy j-1 <= k <= mid-1
    let lo = k_left.max(j - 1);
    let hi = k_right.min(mid - 1);

    for k in lo..=hi {
        let v = dp_prev[k] + sse_cost(k + 1, mid, s, s2);
        if v < best {
            best = v;
            best_k = k;
        }
    }
    dp_cur[mid] = best;
    argmin_cur[mid] = best_k;

    // Monotone ranges
    if i_left <= mid.saturating_sub(1) {
        compute_layer(j, i_left, mid - 1, k_left, best_k, dp_prev, dp_cur, argmin_cur, s, s2);
    }
    if mid + 1 <= i_right {
        compute_layer(j, mid + 1, i_right, best_k, k_right, dp_prev, dp_cur, argmin_cur, s, s2);
    }
}

#[pyfunction]
fn jenks_breaks_optimized<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,   // borrow NumPy array, no copy
    num_classes: usize
) -> PyResult<Vec<usize>> {
    let x = data.as_slice()?; // &[f64]
    let n = x.len();
    if num_classes == 0 || num_classes > n {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid num_classes."));
    }

    // Prefix sums (1-based)
    let mut s  = vec![0.0; n + 1];
    let mut s2 = vec![0.0; n + 1];
    for i in 1..=n {
        let v = x[i - 1];
        s[i]  = s[i - 1] + v;
        s2[i] = s2[i - 1] + v * v;
    }

    // DP buffers
    let mut dp_prev = vec![f64::INFINITY; n + 1];
    let mut dp_cur  = vec![f64::INFINITY; n + 1];

    // Backpointers: argmin[j][i] stores k that ends segment j-1 at i=k (1..n)
    let mut argmin = vec![0usize; (num_classes + 1) * (n + 1)];
    let idx = |j: usize, i: usize| j * (n + 1) + i;

    // j = 1: one class for first i points
    for i in 1..=n {
        dp_prev[i] = sse_cost(1, i, &s, &s2);
        argmin[idx(1, i)] = 0; // previous end at k=0
    }

    // Heavy compute without holding the GIL
    py.allow_threads(|| {
        for j in 2..=num_classes {
            // compute dp_cur[i] for i in [j..n] using D&C; valid k in [j-1..i-1]
            compute_layer(j, j, n, j - 1, n - 1, &dp_prev, &mut dp_cur, &mut argmin[idx(j, 0)..idx(j, 0) + (n + 1)], &s, &s2);
            std::mem::swap(&mut dp_prev, &mut dp_cur);
            dp_cur.fill(f64::INFINITY);
        }
    });

    // Backtrack breaks (return zero-based end indices of the earlier classes)
    let mut breaks = vec![0usize; num_classes - 1];
    let mut i = n;
    for j in (2..=num_classes).rev() {
        let k = argmin[idx(j, i)];   // k in 0..i-1 (1-based end at k)
        breaks[j - 2] = k.saturating_sub(1); // convert to zero-based end index
        i = k;
    }
    Ok(breaks)
}

#[pymodule]
fn jenks_breaks(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jenks_breaks_optimized, m)?)?;
    Ok(())
}
