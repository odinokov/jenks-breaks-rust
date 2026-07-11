//! Pure-Rust Fisher-Jenks core: exact 1-D minimum-SSE partitioning in O(k*n)
//! via SMAWK. No Python dependency, so it is unit-testable with `cargo test`.
//!
//! Caller contract (enforced by the Python wrapper): `x` is finite and sorted
//! ascending, and `1 <= k <= x.len()`.

use crate::smawk;

/// Segment sum-of-squared-error oracle backed by prefix sums over centered data.
///
/// SSE is invariant under translation and positive scaling.  We always center at a
/// midpoint computed without overflow.  For unusually wide inputs, we also scale only
/// when necessary to keep the prefix sum of squares finite.  This preserves the
/// argmin while avoiding `inf - inf` / `NaN` costs for otherwise valid finite input.
struct Sse {
    /// Prefix sums of normalized values, 1-based: `s[i] = sum_{t<i} y[t]`.
    s: Vec<f64>,
    /// Prefix sums of normalized squares, 1-based.
    s2: Vec<f64>,
}

impl Sse {
    fn new(x: &[f64]) -> Self {
        let n = x.len();
        debug_assert!(!x.is_empty());

        // `0.5 * a + 0.5 * b` remains finite for finite endpoints, unlike
        // `0.5 * (a + b)` when their sum overflows.
        let center = 0.5 * x[0] + 0.5 * x[n - 1];
        let half_span = (x[0] - center).abs().max((x[n - 1] - center).abs());

        // `s2` can grow to n * half_span^2.  Scale only when that would overflow;
        // keeping the usual path unscaled avoids an unnecessary multiply per input.
        let max_unscaled_span = f64::MAX.sqrt() / (n as f64).sqrt();
        let inv_scale = if half_span > max_unscaled_span {
            half_span.recip()
        } else {
            1.0
        };

        let mut s = Vec::with_capacity(n + 1);
        let mut s2 = Vec::with_capacity(n + 1);
        s.push(0.0);
        s2.push(0.0);
        let mut sum = 0.0;
        let mut squares = 0.0;
        for &value in x {
            let normalized = (value - center) * inv_scale;
            sum += normalized;
            squares += normalized * normalized;
            s.push(sum);
            s2.push(squares);
        }
        Self { s, s2 }
    }

    /// SSE of the 1-based inclusive segment `[start..=end]` (requires `start <= end`).
    #[inline(always)]
    fn cost(&self, start: usize, end: usize) -> f64 {
        debug_assert!(start >= 1 && start <= end && end < self.s.len());
        let width = (end - start + 1) as f64;
        let sum = self.s[end] - self.s[start - 1];
        let squares = self.s2[end] - self.s2[start - 1];

        // This is algebraically `squares - sum * sum / width`, but dividing first
        // avoids overflowing the intermediate `sum * sum` while retaining the
        // simple scalar operations most CPUs optimize best in this hot oracle.
        squares - sum * (sum / width)
    }
}

/// Compute the optimal Jenks breaks for sorted `x` split into `k` classes.
///
/// Returns the `k-1` zero-based indices marking the last element of each of the
/// first `k-1` classes (empty when `k == 1`).
pub fn jenks_indices(x: &[f64], k: usize) -> Vec<usize> {
    let n = x.len();
    debug_assert!(k >= 1 && k <= n);

    // These exact cases avoid all prefix/DP allocation.  In particular, k == n used
    // to request O(n^2) work and memory despite its only valid partition being known.
    if k == 1 {
        return Vec::new();
    }
    if k == n {
        return (0..n - 1).collect();
    }

    let sse = Sse::new(x);

    // The two-class objective has a single linear scan.  It avoids setting up SMAWK
    // and a backpointer layer for a result consisting of just one split.
    if k == 2 {
        let mut best_split = 1;
        let mut best_cost = sse.cost(1, 1) + sse.cost(2, n);
        for split in 2..n {
            let candidate = sse.cost(1, split) + sse.cost(split + 1, n);
            if candidate < best_cost {
                best_cost = candidate;
                best_split = split;
            }
        }
        return vec![best_split - 1];
    }

    // dp_prev[i] = best SSE partitioning the first i points into the previous
    // number of classes.  `dp_cur[j..]` is filled directly by SMAWK every layer.
    let mut dp_prev = vec![0.0; n + 1];
    for (index, slot) in dp_prev.iter_mut().enumerate().skip(1) {
        *slot = sse.cost(1, index);
    }
    let mut dp_cur = vec![0.0; n + 1];

    // Store only valid backpointers: at layer j, rows i in j..=n exist.  The former
    // rectangular `(k + 1) * (n + 1)` allocation included all invalid cells.
    let mut layer_offsets = vec![0usize; k + 1];
    let mut backpointer_len = 0usize;
    for (j, layer_offset) in layer_offsets.iter_mut().enumerate().skip(2) {
        *layer_offset = backpointer_len;
        backpointer_len = backpointer_len
            .checked_add(n - j + 1)
            .expect("Jenks backpointer allocation size overflowed usize");
    }
    let mut backpointers = vec![0usize; backpointer_len];

    for j in 2..=k {
        let row_count = n - j + 1;
        let layer_start = layer_offsets[j];
        let layer = &mut backpointers[layer_start..layer_start + row_count];

        // Totally-monotone oracle: ending class j at row i by splitting after
        // `split`.  The invalid staircase (split >= i) is +infinity; every row has
        // a valid split, so SMAWK never selects its invalid region.
        let oracle = |i: usize, split: usize| -> f64 {
            if split >= i {
                f64::INFINITY
            } else {
                dp_prev[split] + sse.cost(split + 1, i)
            }
        };

        // Rows and columns are contiguous ranges, so this version avoids creating
        // O(n) row/column vectors per DP layer.  It also writes the winning cost
        // directly into dp_cur, eliminating a second oracle call per row.
        smawk::row_minima_contiguous(j, n + 1, j - 1, n, &oracle, layer, &mut dp_cur[j..]);
        std::mem::swap(&mut dp_prev, &mut dp_cur);
    }

    // Backtrack: class j-1 ends at 1-based position `split`, i.e. zero-based
    // `split - 1`.
    let mut breaks = vec![0usize; k - 1];
    let mut end = n;
    for j in (2..=k).rev() {
        let split = backpointers[layer_offsets[j] + end - j];
        breaks[j - 2] = split - 1;
        end = split;
    }
    breaks
}

/// Total within-class SSE of a partition given its break indices, computed with a
/// numerically stable per-segment two-pass mean subtraction. Used by tests and as a
/// public utility for validating a partition's quality.
pub fn partition_sse(x: &[f64], breaks: &[usize]) -> f64 {
    #[inline]
    fn segment_sse(segment: &[f64]) -> f64 {
        if segment.is_empty() {
            return 0.0;
        }
        // Center first so the mean calculation itself does not overflow for finite
        // large-offset values.  Inputs are normally sorted, so the endpoints bound
        // the segment and the midpoint is a good center.
        let center = 0.5 * segment[0] + 0.5 * segment[segment.len() - 1];
        let mean = center
            + segment.iter().map(|&value| value - center).sum::<f64>() / segment.len() as f64;
        segment
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum()
    }

    let mut start = 0usize;
    let mut total = 0.0;
    for &break_index in breaks {
        let end = break_index + 1;
        if end > start {
            total += segment_sse(&x[start..end]);
        }
        start = end;
    }
    if start < x.len() {
        total += segment_sse(&x[start..]);
    }
    total
}
