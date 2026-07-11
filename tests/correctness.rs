//! Differential and property tests for the SMAWK Jenks core against an
//! obviously-correct O(k·n²) brute-force DP.
#![allow(clippy::needless_range_loop)] // DP index arithmetic reads clearer with explicit indices

use jenks_breaks::core::{jenks_indices, partition_sse};
use jenks_breaks::smawk::row_minima_contiguous;

/// Brute-force reference: exact minimum-SSE partition via straightforward
/// O(k·n²) dynamic programming. Independent of the SMAWK implementation.
fn brute_indices(x: &[f64], k: usize) -> Vec<usize> {
    let n = x.len();
    // Midpoint-centered prefix sums: an accurate objective (raw sums suffer
    // catastrophic cancellation on large-offset data). Independence from the core
    // is at the algorithm level — this is an exhaustive O(k·n²) DP, not SMAWK.
    let c = if n == 0 { 0.0 } else { 0.5 * (x[0] + x[n - 1]) };
    let mut s = vec![0.0f64; n + 1];
    let mut s2 = vec![0.0f64; n + 1];
    for i in 1..=n {
        let v = x[i - 1] - c;
        s[i] = s[i - 1] + v;
        s2[i] = s2[i - 1] + v * v;
    }
    let cost = |a: usize, b: usize| -> f64 {
        let w = (b - a + 1) as f64;
        let sum = s[b] - s[a - 1];
        let sq = s2[b] - s2[a - 1];
        sq - sum * sum / w
    };

    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; n + 1]; k + 1];
    let mut arg = vec![vec![0usize; n + 1]; k + 1];
    for i in 1..=n {
        dp[1][i] = cost(1, i);
    }
    for j in 2..=k {
        for i in j..=n {
            for split in (j - 1)..=(i - 1) {
                let v = dp[j - 1][split] + cost(split + 1, i);
                if v < dp[j][i] {
                    dp[j][i] = v;
                    arg[j][i] = split;
                }
            }
        }
    }
    let mut breaks = vec![0usize; k.saturating_sub(1)];
    let mut i = n;
    for j in (2..=k).rev() {
        let split = arg[j][i];
        breaks[j - 2] = split - 1;
        i = split;
    }
    breaks
}

/// Assert `breaks` is a valid partition: strictly increasing, in range, length k-1.
fn assert_valid_partition(breaks: &[usize], n: usize, k: usize) {
    assert_eq!(breaks.len(), k - 1, "wrong number of breaks");
    let mut prev: isize = -1;
    for &b in breaks {
        assert!(b < n, "break {b} out of range for n={n}");
        assert!(
            (b as isize) > prev,
            "breaks not strictly increasing: {breaks:?}"
        );
        prev = b as isize;
    }
    // The final class [last+1 .. n-1] must be non-empty.
    if let Some(&last) = breaks.last() {
        assert!(
            last + 1 < n,
            "final class is empty: last break {last}, n={n}"
        );
    }
}

/// The optimum SSE is unique even when the argmin partition is not, so compare on SSE.
fn assert_same_optimum(x: &[f64], k: usize) {
    let got = jenks_indices(x, k);
    let want = brute_indices(x, k);
    assert_valid_partition(&got, x.len(), k);
    let sse_got = partition_sse(x, &got);
    let sse_want = partition_sse(x, &want);
    let tol = 1e-6 * (1.0 + sse_want.abs());
    assert!(
        (sse_got - sse_want).abs() <= tol,
        "SSE mismatch for k={k}: smawk={sse_got} brute={sse_want}\n  x={x:?}\n  got={got:?} want={want:?}"
    );
}

/// Deterministic LCG so tests are reproducible without external crates.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn exhaustive_small_n() {
    // For every small n and every k, several value patterns must match brute force.
    for n in 1..=16usize {
        for k in 1..=n {
            // pattern A: distinct ramp
            let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
            assert_same_optimum(&a, k);
            // pattern B: all equal (degenerate, cost 0)
            let b = vec![3.5f64; n];
            assert_same_optimum(&b, k);
            // pattern C: heavy ties
            let c: Vec<f64> = (0..n).map(|i| (i / 3) as f64).collect();
            assert_same_optimum(&c, k);
            // pattern D: two clusters
            let d: Vec<f64> = (0..n)
                .map(|i| if i < n / 2 { 0.0 } else { 100.0 })
                .collect();
            assert_same_optimum(&d, k);
        }
    }
}

#[test]
fn randomized_medium_n() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for _ in 0..400 {
        let n = 2 + (rng.next_u64() as usize % 60);
        let k = 1 + (rng.next_u64() as usize % n);
        // random values, then sort ascending
        let mut x: Vec<f64> = (0..n).map(|_| rng.unit() * 2000.0 - 1000.0).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_same_optimum(&x, k);
    }
}

#[test]
fn large_offset_precision() {
    // Values with a huge offset but small spread: the classic cancellation case.
    let mut rng = Lcg(0xdead_beef_cafe_babe);
    for _ in 0..50 {
        let n = 20 + (rng.next_u64() as usize % 40);
        let k = 1 + (rng.next_u64() as usize % n.min(8));
        let mut x: Vec<f64> = (0..n).map(|_| 1.0e9 + rng.unit()).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_same_optimum(&x, k);
    }
}

#[test]
fn edge_cases() {
    assert_eq!(jenks_indices(&[42.0], 1), Vec::<usize>::new());
    assert_eq!(jenks_indices(&[1.0, 2.0, 3.0], 3), vec![0, 1]);
    assert_eq!(jenks_indices(&[1.0, 1.0, 1.0, 1.0], 1), Vec::<usize>::new());
    // Two obvious clusters split at the gap.
    let x = [1.0, 1.1, 1.2, 9.0, 9.1, 9.2];
    assert_eq!(jenks_indices(&x, 2), vec![2]);
}

#[test]
fn very_large_offset_precision() {
    // Well below the ~1e154 scaling threshold, so both centered-prefix cost and the
    // two-pass reference are exact — pins correctness far above the 1e9 case.
    let mut rng = Lcg(0x0bad_c0de_1234_5678);
    for _ in 0..40 {
        let n = 10 + (rng.next_u64() as usize % 30);
        let k = 1 + (rng.next_u64() as usize % n.min(6));
        let mut x: Vec<f64> = (0..n).map(|_| 1.0e12 + rng.unit() * 1.0e3).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_same_optimum(&x, k);
    }
}

#[test]
fn partition_sse_hand_computed() {
    // Two segments [0,2] (mean 1, SSE 2) and [10,14] (mean 12, SSE 8) => 10.
    assert!((partition_sse(&[0.0, 2.0, 10.0, 14.0], &[1]) - 10.0).abs() < 1e-12);
    // One class over the whole array: mean 5, SSE = 1+1+... for [4,5,6] => 2.
    assert!((partition_sse(&[4.0, 5.0, 6.0], &[]) - 2.0).abs() < 1e-12);
    // Each point its own class => zero SSE.
    assert_eq!(partition_sse(&[1.0, 9.0, 100.0], &[0, 1]), 0.0);
}

/// SMAWK, tested directly against an independent brute-force row-minima on a matrix
/// that is totally monotone by construction: squared distance `(x[col] - y[row])²`
/// with increasing `x`, `y` (the classic Monge matrix). Exercises `row_minima_contiguous`
/// outside the Jenks oracle it is normally driven by.
#[test]
fn smawk_matches_bruteforce_row_minima() {
    let mut rng = Lcg(0xfeed_face_0000_1111);
    for _ in 0..300 {
        let rows = 1 + (rng.next_u64() as usize % 40);
        let cols = 1 + (rng.next_u64() as usize % 40);
        // Increasing sequences (sorted random values).
        let mut y: Vec<f64> = (0..rows).map(|_| rng.unit() * 100.0).collect();
        let mut x: Vec<f64> = (0..cols).map(|_| rng.unit() * 100.0).collect();
        y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let oracle = |i: usize, j: usize| {
            let d = x[j] - y[i];
            d * d
        };

        let mut argmins = vec![0usize; rows];
        let mut minima = vec![0.0f64; rows];
        row_minima_contiguous(0, rows, 0, cols, &oracle, &mut argmins, &mut minima);

        for i in 0..rows {
            // Independent brute-force minimum for this row.
            let mut best = f64::INFINITY;
            for j in 0..cols {
                best = best.min(oracle(i, j));
            }
            // Compare on value (argmin may tie), and that the reported argmin achieves it.
            assert!((minima[i] - best).abs() <= 1e-9 * (1.0 + best.abs()));
            assert!((oracle(i, argmins[i]) - best).abs() <= 1e-9 * (1.0 + best.abs()));
        }
    }
}
