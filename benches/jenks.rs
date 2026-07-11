use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use jenks_breaks::core::jenks_indices;
use std::hint::black_box;

/// Deterministic pseudo-random sorted data, no external RNG crate.
fn sorted_data(n: usize) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut x: Vec<f64> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2000.0 - 1000.0
        })
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x
}

fn bench_jenks(c: &mut Criterion) {
    let mut group = c.benchmark_group("jenks_indices");
    for &n in &[100usize, 1_000, 10_000, 100_000, 1_000_000] {
        let data = sorted_data(n);
        for &k in &[5usize, 10] {
            group.bench_with_input(BenchmarkId::new(format!("k{k}"), n), &n, |b, _| {
                b.iter(|| jenks_indices(black_box(&data), black_box(k)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_jenks);
criterion_main!(benches);
