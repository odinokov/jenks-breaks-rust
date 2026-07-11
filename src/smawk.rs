//! SMAWK row minima of a totally monotone matrix in O(rows + cols).
//!
//! The Jenks DP supplies consecutive row and column ranges, and the argmin column is
//! non-decreasing as the row grows (SSE cost is concave-Monge). Carrying rows as
//! `(start, stride, count)` rather than allocating a `Vec<usize>` at every recursive
//! level keeps the O(k·n) hot path allocation-free.
//!
//! Reference: Aggarwal, Klawe, Moran, Shor, Wilber, "Geometric applications of a
//! matrix-searching algorithm" (1987).

/// Find row minima for a totally monotone matrix over contiguous ranges.
///
/// `row_start..row_end` and `col_start..col_end` must be non-empty increasing ranges.
/// `argmins` and `minima` are indexed relative to `row_start` and must each have
/// length `row_end - row_start`. Both the winning column and its oracle value are
/// written, so the caller can use the dynamic-programming values directly.
pub fn row_minima_contiguous<F>(
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    f: &F,
    argmins: &mut [usize],
    minima: &mut [f64],
) where
    F: Fn(usize, usize) -> f64,
{
    debug_assert!(row_start < row_end);
    debug_assert!(col_start < col_end);
    debug_assert_eq!(argmins.len(), row_end - row_start);
    debug_assert_eq!(minima.len(), row_end - row_start);

    let columns = reduce(
        row_start,
        1,
        row_end - row_start,
        col_start..col_end,
        col_end - col_start,
        f,
    );
    let mut output = Output {
        row_start,
        argmins,
        minima,
    };
    solve_reduced(row_start, 1, row_end - row_start, &columns, f, &mut output);
}

/// Output buffers shared by every recursive level. Rows are indexed relative to the
/// top-level row range, so recursive strided subsets need no additional allocation.
struct Output<'a> {
    row_start: usize,
    argmins: &'a mut [usize],
    minima: &'a mut [f64],
}

/// Solve a matrix whose columns have already undergone the SMAWK REDUCE step for this
/// row set. Recursive calls reduce again for their smaller odd-row subset.
fn solve_reduced<F>(
    row_start: usize,
    row_stride: usize,
    row_count: usize,
    columns: &[usize],
    f: &F,
    output: &mut Output<'_>,
) where
    F: Fn(usize, usize) -> f64,
{
    debug_assert!(!columns.is_empty());

    // Recurse on odd-indexed rows (1, 3, 5, ...); their minima bound the scan for each
    // intervening even row.
    let odd_count = row_count / 2;
    if odd_count != 0 {
        let odd_start = row_start + row_stride;
        let odd_stride = row_stride * 2;
        let odd_columns = reduce(
            odd_start,
            odd_stride,
            odd_count,
            columns.iter().copied(),
            columns.len(),
            f,
        );
        solve_reduced(odd_start, odd_stride, odd_count, &odd_columns, f, output);
    }

    // INTERPOLATE even-indexed rows. The lower bound is the previous odd row's argmin
    // (carried in `column_index`); the upper bound is the next odd row's argmin, or the
    // last candidate for the final even row.
    let mut column_index = 0usize;
    let mut row_index = 0usize;
    while row_index < row_count {
        let row = row_start + row_index * row_stride;
        let upper_column = if row_index + 1 < row_count {
            let next_odd_row = row_start + (row_index + 1) * row_stride;
            output.argmins[next_odd_row - output.row_start]
        } else {
            *columns.last().unwrap()
        };

        let mut best_column = columns[column_index];
        let mut best_value = f(row, best_column);
        while columns[column_index] != upper_column {
            column_index += 1;
            let value = f(row, columns[column_index]);
            if value < best_value {
                best_value = value;
                best_column = columns[column_index];
            }
        }

        let output_index = row - output.row_start;
        output.argmins[output_index] = best_column;
        output.minima[output_index] = best_value;
        row_index += 2;
    }
}

/// SMAWK REDUCE: retain at most `row_count` candidate columns while preserving every
/// row minimum. Rows are represented arithmetically so no row vector is allocated to
/// index a contiguous or strided subset.
fn reduce<F, I>(
    row_start: usize,
    row_stride: usize,
    row_count: usize,
    columns: I,
    column_count: usize,
    f: &F,
) -> Vec<usize>
where
    F: Fn(usize, usize) -> f64,
    I: IntoIterator<Item = usize>,
{
    let mut stack = Vec::with_capacity(row_count.min(column_count));
    for column in columns {
        while let Some(&top) = stack.last() {
            let row = row_start + (stack.len() - 1) * row_stride;
            if f(row, top) < f(row, column) {
                break;
            }
            // On ties keep the later column; deterministic and matches the DP's
            // preference for the earliest valid split among equal-cost options.
            stack.pop();
        }
        if stack.len() < row_count {
            stack.push(column);
        }
    }
    stack
}
