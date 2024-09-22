use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// This function calculates Jenks natural breaks optimization using dynamic programming.
#[pyfunction]
fn jenks_breaks_optimized(data: Vec<f64>, num_classes: usize) -> PyResult<Vec<usize>> {
    let n_data = data.len();

    // Error handling for invalid input
    if num_classes == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of classes must be a positive integer.",
        ));
    }
    if num_classes > n_data {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of classes cannot exceed number of data points.",
        ));
    }

    // Check if the data is sorted; if not, return an error
    if !data.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "The inpurt NumPy array must be sorted.",
        ));
    }

    // Precompute cumulative sums and cumulative sums of squares
    let mut cumulative_sum = vec![0.0; n_data + 1];
    let mut cumulative_sum_squares = vec![0.0; n_data + 1];

    for i in 1..=n_data {
        cumulative_sum[i] = cumulative_sum[i - 1] + data[i - 1];
        cumulative_sum_squares[i] = cumulative_sum_squares[i - 1] + data[i - 1] * data[i - 1];
    }

    // Initialize DP tables: lower_class_limits and variance_combinations
    let mut lower_class_limits = vec![0; (n_data + 1) * (num_classes + 1)];
    let mut variance_combinations = vec![f64::INFINITY; (n_data + 1) * (num_classes + 1)];

    let idx = |i, j| i * (num_classes + 1) + j;

    // Initialize for j=1 (first class)
    for i in 1..=n_data {
        lower_class_limits[idx(i, 1)] = 1;
        let sum = cumulative_sum[i] - cumulative_sum[0];
        let sum_squares = cumulative_sum_squares[i] - cumulative_sum_squares[0];
        let variance = sum_squares - (sum * sum) / (i as f64);
        variance_combinations[idx(i, 1)] = variance;
    }

    // Dynamic Programming: Fill DP tables for classes >1
    for j in 2..=num_classes {
        for i in j..=n_data {
            let mut min_variance = f64::INFINITY;
            let mut min_k = 0;

            // Instead of recalculating variance for every combination, reuse cumulative sums
            for k in (j - 1)..i {
                let sum = cumulative_sum[i] - cumulative_sum[k];
                let sum_squares = cumulative_sum_squares[i] - cumulative_sum_squares[k];
                let w = (i - k) as f64;
                let variance = sum_squares - (sum * sum) / w;

                let total_variance = variance_combinations[idx(k, j - 1)] + variance;

                if total_variance < min_variance {
                    min_variance = total_variance;
                    min_k = k + 1;
                }
            }

            lower_class_limits[idx(i, j)] = min_k;
            variance_combinations[idx(i, j)] = min_variance;
        }
    }

    // Backtracking to find the class break indices
    let mut break_indices = vec![0; num_classes - 1];
    let mut k = n_data;
    for j in (2..=num_classes).rev() {
        let lower = lower_class_limits[idx(k, j)] - 1;
        break_indices[j - 2] = lower;
        k = lower;
    }

    Ok(break_indices)
}

/// Define the Python module
#[pymodule]
fn jenks_breaks(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add the wrapped function to the Python module
    m.add_function(wrap_pyfunction!(jenks_breaks_optimized, m)?)?;
    Ok(())
}
