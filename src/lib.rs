use pyo3::prelude::*;  // Import PyO3's prelude for working with Python.
use pyo3::wrap_pyfunction;  // Import a macro to expose Rust functions to Python.

/// This function calculates Jenks natural breaks optimization using dynamic programming.
#[pyfunction]  // Exposes the function to Python.
fn jenks_breaks_optimized(data: Vec<f64>, num_classes: usize) -> PyResult<Vec<usize>> {
    let n_data = data.len();  // Get the length of the input data.

    // Error handling for invalid input.
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

    // Check if the data is sorted; if not, return an error.
    if !data.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "The input NumPy array must be sorted.",
        ));
    }

    // Precompute cumulative sums and cumulative sums of squares for variance calculation.
    let mut cumulative_sum = vec![0.0; n_data + 1];  // Cumulative sum of data.
    let mut cumulative_sum_squares = vec![0.0; n_data + 1];  // Cumulative sum of squares of data.

    // Fill the cumulative sum and sum of squares.
    for i in 1..=n_data {
        cumulative_sum[i] = cumulative_sum[i - 1] + data[i - 1];
        cumulative_sum_squares[i] = cumulative_sum_squares[i - 1] + data[i - 1] * data[i - 1];
    }

    // Initialize tables for class limits and variance combinations.
    let mut lower_class_limits = vec![0; (n_data + 1) * (num_classes + 1)];
    let mut variance_combinations = vec![f64::INFINITY; (n_data + 1) * (num_classes + 1)];

    // Helper closure to calculate the 1D index for our 2D DP tables.
    let idx = |i, j| i * (num_classes + 1) + j;

    // Initialize the first class (j = 1) for all data points.
    for i in 1..=n_data {
        lower_class_limits[idx(i, 1)] = 1;  // Start of the first class.
        let sum = cumulative_sum[i] - cumulative_sum[0];
        let sum_squares = cumulative_sum_squares[i] - cumulative_sum_squares[0];
        let variance = sum_squares - (sum * sum) / (i as f64);  // Calculate variance.
        variance_combinations[idx(i, 1)] = variance;
    }

    // Dynamic Programming: Compute the best class breakpoints for classes > 1.
    for j in 2..=num_classes {
        for i in j..=n_data {
            let mut min_variance = f64::INFINITY;
            let mut min_k = 0;

            // Find the best breakpoint using cumulative sums.
            for k in (j - 1)..i {
                let sum = cumulative_sum[i] - cumulative_sum[k];
                let sum_squares = cumulative_sum_squares[i] - cumulative_sum_squares[k];
                let w = (i - k) as f64;
                let variance = sum_squares - (sum * sum) / w;

                let total_variance = variance_combinations[idx(k, j - 1)] + variance;

                // Update the minimum variance and the best breakpoint.
                if total_variance < min_variance {
                    min_variance = total_variance;
                    min_k = k + 1;
                }
            }

            // Store the best break point and its variance.
            lower_class_limits[idx(i, j)] = min_k;
            variance_combinations[idx(i, j)] = min_variance;
        }
    }

    // Backtrack to find the break indices for each class.
    let mut break_indices = vec![0; num_classes - 1];  // Holds the final breakpoints.
    let mut k = n_data;  // Start backtracking from the last data point.
    for j in (2..=num_classes).rev() {
        break_indices[j - 2] = lower_class_limits[idx(k, j)] - 1;
        k = break_indices[j - 2];
    }

    Ok(break_indices)  // Return the computed break indices.
}

/// Define the Python module.
#[pymodule]  // This macro tells PyO3 to treat this as the module initialization function.
fn jenks_breaks(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add the optimized Jenks Breaks function to the Python module.
    m.add_function(wrap_pyfunction!(jenks_breaks_optimized, m)?)?;
    Ok(())
}
