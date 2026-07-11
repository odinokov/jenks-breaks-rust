//! Python bindings for the Fisher-Jenks natural-breaks core.
//!
//! Accepts 1-D float32 or float64 NumPy arrays (computed in f64), validates input,
//! and releases the GIL during the O(k*n) SMAWK optimization.

pub mod core;
pub mod smawk;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy a 1-D NumPy array into an owned `Vec<f64>` while validating the contract.
///
/// The copy is necessary before releasing the GIL: another Python thread can mutate a
/// NumPy array while the core is running.  Combining the copy and validation avoids a
/// second full pass over input data, which is particularly valuable for `k == 1` and
/// small class counts.
fn copy_and_validate<I>(
    values: I,
    n: usize,
    num_classes: usize,
    assume_sorted: bool,
) -> PyResult<Vec<f64>>
where
    I: Iterator<Item = f64>,
{
    if n == 0 {
        return Err(PyValueError::new_err("data must be non-empty"));
    }
    if num_classes < 1 || num_classes > n {
        return Err(PyValueError::new_err(format!(
            "num_classes must be in 1..={n}, got {num_classes}"
        )));
    }

    let mut out = Vec::with_capacity(n);
    if assume_sorted {
        for value in values {
            if !value.is_finite() {
                return Err(PyValueError::new_err(
                    "data contains NaN or infinite values",
                ));
            }
            out.push(value);
        }
    } else {
        let mut previous = 0.0;
        for (index, value) in values.enumerate() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(
                    "data contains NaN or infinite values",
                ));
            }
            if index != 0 && value < previous {
                return Err(PyValueError::new_err(
                    "data must be sorted ascending (or pass assume_sorted=True to skip the check)",
                ));
            }
            previous = value;
            out.push(value);
        }
    }
    Ok(out)
}

/// Extract a supported NumPy dtype, then copy and validate it in a single pass.
fn to_f64_vec(
    data: &Bound<'_, PyAny>,
    num_classes: usize,
    assume_sorted: bool,
) -> PyResult<Vec<f64>> {
    if let Ok(array) = data.extract::<PyReadonlyArray1<f64>>() {
        let view = array.as_array();
        return copy_and_validate(view.iter().copied(), view.len(), num_classes, assume_sorted);
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<f32>>() {
        let view = array.as_array();
        return copy_and_validate(
            view.iter().map(|&value| value as f64),
            view.len(),
            num_classes,
            assume_sorted,
        );
    }
    Err(PyValueError::new_err(
        "data must be a 1-D float32 or float64 numpy array",
    ))
}

fn indices_impl(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    num_classes: usize,
    assume_sorted: bool,
) -> PyResult<Vec<usize>> {
    let x = to_f64_vec(data, num_classes, assume_sorted)?;
    Ok(py.detach(|| core::jenks_indices(&x, num_classes)))
}

/// Zero-based end index of each of the first `num_classes - 1` classes.
#[pyfunction]
#[pyo3(signature = (data, num_classes, *, assume_sorted=false))]
fn jenks_break_indices(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    num_classes: usize,
    assume_sorted: bool,
) -> PyResult<Vec<usize>> {
    indices_impl(py, data, num_classes, assume_sorted)
}

/// jenkspy-compatible boundary values: `[min, ...upper bounds..., max]` (length `num_classes + 1`).
#[pyfunction]
#[pyo3(signature = (data, num_classes, *, assume_sorted=false))]
fn jenks_breaks(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    num_classes: usize,
    assume_sorted: bool,
) -> PyResult<Vec<f64>> {
    let x = to_f64_vec(data, num_classes, assume_sorted)?;
    let indices = py.detach(|| core::jenks_indices(&x, num_classes));
    let mut out = Vec::with_capacity(num_classes + 1);
    out.push(x[0]);
    out.extend(indices.iter().map(|&index| x[index]));
    out.push(x[x.len() - 1]);
    Ok(out)
}

/// Deprecated alias for [`jenks_break_indices`]; kept for backward compatibility.
#[pyfunction]
#[pyo3(signature = (data, num_classes))]
fn jenks_breaks_optimized(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    num_classes: usize,
) -> PyResult<Vec<usize>> {
    let category = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
    py.import("warnings")?.call_method1(
        "warn",
        (
            "jenks_breaks_optimized is deprecated; use jenks_break_indices",
            &category,
        ),
    )?;
    indices_impl(py, data, num_classes, false)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(jenks_break_indices, module)?)?;
    module.add_function(wrap_pyfunction!(jenks_breaks, module)?)?;
    module.add_function(wrap_pyfunction!(jenks_breaks_optimized, module)?)?;
    Ok(())
}
