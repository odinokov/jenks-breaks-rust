from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64] | npt.NDArray[np.float32] | Any

def jenks_break_indices(
    data: FloatArray,
    num_classes: int,
    *,
    assume_sorted: bool = ...,
) -> list[int]:
    """Zero-based end index of each of the first ``num_classes - 1`` classes.

    ``data`` must be a 1-D, ascending-sorted float32/float64 NumPy array.
    Pass ``assume_sorted=True`` to skip the O(n) sortedness check.
    """
    ...

def jenks_breaks(
    data: FloatArray,
    num_classes: int,
    *,
    assume_sorted: bool = ...,
) -> list[float]:
    """jenkspy-compatible class boundaries ``[min, ...upper bounds..., max]``.

    Returns ``num_classes + 1`` values. ``data`` must be a 1-D, ascending-sorted
    float32/float64 NumPy array.
    """
    ...

def jenks_breaks_optimized(data: FloatArray, num_classes: int) -> list[int]:
    """Deprecated alias for :func:`jenks_break_indices`."""
    ...
