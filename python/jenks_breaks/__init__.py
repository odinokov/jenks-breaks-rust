"""Fast exact Fisher-Jenks natural breaks via SMAWK (O(k*n))."""

from ._native import (
    jenks_break_indices,
    jenks_breaks,
    jenks_breaks_optimized,
)

__all__ = [
    "jenks_break_indices",
    "jenks_breaks",
    "jenks_breaks_optimized",
]

__version__ = "0.2.0"
