"""Python-side tests: no network. Fixture is vendored under tests_py/data/."""

import json
from pathlib import Path

import numpy as np
import pytest

from jenks_breaks import jenks_break_indices, jenks_breaks, jenks_breaks_optimized

DATA_DIR = Path(__file__).parent / "data"

# Expected 5-class boundaries for the vendored jenkspy fixture (from jenkspy).
EXPECTED_5 = [
    0.0028109620325267315,
    2.0935479691252112,
    4.205495140049607,
    6.178148351609707,
    8.09175917180255,
    9.997982932254672,
]


@pytest.fixture(scope="module")
def fixture_sorted():
    data = json.loads((DATA_DIR / "test.json").read_text())
    return np.sort(np.asarray(data, dtype=np.float64))


def test_boundaries_match_jenkspy_reference(fixture_sorted):
    got = jenks_breaks(fixture_sorted, 5)
    assert len(got) == 6
    np.testing.assert_allclose(got, EXPECTED_5, rtol=0, atol=1e-9)


def test_indices_consistent_with_values(fixture_sorted):
    idx = jenks_break_indices(fixture_sorted, 5)
    vals = jenks_breaks(fixture_sorted, 5)
    assert len(idx) == 4
    reconstructed = [fixture_sorted[0], *fixture_sorted[np.array(idx)], fixture_sorted[-1]]
    np.testing.assert_allclose(vals, reconstructed, rtol=0, atol=0)


def test_deprecated_alias_matches(fixture_sorted):
    assert jenks_breaks_optimized(fixture_sorted, 5) == jenks_break_indices(fixture_sorted, 5)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 12])
def test_matches_jenkspy_various_k(fixture_sorted, k):
    jenkspy = pytest.importorskip("jenkspy")
    ours = jenks_breaks(fixture_sorted, k)
    theirs = jenkspy.jenks_breaks(fixture_sorted, k)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)


def test_float32_input_supported(fixture_sorted):
    f32 = fixture_sorted.astype(np.float32)
    idx = jenks_break_indices(f32, 5)
    assert len(idx) == 4
    assert idx == sorted(idx)


def test_k_equals_n():
    x = np.arange(5, dtype=np.float64)
    assert jenks_break_indices(x, 5) == [0, 1, 2, 3]


def test_all_equal_values():
    x = np.full(10, 7.0)
    idx = jenks_break_indices(x, 3)
    assert len(idx) == 2
    assert idx == sorted(idx)


def test_errors():
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        jenks_break_indices(x, 0)
    with pytest.raises(ValueError):
        jenks_break_indices(x, 4)
    with pytest.raises(ValueError):
        jenks_break_indices(np.array([]), 1)
    with pytest.raises(ValueError):
        jenks_break_indices(np.array([1.0, np.nan, 3.0]), 2)
    with pytest.raises(ValueError):
        jenks_break_indices(np.array([3.0, 1.0, 2.0]), 2)  # unsorted


def test_assume_sorted_skips_check():
    # Unsorted input is accepted (garbage-in) when the check is skipped.
    x = np.array([3.0, 1.0, 2.0])
    jenks_break_indices(x, 2, assume_sorted=True)


def test_assume_sorted_matches_checked(fixture_sorted):
    # On genuinely-sorted input, skipping the check must not change the result.
    assert jenks_break_indices(fixture_sorted, 5, assume_sorted=True) == jenks_break_indices(
        fixture_sorted, 5
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int32),
        [1.0, 2.0, 3.0],  # Python list, not an ndarray
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2-D
    ],
)
def test_rejects_unsupported_inputs(bad):
    with pytest.raises((ValueError, TypeError)):
        jenks_break_indices(bad, 2)


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_jenks_breaks_properties(fixture_sorted, k):
    boundaries = jenks_breaks(fixture_sorted, k)
    assert len(boundaries) == k + 1
    assert boundaries[0] == fixture_sorted[0]
    assert boundaries[-1] == fixture_sorted[-1]
    assert all(boundaries[i] <= boundaries[i + 1] for i in range(len(boundaries) - 1))
    idx = jenks_break_indices(fixture_sorted, k)
    expected = [fixture_sorted[0], *(fixture_sorted[i] for i in idx), fixture_sorted[-1]]
    np.testing.assert_array_equal(boundaries, expected)


def test_non_contiguous_input(fixture_sorted):
    strided = fixture_sorted[::2]  # non-contiguous view, still ascending
    assert not strided.flags["C_CONTIGUOUS"]
    assert jenks_break_indices(strided, 4) == jenks_break_indices(
        np.ascontiguousarray(strided), 4
    )


def test_rejects_infinite():
    with pytest.raises(ValueError, match="NaN or infinite"):
        jenks_break_indices(np.array([1.0, np.inf, 3.0]), 2)
    with pytest.raises(ValueError, match="NaN or infinite"):
        jenks_break_indices(np.array([-np.inf, 1.0, 3.0]), 2)


def test_nan_rejected_even_when_assume_sorted():
    with pytest.raises(ValueError, match="NaN or infinite"):
        jenks_break_indices(np.array([1.0, np.nan, 3.0]), 2, assume_sorted=True)


def test_deprecated_alias_warns(fixture_sorted):
    with pytest.warns(DeprecationWarning):
        jenks_breaks_optimized(fixture_sorted, 5)
