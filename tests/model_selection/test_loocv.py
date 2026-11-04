"""Tests for LOOCV splitter."""

from ml_library.model_selection.loocv import loocv_indices


def test_loocv_returns_n_splits():
    """Test if loocv_indices returns n splits."""
    splits = loocv_indices(n=4)
    assert len(splits) == 4
