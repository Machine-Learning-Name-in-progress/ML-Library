"""Tests for LOOCV splitter."""

import numpy as np
import pytest

from ml_library.model_selection.loocv import loocv_indices


def test_loocv_returns_n_splits() -> None:
    """Test if loocv_indices returns n splits."""
    splits = loocv_indices(n=4)
    assert len(splits) == 4


def test_loocv_invalid_parameters() -> None:
    """Test if n < 2 raises ValueError."""
    with pytest.raises(ValueError):
        loocv_indices(n=1)
    with pytest.raises(ValueError):
        loocv_indices(n=0)


def test_loocv_split_sizes_and_no_overlap() -> None:
    """Test if each split leaves one observation out without overlap."""
    n = 5
    splits = loocv_indices(n=n)

    for i, (train_idx, val_idx) in enumerate(splits):
        assert len(val_idx) == 1
        assert val_idx[0] == i
        assert len(train_idx) == n - 1
        assert i not in train_idx


def test_loocv_stability() -> None:
    """Test deterministic stability across multiple calls."""
    splits_first = loocv_indices(n=5)
    splits_second = loocv_indices(n=5)

    for (train_1, val_1), (train_2, val_2) in zip(splits_first, splits_second):
        np.testing.assert_array_equal(train_1, train_2)
        np.testing.assert_array_equal(val_1, val_2)
