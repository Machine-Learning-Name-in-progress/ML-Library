"""Tests for K-fold cross-validation splitter."""

import numpy as np
import pytest

from ml_library.model_selection.kfold import kfold_indices


def test_kfold_returns_k_splits() -> None:
    """Test if kfold_indices returns k splits."""
    splits = kfold_indices(n=6, k=3)
    assert len(splits) == 3


def test_kfold_invalid_parameters() -> None:
    """Test if invalid k or n values raise ValueError."""
    with pytest.raises(ValueError):
        kfold_indices(n=5, k=1)
    with pytest.raises(ValueError):
        kfold_indices(n=4, k=5)


def test_kfold_no_overlap_and_full_coverage() -> None:
    """Test if train and validation sets are disjoint and cover all indices."""
    n, k = 67, 5
    splits = kfold_indices(n=n, k=k)
    all_val_indices = []

    for train_idx, val_idx in splits:
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(train_idx) + len(val_idx) == n
        all_val_indices.append(val_idx)

    concatenated_val = np.concatenate(all_val_indices)
    np.testing.assert_array_equal(np.sort(concatenated_val), np.arange(n))


def test_kfold_stability() -> None:
    """Test deterministic stability across multiple calls."""
    splits_first = kfold_indices(n=7, k=3)
    splits_second = kfold_indices(n=7, k=3)

    for (train_1, val_1), (train_2, val_2) in zip(splits_first, splits_second):
        np.testing.assert_array_equal(train_1, train_2)
        np.testing.assert_array_equal(val_1, val_2)
