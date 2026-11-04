"""Tests for K-fold cross-validation splitter."""

from ml_library.model_selection.kfold import kfold_indices


def test_kfold_returns_k_splits():
    """Test if kfold_indices returns k splits."""
    splits = kfold_indices(n=6, k=3)
    assert len(splits) == 3
