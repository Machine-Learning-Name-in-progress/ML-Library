import pytest
import numpy as np
from src.ml_library.metrics.regression_metrics import *

y_true = [3.0, -1.0, 6.0, 7.0]
y_pred = [2.0, 0.0, 6.0, 8.0]
mse = (1**2 + 1**2 + 0**2 + 1**2) / 4
rmse = np.sqrt(mse)
mae = (1 + 1 + 0 + 1) / 4
mape = 100 * (1/3 + 1/1 + 0 + 1/7) / 4
rss = 3.0
tss = (3.0 - 15/4)**2 + (-1.0 - 15/4)**2 + (6.0 - 15/4)**2 + (7.0 - 15/4)**2
r2 = 1 - rss / tss
k, n = 1, len(y_true)
r2_adj = 1 - (1-r2)*(n-1)/(n-k-1)

atol = 1e-6

def test_reg_metrics_exceptions():
    with pytest.raises(ValueError, match="lists must have the same size"):
        mean_absolute_error(y_true, [1, 2, 3])

    with pytest.raises(ValueError, match="sample size must be strictly greater"):
        r_squared_adjusted(y_true, y_pred, k=3)

    with pytest.raises(ValueError, match="must be positive"):
        r_squared_adjusted(y_true, y_pred, k=-1)

    with pytest.raises(TypeError, match="must be an integer"):
        r_squared_adjusted(y_true, y_pred, k=1.5)

    with pytest.raises(ValueError, match="R2 is not defined"):
        r_squared([1.0, 1.0, 1.0, 1.0], y_pred)

    with pytest.raises(ValueError, match="MAPE is not defined"):
        mean_absolute_percentage_error([0.0, 1.0], [1.0, 1.0])

    with pytest.raises(TypeError, match="must be array-like"):
        mean_absolute_error(42, 67)

    with pytest.raises(TypeError, match="must contain only numerical entries"):
        mean_absolute_error(["string", 67], [32, 21])

def test_reg_metrics_mse():
    assert mean_squared_error(y_true, y_pred) == pytest.approx(mse, abs=atol)

def test_reg_metrics_rmse():
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(rmse, abs=atol)

def test_reg_metrics_mae():
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(mae, abs=atol)

def test_reg_metrics_mape():
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(mape, abs=atol)

def test_reg_metrics_r2():
    assert r_squared(y_true, y_pred) == pytest.approx(r2, abs=atol)

def test_reg_metrics_r2_adj():
    assert r_squared_adjusted(y_true, y_pred, k) == pytest.approx(r2_adj, abs=atol)

def test_reg_metrics_rss_tss():
    assert residual_sum_of_squares(y_true, y_pred) == pytest.approx(rss, abs=atol)
    assert total_sum_of_squares(y_true) == pytest.approx(tss, abs=atol)
