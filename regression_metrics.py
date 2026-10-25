import numpy as np

def mean_squared_error(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)

    n = len(y_true)
    mean_sq_error = residual_sum_of_squares(y_true, y_pred) / n

    return mean_sq_error

def root_mean_squared_error(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)

    root_mean_sq_error = np.sqrt(mean_squared_error(y_true, y_pred))

    return root_mean_sq_error

def mean_absolute_error(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)
    
    n = len(y_true)
    abs_errors = np.abs(y_true - y_pred)
    mean_abs_error = sum(abs_errors) / n

    return mean_abs_error

def mean_absolute_percentage_error(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)

    n = len(y_true)
    errors = np.abs((y_true - y_pred) / y_true)
    mean_abs_per_error = 100 * (sum(errors)) / n
    
    return mean_abs_per_error

def r_squared(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)

    r_squared = 1 - residual_sum_of_squares(y_true, y_pred) / total_sum_of_squares(y_true)
    
    return r_squared

def r_squared_adjusted(y_true: list, y_pred: list) -> float:
    # TODO: implement adjusted R2.
    pass

def residual_sum_of_squares(y_true: list, y_pred: list) -> float:
    _metric_exceptions(y_true, y_pred)

    rss = sum((y_true - y_pred)**2)

    return rss

def total_sum_of_squares(y_true: list) -> float:
    _metric_exceptions(y_true)

    n = len(y_true)
    mean = np.sum(y_true) / n
    tss = sum((y_true - mean)**2)

    return tss

def _metric_exceptions(list1: list, list2: list = None) -> None:
    # TODO: add type validation and other exceptions.
    if len(list1) != len(list2):
        raise ValueError(f"Both lists must have the same size. Got sizes {len(list1)} and {len(list2)}.")