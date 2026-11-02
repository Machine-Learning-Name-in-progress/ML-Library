import numpy as np

def mean_squared_error(y_true: list, y_pred: list) -> float:
    """
    Calculates the Mean Squared Error (MSE) between observed and predicted values.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.

    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)

    n = len(y_true)
    mean_sq_error = residual_sum_of_squares(y_true, y_pred) / n

    return mean_sq_error

def root_mean_squared_error(y_true: list, y_pred: list) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between observed and predicted values.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.

    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)

    root_mean_sq_error = np.sqrt(mean_squared_error(y_true, y_pred))

    return root_mean_sq_error

def mean_absolute_error(y_true: list, y_pred: list) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between observed and predicted values.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.
    
    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)
    
    n = len(y_true)
    abs_errors = np.abs(y_true - y_pred)
    mean_abs_error = sum(abs_errors) / n

    return mean_abs_error

def mean_absolute_percentage_error(y_true: list, y_pred: list) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between observed and predicted values.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.

    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)

    n = len(y_true)

    if len(y_true[y_true == 0]) != 0:
        raise ValueError("MAPE is not defined when y_true contains 0.")

    errors = np.abs((y_true - y_pred) / y_true)
    mean_abs_per_error = 100 * (sum(errors)) / n
    
    return mean_abs_per_error

def r_squared(y_true: list, y_pred: list) -> float:
    """
    Calculates the R-squared score for a list of observed and predicted values.

    Args: 
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.
    
    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)

    tss = total_sum_of_squares(y_true)
    rss = residual_sum_of_squares(y_true, y_pred)

    if tss == 0:
        raise ValueError("R2 is not defined when TSS = 0.")

    r_squared = 1 - rss / tss
    
    return r_squared

def r_squared_adjusted(y_true: list, y_pred: list, k: int) -> float:
    """
    Calculates the R-squared adjusted score for a list of observed and predicted values, and the number of features.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.
        k (int): The number of features used in the model.

    Returns:
        float
    """
    if not isinstance(k, int):
        raise TypeError("The number of features must be an integer.")

    if k <= 0:
        raise ValueError("The number of features must be positive.")

    y_true, y_pred = _metric_exceptions(y_true, y_pred)
    n = len(y_true)

    if n <= k + 1:
        raise ValueError(f"The sample size must be strictly greater than k + 1, where k is the number of features.")

    r2 = r_squared(y_true, y_pred)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    return r2_adj

def residual_sum_of_squares(y_true: list, y_pred: list) -> float:
    """
    Calculates the Residual Sum of Squares (RSS) for a list of observed and predicted values.

    Args:
        y_true (list): List containing the observed values.
        y_pred (list): List containing the predicted values.

    Returns:
        float
    """
    y_true, y_pred = _metric_exceptions(y_true, y_pred)

    rss = sum((y_true - y_pred)**2)

    return rss

def total_sum_of_squares(y_true: list) -> float:
    """
    Calculates the Total Sum of Squares (TSS) for a list of observed values.

    Args:
        y_true (list): List containing the observed values

    Returns:
        float
    """
    y_true = _metric_exceptions(y_true)

    n = len(y_true)
    mean = np.sum(y_true) / n
    tss = sum((y_true - mean)**2)

    return tss

def _metric_exceptions(list1: list, list2: list = None) -> np.ndarray:
    """
    Validates inputs and converts them to numpy arrays.

    Args:
        list1 (list): First input to be validated.
        list2 (list, optional): Second input to be validated. If not provided, default is None.
    
    Returns:
        np.ndarray
    """
    if not isinstance(list1, (list, np.ndarray, tuple)):
        raise TypeError(f"Inputs must be array-like (list, numpy array or tuple). Got {type(list1).__name__}.")

    try:
        list1 = np.asarray(list1, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Input must contain only numerical entries and be convertible to a numpy array.")

    if list2 is not None:
        if not isinstance(list2, (list, np.ndarray, tuple)):
            raise TypeError(f"Inputs must be array-like (list, numpy array or tuple). Got {type(list2).__name__}.")

        try:
            list2 = np.asarray(list2, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("Input must contain only numerical entries and be convertible to a numpy array.")
        
        if len(list1) != len(list2):
            raise ValueError(f"Both lists must have the same size. Got sizes {len(list1)} and {len(list2)}.")
        
        return list1, list2
    return list1