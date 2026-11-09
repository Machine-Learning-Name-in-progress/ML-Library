import warnings

import numpy as np
import pandas as pd


def standardization(
    data: pd.DataFrame, columns: list[str] = ()
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a z-score standardization on all numerical features of the data, using the expression:
    z_ij = (x_ij - mean_j) / std_j
    For i-th value of the j-th feature. std_j is the sample standard deviation
    (divides by n - 1).
    Constant or numerically constant columns are set to 0.0. A column is
    numerically constant when std_j <= n * eps * max_i |x_ij|, where eps is the
    float64 machine precision.

    Args:
      data (pd.DataFrame): Dataframe of the original pre-standard values
      columns (list[str]): list of columns where will be applied. If empty, will
                           standardize all numerical columns

    Returns:
      tuple[pd.DataFrame, dict]: Dataframe with standardized columns and a dict with
                                 mean and std used in each column
    """

    # Create a copy to prevent changes on original dataframe
    df = data.copy()

    # If no columns are given, consider all numerical
    if not columns:
        columns = list(df.select_dtypes(include="number"))

    else:
        # Validate columns (existing)
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

            # Validate columns (dtype)
            if df[[column]].select_dtypes(include="number").empty:
                raise TypeError(
                    f"{column} is not a numerical column and cannot be standardized"
                )

    # Machine precision of float64 (about 2.2e-16) and number of samples
    eps = np.finfo(np.float64).eps  # <- novo
    n = len(df)  # <- novo

    # Dict to save metrics for each column
    metrics = {}

    for j in columns:
        j_mean = df[j].mean()
        j_std = df[j].std()

        # Save metrics on dict
        metrics[j] = {"mean": j_mean, "std": j_std}

        # Prevent ZeroDivision Error
        # if max == min: column is exactly constant
        # if std <= n * eps * max|x|: the variation is below the rounding
        #                             error of the mean, so it is treated as constant
        scale = df[j].abs().max()
        is_constant = (df[j].max() == df[j].min()) or (j_std <= n * eps * scale)

        # Column varies more than the rounding error
        if not is_constant:
            df[j] = (df[j] - j_mean) / j_std
        else:
            warnings.warn(
                f"{j} column has (numerically) zero variance, so all values are set to 0"
            )
            df[j] = 0.0
            metrics[j]["std"] = 0.0

    return df, metrics
