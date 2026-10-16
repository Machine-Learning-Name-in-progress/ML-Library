import pandas as pd

# TODO: error treatment


def standardization(data: pd.DataFrame, columns: list[str] = []) -> pd.DataFrame:
    """
    Apply a standardization on all numerical features of the data, using the expression:
    z_ij = (x_ij - mean_j) / std_j
    For i-th value of the j-th feature

    Args:
      data (pd.DataFrame): Dataframe of the pre-standard values
      columns (list[str]): list of columns where will be applied. If empty, will
                           standard all numerical columns

    Returns:
      (pd.DataFrame) with standardized columns
    """

    df = data.copy()

    if columns == []:
        columns = list(df.select_dtypes(include="number"))

    for j in columns:
        j_mean = df[j].mean()
        j_std = df[j].std()

        df[j] = (df[j] - j_mean) / j_std

    return df
