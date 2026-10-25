import pandas as pd


def standardization(data: pd.DataFrame, columns: list[str] = ()) -> pd.DataFrame:
    """
    Apply a z-score standardization on all numerical features of the data, using the expression:
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

    if not columns:
        columns = list(df.select_dtypes(include="number"))
    else:
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

            if df[[column]].select_dtypes(include="number").empty:
                raise TypeError(
                    f"{column} is not a numerical column and cannot be standardized"
                )

    for j in columns:
        j_mean = df[j].mean()
        j_std = df[j].std()

        if j_std != 0:
            df[j] = (df[j] - j_mean) / j_std
        else:
            print(f"{j} column have variance = 0 -> all values are constants")
            print(f"{j} column is defined equals 0")
            df[j] = 0

    return df
