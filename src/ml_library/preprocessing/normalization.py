import pandas as pd


def normalization(data: pd.DataFrame, columns: list[str] = ()) -> pd.DataFrame:
    """
    Apply a min-max normalization on all numerical features of the data, using the expression:
    z_ij = (x_ij - min-i[x_ij]) / (max-i[x_ij]) - min-i[x_ij])
    For i-th value of the j-th feature.
    If some column has variance = 0 (all values are the same), then all are set to 0

    Args:
      data (pd.DataFrame): Dataframe of the pre-normalized values
      columns (list[str]): list of columns where will be applied. If empty, will normalize
                           all numerical columns

    Returns:
      (pd.DataFrame) with normalized columns
    """

    df = data.copy()

    # If no column is given, consider all numerical columns
    if not columns:
        columns = list(df.select_dtypes(include="number"))
    else:
        # Validate columns (existing)
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

            # Validate if all given columns are numerical
            if df[[column]].select_dtypes(include="number").empty:
                raise TypeError(
                    f"{column} is not a numerical column and cannot be normalized"
                )

    for j in columns:
        ij_max = df[j].max()  # Compute max value
        ij_min = df[j].min()  # Compute min value

        # Treatment for ZeroDivision:
        # If ij_max == ij_min, the denominator would be zero
        if ij_max != ij_min:
            df[j] = (df[j] - ij_min) / (ij_max - ij_min)
        else:
            # If a column has variance = 0, set all values on 0 by convention
            print(f"{j} has no variance, so all values are set on 0")
            df[j] = 0

    return df
