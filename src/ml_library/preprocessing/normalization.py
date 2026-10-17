import pandas as pd

# TODO: error treatment


def normalization(data: pd.DataFrame, columns: list[str] = ()) -> pd.DataFrame:
    """
    Apply a min-max normalization on all numerical feature of the data, using the expression:
    z_ij = (x_ij - min-i[x_ij]) / (max-i[x_ij]) - min-i[x_ij])
    For i-th value of the j-th feature

    Args:
      data (pd.DataFrame): Dataframe of the pre-normalized values
      columns (list[str]): list of columns where will be applied. If empty, will
                           standard all numerical columns

    Returns:
      (pd.DataFrame) with normalized columns
    """

    df = data.copy()

    if columns == ():
        columns = list(df.select_dtypes(include="number"))
    else:
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

            if df[[column]].select_dtypes(include="number").empty:
                raise TypeError(
                    f"{column} is not a numerical column and cannot be normalized"
                )

    for j in columns:
        ij_max = df[j].max()
        ij_min = df[j].min()

        # Treatment for ZeroDivision
        if ij_max != ij_min:
            df[j] = (df[j] - ij_min) / (ij_max - ij_min)
        else:
            print(f"Max and min of {j} are equal -> {j} have only constant values")
            print(f"{j} have no variance, so all values are set on 0")
            df[j] = 0

    return df
