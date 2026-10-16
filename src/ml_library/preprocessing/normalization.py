import pandas as pd

# TODO: error treatment


def normalization(data: pd.DataFrame, columns: list[str] = []) -> pd.DataFrame:
    """
    Apply a normalization on all numerical feature of the data, using the expression:
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

    if columns == []:
        columns = list(df.select_dtypes(include="number"))

    for j in columns:
        ij_max = df[j].max()
        ij_min = df[j].min()

        df[j] = (df[j] - ij_min) / (ij_max - ij_min)

    return df
