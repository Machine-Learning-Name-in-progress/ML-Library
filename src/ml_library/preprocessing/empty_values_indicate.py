import pandas as pd

# TODO: error treatment


def empty_values_indicate(
    data: pd.DataFrame, columns: list[str] = (), count: int = 1
) -> pd.DataFrame:
    """
    Create a new binary (0 or 1) column representing if the row have atleast some
    quantity of empty values, returning 1 if the row have, and 0 if not.

    Args:
      data (pd.DataFrame): Dataframe that will be used
      columns (list[str]): columns that will be analyzed. If empty, will analyze
                           all columns
      count (int): minimum empty values to consider the binary column as 1

    Return:
      (pd.DataFrame): the original DataFrame with one more binary column.
    """

    # Validate count value
    if isinstance(count, int) and count > 0:
        pass
    else:
        raise ValueError("'count' need to be a interger bigger than 0")

    # Validate columns
    for column in columns:
        if not column in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    df = data.copy()

    if columns == ():
        columns = list(df.columns)

    df[f"{columns}_{count}-empty_indicator"] = df[columns].isnull().sum(axis=1) >= count

    return df
