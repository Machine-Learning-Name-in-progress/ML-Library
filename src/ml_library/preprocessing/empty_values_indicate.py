import pandas as pd


def empty_values_indicate(
    data: pd.DataFrame,
    columns: list[str] = (),
    count: int = 1,
    indicator_name: str = "",
) -> pd.DataFrame:
    """
    Create a new binary (0 or 1) column representing if the row have atleast some
    quantity of NaN values, returning 1 if the row have, and 0 if not.

    Args:
      data (pd.DataFrame): Dataframe that will be used
      columns (list[str]): Columns that will be analyzed. If empty, will analyze
                           all columns
      count (int): minimum count of NaN values to consider the binary column as 1
      indicator_name (str): Optional name for indicator column

    Return:
      (pd.DataFrame): a copy of the original DataFrame with one additional binary column.
    """

    # Validate count value
    if not isinstance(count, int) or count <= 0:
        raise ValueError("'count' need to be a integer greater than 0")

    # Validate columns
    for column in columns:
        if column not in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    df = data.copy()

    if not columns:
        columns = list(df.columns)

    if indicator_name == "":
        indicator_name = f"{columns}-{count}"  # Standard name

    df[indicator_name] = df[columns].isnull().sum(axis=1) >= count
    df[indicator_name] = df[indicator_name].astype(int)

    return df
