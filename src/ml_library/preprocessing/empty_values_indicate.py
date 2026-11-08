import pandas as pd


def empty_values_indicate(
    data: pd.DataFrame,
    columns: list[str] = (),
    count: int = 1,
    indicator_name: str = "indicate",
) -> pd.DataFrame:
    """
    Create a new binary (0 or 1) column representing if the row has at least some
    quantity of missing values (NaN, None or pd.NA), returning 1 if the row has, and
    0 if not.

    Args:
      data (pd.DataFrame): Dataframe to be analyzed
      columns (list[str]): Columns that will be analyzed. If empty, will analyze
                           all columns
      count (int): minimum count of missing values (NaN, None or pd.NA) to consider
                   the binary column as 1.
                   Cannot be greater than the number of analyzed columns.
      indicator_name (str): Name for indicator column. "indicate" by default.
                            Must be different from all existing column names.

    Returns:
      (pd.DataFrame): a copy of the original DataFrame with one additional binary column.
    """

    # Create a copy of dataframe to prevent changes on original dataframe
    df = data.copy()

    # Pandas treats tuple as a single key, so normalize to list
    columns = list(columns)

    # Validate count value
    if not pd.api.types.is_integer(count) or count <= 0:
        raise ValueError("'count' needs to be an integer greater than 0")

    # Validate columns (existing)
    for column in columns:
        if column not in data.columns:
            raise ValueError(f"'{column}' column does not exist on dataframe")

    # Prevent column overwrite
    if indicator_name in df.columns:
        raise ValueError(
            f"'{indicator_name}' is an existing column on dataframe. Give another name."
        )

    # If no column is given, analyze all columns
    if not columns:
        columns = list(df.columns)

    # A row cannot have more missing values than analyzed columns
    if count > len(columns):
        raise ValueError(
            f"'count' ({count}) cannot be greater than the number of analyzed "
            f"columns ({len(columns)})"
        )

    # True/False column: does the row have at least 'count' missing values?
    # isnull() detects NaN, None and pd.NA
    df[indicator_name] = df[columns].isnull().sum(axis=1) >= count
    # Converts True/False -> 1/0
    df[indicator_name] = df[indicator_name].astype(int)

    return df
