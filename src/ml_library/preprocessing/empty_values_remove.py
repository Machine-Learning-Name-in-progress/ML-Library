import pandas as pd


def empty_values_remove(
    data: pd.DataFrame, any_missing: bool = True, columns: list[str] = ()
) -> pd.DataFrame:
    """
    Remove empty values from the DataFrame. Removes rows according to whether they contain
    some NaN values in selected columns, or all selected values are NaN.

    Args:
        data (pd.DataFrame): Pandas DataFrame where will be removed empty values
        any_missing (bool): If True, removes rows with any missing values; if False, will
                            remove a row only if all selected columns are empty.
                            True by default.
        columns (list[str]): analyze only the given columns. If empty, all columns will be
                             considered

    Returns:
        (pd.DataFrame): new dataframe with removed rows. Index is reset
    """
    method = "any" if any_missing else "all"

    df = data.copy()

    # If a tuple is given, then convert to list
    columns = list(columns)

    # Validate columns (existing)
    for column in columns:
        if column not in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    # If no column is given, analyze all columns
    if not columns:
        columns = list(df.columns)

    df = df.dropna(how=method, subset=columns)

    # Reset index to 0,1,...
    return df.reset_index(drop=True)
