import pandas as pd


def empty_values_remove(
    data: pd.DataFrame, columns: list[str] = (), any_missing: bool = True
) -> pd.DataFrame:
    """
    Remove empty values from the DataFrame. Removes rows according to whether they contain
    some NaN values in selected columns, or all selected values are NaN.

    Args:
        data (pd.DataFrame): DataFrame from which rows with empty values will be removed
        columns (list[str]): analyze only the given columns. If empty, all columns will be
                             considered
        any_missing (bool): If True, removes rows with any missing values; if False, will
                            remove a row only if all selected columns are empty.
                            True by default.

    Returns:
        (pd.DataFrame): a new DataFrame without the removed rows, with reset index
    """
    method = "any" if any_missing else "all"

    df = data.copy()

    # If a tuple is given, then convert to list
    columns = list(columns)

    # Validate columns (existing)
    for column in columns:
        if column not in data.columns:
            raise ValueError(f"'{column}' column does not exist on dataframe")

    # If no column is given, analyze all columns
    if not columns:
        columns = list(df.columns)

    df = df.dropna(how=method, subset=columns)

    # Reset index to 0,1,...
    return df.reset_index(drop=True)
