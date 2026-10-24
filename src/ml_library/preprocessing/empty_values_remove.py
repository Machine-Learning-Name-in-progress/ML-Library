import pandas as pd


def empty_values_remove(data: pd.DataFrame, any_missing: bool = True) -> pd.DataFrame:
    """
    Remove empty values from the DataFrame. Removes rows according to whether they contain any missing values NaN or only NaN.

    Args:
        data (pd.DataFrame): Pandas DataFrame where will be removed empty values
        any_missing (bool): If True, removes rows with any missing values; if False, will remove a row only if all columns are empty.

    Returns:
        (pd.DataFrame): new dataframe with removed rows
    """
    method = "any" if any_missing else "all"

    return data.dropna(how=method)
