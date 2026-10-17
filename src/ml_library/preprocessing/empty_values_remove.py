import pandas as pd


def empty_values_remove(data: pd.DataFrame, any_missing: bool = True) -> pd.DataFrame:
    """
    Remove empty values from the DataFrame. All rows where atleast have 1 empty
    column will be removed

    Args:
        data (pd.DataFrame): Pandas DataFrame where will be removed empty values
        any_missing (bool): if False, will remove a row only if all columns are empty.
                            "True" by standard
    """
    method = "any" if any_missing else "all"

    return data.dropna(how=method)
