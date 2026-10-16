import pandas as pd

# TODO: error treatment


def empty_values_remove(data: pd.DataFrame, unique: bool = True) -> pd.DataFrame:
    """
    Remove empty values from the DataFrame. All rows where atleast have 1 empty
    column will be removed

    Args:
        data (pd.DataFrame): Pandas DataFrame where will be removed empty values
        unique (bool): if False, will remove a row only if all columns are empty.
                       "True" by standard
    """
    if unique:
        return data.dropna(how="any")
    else:
        return data.dropna(how="all")
