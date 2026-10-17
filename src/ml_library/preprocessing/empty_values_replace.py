import pandas as pd

# TODO: non-numerical columns existing


def empty_values_replace(
    data: pd.DataFrame, method: str, columns: list[str] = (), input: any = None
) -> pd.DataFrame:
    """
    Replaces empty values by a specific value in columns of the dataframe.

    Args:
      data (pd.DataFrame): Pandas DataFrame where will be replaced empty values

      method {0 or "mean", 1 or "median", 2 or "input"}: defines the replace method:
          - 0 or "mean": will replace empty values by column mean (numerical only)
          - 1 or "median": will replace empty values by column median (numerical only)
          - 2 or "input": will replace empty values by provided "input" parameter

      columns (list[str]):  columns where will be replaced. If empty, all columns
                            will be replaced. If "mean" or "median" method is provided,
                            only numerical columns will be replaced.
      input (any): value that will replace empty values
    """

    if not method in [0, 1, 2, "mean", "median", "input"]:
        raise ValueError(f"'{method}' is not a valid method")

    for column in columns:
        if not column in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    df = data.copy()

    if method in [0, "mean"]:
        if columns == ():
            columns = list(df.select_dtypes(include="number"))

        for column in columns:
            column_mean = df[column].mean()
            df[column] = df[column].fillna(column_mean)

        return df

    elif method in [1, "median"]:
        if columns == ():
            columns = list(df.select_dtypes(include="number"))

        for column in columns:
            column_median = df[column].median()
            df[column] = df[column].fillna(column_median)

        return df

    elif method in [2, "input"]:
        if columns == ():
            columns = list(df.columns)

        for column in columns:
            df[column] = df[column].fillna(input)

        return df
