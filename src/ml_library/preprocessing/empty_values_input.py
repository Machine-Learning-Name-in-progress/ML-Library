import pandas as pd


# TODO: implement a solution for possible errors:
#   - Wrong type of variables
#   - Non-numerical columns to analyze
#   - No columns finded (by provided name)
def empty_values_input(
    data: pd.DataFrame, method: str, columns: list[str] = [], input: any = None
) -> pd.DataFrame:
    """
    Replaces empty values by a value

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
    df = data.copy()

    if method in [0, "mean"]:
        if columns == []:
            columns = list(df.select_dtypes(include="number"))

        for column in columns:
            column_mean = df[columns].mean()
            df[column].fillna(column_mean)

        return df

    elif method in [1, "median"]:
        if columns == []:
            columns = list(df.select_dtypes(include="number"))

        for column in columns:
            column_median = df[columns].meadian()
            df[column].fillna(column_median)

        return df

    elif method in [2, "input"]:
        if columns == []:
            columns = list(df.columns())

        for column in columns:
            df[column] = input

        return df
