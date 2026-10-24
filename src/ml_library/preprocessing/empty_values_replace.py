import pandas as pd

# TODO: non-numerical columns existing


def empty_values_replace(
    data: pd.DataFrame, method: str, columns: list[str] = (), fill_value=None
) -> pd.DataFrame:
    """
    Replaces empty values by a specific value in columns of the dataframe.

    Args:
      data (pd.DataFrame): Pandas DataFrame where will be replaced empty values

      method {0 or "mean", 1 or "median", 2 or "fill"}: defines the replace method:
          - 0 or "mean": will replace empty values by column mean (numerical only)
          - 1 or "median": will replace empty values by column median (numerical only)
          - 2 or "fill": will replace empty values by provided "fill_value" parameter

      columns (list[str]):  columns where will be replaced. If empty, all columns
                            will be replaced. If "mean" or "median" method is provided,
                            only numerical columns will be replaced.
      fill_value (any): Optional when not using fill method. Value that will replace empty values
    """

    if method not in [0, 1, 2, "mean", "median", "fill"]:
        raise ValueError(f"'{method}' is not a valid method")

    for column in columns:
        if not column in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    df = data.copy()

    if method in [0, "mean"]:
        if not columns:
            columns = list(df.select_dtypes(include="number"))
        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        for column in columns:
            column_mean = df[column].mean()
            df[column] = df[column].fillna(column_mean)

        return df

    elif method in [1, "median"]:
        if not columns:
            columns = list(df.select_dtypes(include="number"))

        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        for column in columns:
            column_median = df[column].median()
            df[column] = df[column].fillna(column_median)

        return df

    elif method in [2, "fill"]:
        if fill_value is None:
            raise ValueError(
                "'fill_value' parameter must be provided when using 'fill' method"
            )

        if not columns:
            columns = list(df.columns)

        for column in columns:
            df[column] = df[column].fillna(fill_value)

        return df
