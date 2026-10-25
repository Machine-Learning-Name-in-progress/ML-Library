import pandas as pd


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
      fill_value (any): Required when method is "fill". Value that will replace
                        empty values

    Returns:
        (pd.DataFrame): new dataframe with replaced values

    """

    # We need specifically a list, then convert if a tuple are given
    columns = list(columns)

    # Validate method
    if method not in [0, 1, 2, "mean", "median", "fill"]:
        raise ValueError(f"'{method}' is not a valid method")

    # Validate columns (existing)
    for column in columns:
        if column not in set(data.columns):
            raise ValueError(f"'{column}' column does not exist on dataframe")

    # Create a copy to prevent changes to the original dataframe
    df = data.copy()

    # Method "mean"
    if method in [0, "mean"]:
        # If no columns is given, consider all numerical columns
        if not columns:
            columns = list(df.select_dtypes(include="number"))
        # Select only numerical columns from given list
        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        # Compute the mean and fill
        for column in columns:
            column_mean = df[column].mean()
            df[column] = df[column].fillna(column_mean)

        return df

    # Method "median"
    elif method in [1, "median"]:
        # If no columns are given, consider all numerical columns
        if not columns:
            columns = list(df.select_dtypes(include="number"))
        # Select only numerical columns from given list
        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        # Find the median and fill
        for column in columns:
            column_median = df[column].median()
            df[column] = df[column].fillna(column_median)

        return df

    # Method "fill"
    elif method in [2, "fill"]:
        # Validate fill_value
        if fill_value is None:
            raise ValueError(
                "'fill_value' parameter must be provided when using 'fill' method"
            )

        # If no columns are given, then consider all
        if not columns:
            columns = list(df.columns)

        # Fill NaN values with fill_value
        for column in columns:
            df[column] = df[column].fillna(fill_value)

        return df
