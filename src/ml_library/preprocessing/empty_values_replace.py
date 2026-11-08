import pandas as pd


def empty_values_replace(
    data: pd.DataFrame, method: str | int, columns: list[str] = (), fill_value=None
) -> tuple[pd.DataFrame, dict]:
    """
    Replaces empty values by a specific value in columns of the dataframe.

    Args:
      data (pd.DataFrame): Pandas DataFrame whose empty values will be replaced.

      method {0 or "mean", 1 or "median", 2 or "fill"}: defines the replace method:
          - 0 or "mean": will replace empty values by column mean (numerical only)
          - 1 or "median": will replace empty values by column median (numerical only)
          - 2 or "fill": will replace empty values by provided "fill_value" parameter

      columns (list[str]):  columns where NaN values will be replaced. If empty, all columns
                            will be replaced. If "mean" or "median" method is provided,
                            only numerical columns will be replaced.
      fill_value (any): Required when method is "fill". Value that will replace
                        empty values. Must be compatible with the type of every
                        selected column, so give "columns" when the DataFrame mixes
                        numerical and text columns

    Returns:
        tuple[pd.DataFrame, dict]: A copy of the DataFrame with empty values replaced,
                                   and a dict mapping each processed column to the value
                                   used to fill it.

    """

    # We need specifically a list, then convert if a tuple is given
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
        # If no columns are given, consider all numerical columns
        if not columns:
            columns = list(df.select_dtypes(include="number"))
        # Select only numerical columns from given list
        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        means = {}  # Empty dict to save means

        # Compute the mean, save and fill (essential for model tests)

        for column in columns:
            column_mean = df[column].mean()

            # Save on dict
            means[column] = column_mean

            df[column] = df[column].fillna(column_mean)

        return df, means

    # Method "median"
    elif method in [1, "median"]:
        # If no columns are given, consider all numerical columns
        if not columns:
            columns = list(df.select_dtypes(include="number"))
        # Select only numerical columns from given list
        else:
            columns = list(df[columns].select_dtypes(include="number").columns)

        medians = {}  # Empty dict to save medians

        # find the median, save and fill (essential for model tests)
        for column in columns:
            column_median = df[column].median()

            # Save on dict
            medians[column] = column_median

            df[column] = df[column].fillna(column_median)

        return df, medians

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

        fill_values = {}

        # Fill NaN values with fill_value
        for column in columns:
            df[column] = df[column].fillna(fill_value)
            fill_values[column] = fill_value

        return df, fill_values
