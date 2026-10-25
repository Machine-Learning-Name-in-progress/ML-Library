import pandas as pd


def one_hot(
    data: pd.DataFrame, drop: bool = False, columns: list[str] = ()
) -> pd.DataFrame:
    """
    Applies one-hot encoding for class variables, creating binary-columns for each class

    Args:
      data (pd.DataFrame): DataFrame to be encoded
      drop (bool): False by default. if True, remove the original columns, letting only the new binary columns.
      columns (list[str]): Columns of classification features that will be encoded. If empty, will encode all categorical columns

    Return:
      (pd.DataFrame) DataFrame with new one-hot encoding
    """

    # Create a copy to prevent changes to original dataframe.
    df = data.copy()

    # If no columns are given, consider all categorical columns
    if not columns:
        columns = list(df.select_dtypes(include=["object", "category"]))
        if not columns:
            raise ValueError("No categorical columns found to be encoded")

    # Validate columns
    else:
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

    # For each categorical column...
    for column in columns:
        # Validate current column (is it categorical?)
        if df[[column]].select_dtypes(include=["object", "category"]).empty:
            raise TypeError(
                f"{column} is not a categorical column, and cannot be encoded."
            )

        else:
            # Create a list with all categories of current column
            classes = list(df[column].unique())
            for classi in classes:
                # Create a binary (0,1) column for each category of current column
                df[f"{column}_{classi}_onehot"] = (df[column] == classi).astype(int)

            # Drop the original column if drop == True
            if drop:
                df.drop(columns=[column], inplace=True)

    return df
