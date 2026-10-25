import pandas as pd


def one_hot(
    data: pd.DataFrame, drop: bool = False, columns: list[str] = ()
) -> pd.DataFrame:
    """
    Applies one-hot encoding for class variables, creating binary-columns for each class

    Args:
      data (pd.DataFrame): DataFrame to be encode
      drop (bool): False by standard. if True, remove the original columns, letting only the new binary columns.
      columns (list[str]): Columns of classification variables that will be encoded. If empty, will encode all categorical columns

    Return:
      (pd.DataFrame) DataFrame with new one-hot encoding
    """

    df = data.copy()

    if not columns:
        columns = list(df.select_dtypes(include=["object","category"]))
    else:
        if not columns:
            raise ValueError("No categorical columns found to encode")
        
        for column in columns:
            if column not in set(data.columns):
                raise ValueError(f"'{column}' column does not exist on dataframe")

    for col in columns:
        if df[[col]].select_dtypes(include=["object", "category"]).empty:
            print(f"{col} is not a categorical column, and cannot be encoded.")
            classes = list(df[col].unique())
            for classi in classes:
                df[f"{col}_{classi}_onehot"] = (df[col] == classi).astype(int)

            if drop:
                df.drop(columns=[col], inplace=True)

    return df
