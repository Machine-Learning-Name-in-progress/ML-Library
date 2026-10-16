import pandas as pd

# TODO: Error treatments:
#   - Verify the existance of provided column
#   - Analyze and validate the parameters


def one_hot(
    data: pd.DataFrame, drop: bool = False, columns: list[str] = []
) -> pd.DataFrame:
    """
    Applies one-hot encoding for class variables, creating binary-columns for each class

    Args:
      data (pd.DataFrame): DataFrame to be encode
      drop (bool): False by standard. if True, remove the original columns,
                   letting only the new binary columns.
      columns (list[str]): Columns of classification variables that will be encoded.
                           If empty, will encode all cathegorical columns


    Return:
      (pd.DataFrame) DataFrame with new one-hot encoding
    """

    df = data.copy()

    if columns == []:
        columns = list(df.select_dtypes(include=["object", "category"]))

    # TODO: implement solution for "no cathegorical column" case

    for col in columns:
        classes = list(df[col].unique())
        for classi in classes:
            df[f"{col}_{classi}"] = (df[col] == classi).astype(int)

        if drop:
            df.drop(columns=[col], inplace=True)

    return df
