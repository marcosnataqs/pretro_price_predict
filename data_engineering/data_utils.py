import pandas as pd


def add_lags(data: pd.DataFrame, num_lags: int, columns: list) -> pd.DataFrame:
    """
    This function will generate the lags for the columns I chose.
    This means that I can interate each every column I want and set a number or lags that I may use in my model later.
    """
    df = data.copy()
    for column in columns:
        for i in range(1, num_lags + 1):
            df[f"{column}_(t-{i})"] = df[column].shift(i)
    df.dropna(inplace=True)
    return df
