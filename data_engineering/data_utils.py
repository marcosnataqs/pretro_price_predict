import pandas as pd

from io import StringIO
from utils.lake_connector import LakeConnector

def download_data() -> pd.DataFrame:
    """
    Download Petrobras Time serie from Data Lake. 
    """
    dl_connection = LakeConnector('azure')
    blob_name = 'petro/petro.csv'
    blob = dl_connection.connect(blob_name)
    blob_data = blob.download_blob()
    blob_content = blob_data.readall()
    csv_data = blob_content.decode('utf-8')
    data = pd.read_csv(StringIO(csv_data), index_col=0, parse_dates=[0])
    data.index.name = 'Date'
    print(f'Download from silver/{blob_name} was sucessful.')
    return data


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

if __name__ == "__main__":
    
    df = download_data()
    print(df.tail())
    df = add_lags(df, 7, ['pbr'])
    print(df.tail())