import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime, timezone
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.lake_connector import LakeConnector


class DataIngestionPipeline:
    """ 
    Pipeline for extract Petrobras data from Yahoo finance, clean, resample, fill null values and upload to Data Lake.
    """
    def __init__(self, ticker: str, start_date: datetime, end_date: datetime, lake_config:dict = None):
        """
        Inicialize pipeline with main parameters and set up data lake connection for optional upload.
        ---
        Params:
        ticker (str): 'PBR' Petrobras ticker
        start_date (datetime): Extraction initial date 
        end_date (datetime): Extraction final date
        lake_config (dict): Settings for optional connection to Lake Storage. Must contains service_provider 
            and file_path in storage. For example:
            lake_config = {"service_provider": "azure", "file_path": "petro/petro.csv"}
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lake_config = lake_config
        self.data = None

        if self.lake_config:
            service_provider = self.lake_config.get("service_provider")
            self.datalake_connector = LakeConnector(service_provider=service_provider)
        else:
            self.datalake_connector = None

    def extract_yf_data(self) -> pd.DataFrame:
        """
        Extract Petrobras data from Yahoo Finance.
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data.columns = self.data.columns.droplevel(1)
        return self.data
    
    def format_data(self) -> pd.DataFrame:
        """
        Keep only Close column and rename it to 'pbr'
        """
        self.data = self.data.filter(items=['Close'])
        self.data = self.data.rename(columns={'Close': 'pbr'})
        return self.data

    def reindex_resample_data(self) -> pd.DataFrame:
        """
        Reindex and resample data.
        """
        new_index = pd.date_range(start=self.start_date.date(), end=self.end_date.date(), freq='D')
        self.data = self.data.reindex(new_index)
        self.data.index = self.data.index.date
        self.data.index.name = 'Date'
        return self.data

    def drop_weekends_data(self) -> pd.DataFrame:
        """
        Remove weekend data because stock data does not have information collected on weekends.
        """
        self.data.index = pd.to_datetime(self.data.index)
        self.data.loc[:, "d_week"] = self.data.index.dayofweek
        self.data = self.data.loc[self.data['d_week'] <= 4].copy()
        self.data.drop(columns=['d_week'], inplace=True)
        return self.data

    def fill_data(self) -> pd.DataFrame:
        """
        Fill remaining null data by using the next valid observation.
        """
        self.data = self.data.bfill(limit=2)
        return self.data

    def upload_data(self) -> None:
        """
        Uploads data to the data lake if connection parameters are passed.
        """
        if self.datalake_connector is not None:
            try:
                file_path = self.lake_config.get("file_path")
                blob = self.datalake_connector.connect(file_path)
                csv_buffer = StringIO()
                self.data.to_csv(csv_buffer, index=True)
                blob.upload_blob(csv_buffer.getvalue(), overwrite=True)
            except Exception as e:
                print(f"Upload failed.{e}")
            else:
                print('Upload sucessful')
                return True
        else:
            print('Lake provider config not received')

    def run_pipeline(self) -> pd.DataFrame:
        """
        Run all pipeline steps.
        """
        self.extract_yf_data()
        self.format_data()
        self.reindex_resample_data()
        self.drop_weekends_data()
        self.fill_data()

        if self.lake_config:
            self.upload_data()

        return self.data

if __name__ == "__main__":
    
    ticker = 'PBR'
    start_date = datetime.strptime('2008-01-01', '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    # lake_config = {"service_provider": "azure","file_path": "path/filename.csv"}
    datapipeline = DataIngestionPipeline(ticker=ticker, start_date=start_date, end_date=end_date)
    petro = datapipeline.run_pipeline()
    print(petro)
    # petro.to_csv('petro.csv')