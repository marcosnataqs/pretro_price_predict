import pandas as pd
import numpy as np
import torch

from joblib import dump
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class PetroDataset(Dataset):
    """
        The PetroDataset class requires you to make the train test split outside first without any prior treatment.
        After that the class will take care of everything else interactivelly as long as you pass the correct parameters in the constructor
    """

    def __init__(self, 
                 data: pd.DataFrame, 
                 num_features:int, 
                 pipeline_params: dict) -> None:
        self.pipeline_params = pipeline_params
        self.data = data
        self.num_features = num_features
        self.transform_data_pipeline()
    
    def transform_data_pipeline(self):
        """
            This is a custom method where will be implemented data transformations, 
            the idea is to simply pass the crude data to the class and PetroDataset will take care od the test.
        """
        self.add_lags()
        self.scale_data()
        self.reshape_data()

    def add_lags(self) -> None:
        """
            This method will use the pipeline parameters to generate the lags for the columns I chose.
            This means that I can interate each every column I want and set a number or lags that I may use in my model later.
        """
        for column in self.pipeline_params['columns']:
            for i in range(1, self.pipeline_params['num_lags']+1):
                self.data[f'{column}_(t-{i})'] = self.data[column].shift(i)
                self.data.dropna(inplace=True)
            self.num_features+=self.pipeline_params['num_lags']
    
    def scale_data(self) -> None:
        """
            This method is used to scale my train data in the pipeline,
            it also saves my scaler as a joblib/pickle so it can be used later. 
        """
        scaler_filename = "scaler_petro.joblib"
        scaler = MinMaxScaler(feature_range=(-1,1))
        self.data_scaled = scaler.fit_transform(self.data)
        dump(scaler, scaler_filename)

    def reshape_data(self) -> None:
        """
            Ths method will reshape my data spliting the target from the data.
            It also adjusts the dimensions of the numpy array so it matches the required dimensions for LSTM
        """
        X = np.flip(self.data_scaled[:, 1:], axis=1)
        y = self.data_scaled[:, 0]
        self.X = X.reshape((-1, (self.num_features-1), 1))
        self.y = y.reshape((-1, 1))
        print(self.X.shape, self.y.shape)
        print(type(self.X), type(self.y))
    
    def __len__(self) -> int:
        """
            Standard method from Dataset which is responsible for returning the len of my Dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        """
            Standard method from Dataset, where I am first retrieving my data as per index.
            After that I am transforming it into a Tensor so it can be consumed in LSTM
        """
        X = self.X[idx]
        y = self.y[idx]
        ## convert to tensor
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return X, y

if __name__ == '__main__':
    data = pd.read_csv('ml_model_training\petro.csv', index_col=0, parse_dates=True).sort_index()
    target = 'pbr'
    params = {
        'num_lags':3,
        'columns': ['pbr', 'usd'],
        'num_features': 6
    }
    petrodata = PetroDataset(data, target, 6, pipeline_params=params)
    data, target = petrodata[1]
    print(data, target)







