import requests
import yfinance as yf
import datetime
import joblib
import os
from data_engineering.data_utils import add_lags
import pandas as pd

end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=20)
ticker = ['PBR']
scaler = joblib.load(
            os.path.join("ml_model_training", "scaler_petro.joblib")
        )

#data = yf.download(tickers=ticker, start=star_date, end=end_date)['Close'].sort_index()

def extract_yf_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return df

def get_data_to_predict():
    pbr = extract_yf_data('PBR', start_date, end_date)[['Close']]
    pbr.columns = ['pbr']
    pbr.reset_index(inplace=True)
    pbr['Date'] = pd.to_datetime(pbr['Date']).dt.date
    pbr.set_index('Date', inplace=True)
    lagged_data = add_lags(data=pbr, num_lags=7, columns=['pbr'])
    return lagged_data.tail(1)

data = get_data_to_predict()
print(data)

# input_data2 = {
#     "input": {
#         "pbr": 10.5,
#         "brent": 60.2,
#         "wti": 58.7,
#         "production": 2500000,
#         "usd": 5.3,
#     }
# }

input_data = {
    "input": {
        "pbr_(t-7)":data['pbr_(t-7)'].values[0],
        "pbr_(t-6)":data['pbr_(t-6)'].values[0],
        "pbr_(t-5)":data['pbr_(t-5)'].values[0],
        "pbr_(t-4)":data['pbr_(t-4)'].values[0],
        "pbr_(t-3)":data['pbr_(t-3)'].values[0],
        "pbr_(t-2)":data['pbr_(t-2)'].values[0],
        "pbr_(t-1)":data['pbr_(t-1)'].values[0]
    }
}

print(input_data.values)

response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

# Print status code and response content
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

# Try to parse JSON only if the status code is successful
if response.status_code == 200:
    try:
        print("JSON Response:", response.json())
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
else:
    print(f"Request failed with status code: {response.status_code}")
