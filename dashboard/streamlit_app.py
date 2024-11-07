import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Petrobras Price Prediction", page_icon="🛢️", layout="wide"
)

# Title
st.title("🇧🇷🛢️ Petrobras Price Prediction")


# Function to get historical data
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


# Function to get prediction from API
def get_prediction(historical_prices):
    input_data = {
        "input": {
            "pbr_(t-7)": historical_prices[-7],
            "pbr_(t-6)": historical_prices[-6],
            "pbr_(t-5)": historical_prices[-5],
            "pbr_(t-4)": historical_prices[-4],
            "pbr_(t-3)": historical_prices[-3],
            "pbr_(t-2)": historical_prices[-2],
            "pbr_(t-1)": historical_prices[-1],
        }
    }

    try:
        response = requests.post(
            "https://petro-price-predict.onrender.com/predict",
            json=input_data,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            st.error("Error getting prediction from API")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


# Get historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Create columns for metrics
col1, col2, col3, col4, col5 = st.columns(5)

# Get data for each asset
pbr_data = get_stock_data("PBR", start_date, end_date)
brent_data = get_stock_data("BZ=F", start_date, end_date)
wti_data = get_stock_data("CL=F", start_date, end_date)
usd_data = get_stock_data("USDBRL=X", start_date, end_date)

# Get prediction directly
historical_prices = pbr_data["Close"].iloc[-7:].tolist()
prediction = get_prediction(historical_prices)

# Display current prices
with col1:
    st.metric(
        "Petrobras (PBR)",
        f"${pbr_data['Close'].iloc[-1]:.2f}",
        f"{((pbr_data['Close'].iloc[-1] - pbr_data['Close'].iloc[-2])/pbr_data['Close'].iloc[-2]*100):.2f}%",
    )

with col2:
    st.metric(
        "Brent Crude Oil",
        f"${brent_data['Close'].iloc[-1]:.2f}",
        f"{((brent_data['Close'].iloc[-1] - brent_data['Close'].iloc[-2])/brent_data['Close'].iloc[-2]*100):.2f}%",
    )

with col3:
    st.metric(
        "WTI Crude Oil",
        f"${wti_data['Close'].iloc[-1]:.2f}",
        f"{((wti_data['Close'].iloc[-1] - wti_data['Close'].iloc[-2])/wti_data['Close'].iloc[-2]*100):.2f}%",
    )

with col4:
    st.metric(
        "USD/BRL",
        f"R${usd_data['Close'].iloc[-1]:.2f}",
        f"{((usd_data['Close'].iloc[-1] - usd_data['Close'].iloc[-2])/usd_data['Close'].iloc[-2]*100):.2f}%",
    )

with col5:
    # Display prediction metrics
    if prediction:
        pred_change = (
            (prediction - pbr_data["Close"].iloc[-1]) / pbr_data["Close"].iloc[-1]
        ) * 100
        st.metric("Predicted Price", f"${prediction:.2f}", f"{pred_change:.2f}%")
    else:
        st.metric("Predicted Price", "N/A", "N/A")

if prediction:
    st.subheader("Next Day Price Prediction")

    # Create prediction dataframe
    dates = pd.date_range(end_date, end_date + timedelta(days=1), freq="D")
    pred_df = pd.DataFrame(
        {"Date": dates, "Price": [pbr_data["Close"].iloc[-1], prediction]}
    )

    # Create prediction chart
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=pbr_data.index,
            y=pbr_data["Close"],
            name="Historical",
            line=dict(color="blue"),
        )
    )

    # Add prediction
    fig.add_trace(
        go.Scatter(
            x=pred_df["Date"],
            y=pred_df["Price"],
            name="Prediction",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title="Petrobras Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
