import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.title("Tesla Stock Price Prediction")

# Load data
df = pd.read_csv('data/TSLA.csv')[['Close']].dropna()

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

step = 60

# Use last 60 values
last_data = scaled[-step:].reshape(1, step, 1)

# Since model cannot run on Streamlit Cloud, we use a fallback prediction
# (You already have real predictions in artifacts/future_predictions.csv)

try:
    pred_df = pd.read_csv("artifacts/future_predictions.csv")
    st.subheader("Next Predicted Prices")
    st.line_chart(pred_df)
    st.write("Latest Prediction:", pred_df.iloc[-1].values[0])

except Exception:
    st.warning("Prediction file not found. Showing sample output instead.")
    st.write("Predicted Next Close Price (demo): 0.00")
