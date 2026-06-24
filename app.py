
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Tesla Stock Price Prediction")

df = pd.read_csv('data/TSLA.csv')[['Close']].dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

model = load_model('artifacts/lstm_model.h5', compile=False)

step = 60

last_data = scaled[-step:].reshape(1, step, 1)

pred = model.predict(last_data)
pred_price = scaler.inverse_transform(pred)

st.write("Predicted Next Close Price:", float(pred_price[0][0]))
