
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv('data/TSLA.csv')

df = df[['Close']].dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

def create_seq(data, step=60):
    X, y = [], []
    for i in range(step, len(data)):
        X.append(data[i-step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_seq(scaled)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM Model
lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=5, batch_size=32)

lstm.save('artifacts/lstm_model.h5')

print("Training Complete")
