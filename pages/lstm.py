import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.title("LSTM Model for Solar Energy Prediction")

# File Upload for Generation Data
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="lstm_gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="lstm_weather")

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load Data Separately
gen_data = load_data(uploaded_gen)
weather_data = load_data(uploaded_weather)

# Default Data (if no file is uploaded)
default_gen_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data
if weather_data is None:
    weather_data = default_weather_data

# Choose which dataset to use
dataset_choice = st.radio("Select dataset:", ("Generation Data", "Weather Data"))

if dataset_choice == "Generation Data":
    df = gen_data
    target_col = "DAILY_YIELD"
elif dataset_choice == "Weather Data":
    df = weather_data
    target_col = "MODULE_TEMPERATURE"

# Feature Selection
features = [col for col in df.columns if col not in ["DATE_TIME", target_col, "SOURCE_KEY"]]
st.write("Selected Features:", features)

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features + [target_col]])

# Create Sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, :-1])  # All features except target
        y.append(data[i + seq_length, -1])  # Target value
    return np.array(X), np.array(y)

seq_length = st.slider("Select Sequence Length", 1, 30, 10)
X, y = create_sequences(scaled_data, seq_length)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train Model
epochs = st.slider("Select Number of Epochs", 1, 100, 10)
progress_bar = st.progress(0)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16, verbose=1, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch+1)/epochs))])

# Predictions
y_pred = model.predict(X_test)

# Inverse Transform Predictions
y_test_actual = scaler.inverse_transform(np.hstack((X_test[:, -1, :], y_test.reshape(-1, 1))))[:, -1]
y_pred_actual = scaler.inverse_transform(np.hstack((X_test[:, -1, :], y_pred)))[:, -1]

# Display Results
st.write("LSTM Model Performance:")
st.line_chart(pd.DataFrame({"Actual": y_test_actual, "Predicted": y_pred_actual}))
