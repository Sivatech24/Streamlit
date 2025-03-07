import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Random Forest for Solar Energy Prediction")

# File Upload for Generation Data
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="rf_gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="rf_weather")

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

X = df[features]
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Hyperparameters
n_estimators = st.slider("Select Number of Trees", 10, 500, 100)

# Train Model
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display Results
st.write(f"**Mean Absolute Error:** {mae:.4f}")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))
