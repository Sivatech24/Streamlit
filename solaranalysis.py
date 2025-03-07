import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import seaborn as sns
import warnings
import datetime as dt
from sklearn.metrics import confusion_matrix
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset
import streamlit as st
# from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')

"""# Load Generation Data (Plant 1)"""

from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')

st.title("Solar Plant Data Analysis and Forecasting")

# File Upload
uploaded_gen = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="gen")
uploaded_weather = st.file_uploader("Upload Weather Sensor Data CSV", type=["csv"], key="weather")

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load Data
gen_data = load_data(uploaded_gen)
weather_data = load_data(uploaded_weather)

default_gen_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/blob/604e225828bb07cc3bfe6e9040b29e949dc1e73a/Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/blob/604e225828bb07cc3bfe6e9040b29e949dc1e73a/Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data
    gen_1 = default_gen_data
if weather_data is None:
    weather_data = default_weather_data
    sens_1 = default_weather_data

# Data Preview
st.subheader("Generation Data Preview")
st.dataframe(gen_data.head())

st.subheader("Weather Data Preview")
st.dataframe(weather_data.head())

st.subheader("Generation Data Preview")
st.dataframe(gen_data.tail())

st.subheader("Weather Data Preview")
st.dataframe(weather_data.tail())

st.subheader("Generation Data Preview")
st.dataframe(gen_data.describe())

st.subheader("Weather Data Preview")
st.dataframe(weather_data.describe())

# Filter out non-numeric columns
numeric_data = gen_1.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix on the numeric data
corelation = numeric_data.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corelation, annot=True, ax=ax)
st.pyplot(fig)

st.dataframe(sens_1.tail())

st.dataframe(sens_1.describe())

# Filter out non-numeric columns
numeric_data = sens_1.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix on the numeric data
corelation = numeric_data.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corelation, annot=True, ax=ax)
st.pyplot(fig)
