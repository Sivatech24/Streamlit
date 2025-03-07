import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import seaborn as sns
import warnings
import datetime as dt
from sklearn.metrics import confusion_matrix
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset
import streamlit as st
# from pmdarima.arima import auto_arima
# from statsmodels.tsa.stattools import adfuller
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
# from pmdarima import auto_arima
# from pandas.tseries.offsets import DateOffset
from prophet import Prophet
warnings.filterwarnings('ignore')

"""# Load Generation Data (Plant 1)"""

from sklearn.model_selection import train_test_split
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

default_gen_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('https://github.com/Sivatech24/Streamlit/raw/refs/heads/main/Plant_1_Weather_Sensor_Data.csv')

if gen_data is None:
    gen_data = default_gen_data
    gen_1 = default_gen_data
if weather_data is None:
    weather_data = default_weather_data
    sens_1 = default_weather_data

# Data Preview
st.subheader("Generation Data Preview Head")
st.dataframe(gen_data.head())

st.subheader("Weather Data Preview Head")
st.dataframe(weather_data.head())

st.subheader("Generation Data Preview Tail")
st.dataframe(gen_data.tail())

st.subheader("Weather Data Preview Tail")
st.dataframe(weather_data.tail())

st.subheader("Generation Data Describe")
st.dataframe(gen_data.describe())

st.subheader("Weather Data Describe")
st.dataframe(weather_data.describe())

# Filter out non-numeric columns
numeric_data = gen_1.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix on the numeric data
corelation = numeric_data.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corelation, annot=True, ax=ax)
st.subheader("Generation Data HeatMap")
st.pyplot(fig)

# Filter out non-numeric columns
numeric_data = sens_1.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix on the numeric data
corelation = numeric_data.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corelation, annot=True, ax=ax)
st.subheader("Weather Data HeatMap")
st.pyplot(fig)




st.subheader("Datetime Conversion and Resampling")

# Convert DATE_TIME column to datetime format
gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
gen_1['DATE_TIME'] = pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')
sens_1['DATE_TIME'] = pd.to_datetime(sens_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Resample generation data daily
gen_data_daily = gen_data.set_index('DATE_TIME').resample('D').sum().reset_index()

# Display processed data
st.subheader("Daily Aggregated Generation Data")
st.dataframe(gen_data_daily.head())




st.subheader("Daily Yield and Power Analysis")

# Group by DATE_TIME and sum values
df_gen = gen_1.groupby('DATE_TIME').sum().reset_index()
df_gen['time'] = df_gen['DATE_TIME'].dt.time

# Create subplots
fig, ax = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(20, 5))

# Daily yield plot
df_gen.plot(x='DATE_TIME', y='DAILY_YIELD', color='navy', ax=ax[0])
ax[0].set_title('Daily Yield')
ax[0].set_ylabel('kW', color='navy', fontsize=17)

# AC & DC power plot
df_gen.set_index('time').drop('DATE_TIME', axis=1)[['AC_POWER', 'DC_POWER']].plot(style='o', ax=ax[1])
ax[1].set_title('AC Power & DC Power During Day Hours')

# Display the plots in Streamlit
st.pyplot(fig)

st.subheader("Generation Data Analysis")

# Create subplots
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# Plot Daily and Total Yield
gen_data.plot(x='DATE_TIME', y=['DAILY_YIELD', 'TOTAL_YIELD'], ax=ax[0], title="Daily and Total Yield (Generation Data)")

# Plot AC Power & DC Power
gen_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax[1], title="AC Power & DC Power (Generation Data)")

# Display the plots in Streamlit
st.pyplot(fig)

st.subheader("Daily and Total Yield Analysis")

# Create a copy and extract the date
daily_gen = df_gen.copy()
daily_gen['date'] = daily_gen['DATE_TIME'].dt.date

# Group by 'date' and sum only the numerical columns
daily_gen = daily_gen.groupby('date').sum(numeric_only=True)

# Create subplots
fig, ax = plt.subplots(ncols=2, dpi=100, figsize=(20, 5))

# Plot Daily Yield
daily_gen['DAILY_YIELD'].plot(ax=ax[0], color='navy')
ax[0].set_title('Daily Yield')
ax[0].set_ylabel('kW', color='navy', fontsize=17)

# Plot Total Yield as a bar chart
daily_gen['TOTAL_YIELD'].plot(kind='bar', ax=ax[1], color='navy')
ax[1].set_title('Total Yield')

# Adjust x-axis labels
fig.autofmt_xdate(rotation=45)

# Display the plots in Streamlit
st.pyplot(fig)

st.subheader("Weather Sensor Data Analysis")

# Group by 'DATE_TIME' and sum
df_sens = sens_1.groupby('DATE_TIME').sum().reset_index()
df_sens['time'] = df_sens['DATE_TIME'].dt.time

# Create subplots
fig, ax = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(20, 5))

# Irradiation plot
df_sens.plot(x='time', y='IRRADIATION', ax=ax[0], style='o')
ax[0].set_title('Irradiation during day hours')
ax[0].set_ylabel('W/m²', color='navy', fontsize=17)

# Ambient and Module Temperature plot
df_sens.set_index('DATE_TIME').drop('time', axis=1)[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].plot(ax=ax[1])
ax[1].set_title('Ambient and Module Temperature')
ax[1].set_ylabel('°C', color='navy', fontsize=17)

# Display the plots in Streamlit
st.pyplot(fig)

st.subheader("DC to AC Power Conversion Efficiency")

# Create a copy of the data
loss = gen_1.copy()

# Create a new 'day' column containing only the date part from 'DATE_TIME'
loss['day'] = loss['DATE_TIME'].dt.date

# Drop the 'DATE_TIME' column to prevent summing over datetime values
loss = loss.drop(columns=['DATE_TIME'])

# Group by 'day' and sum only numeric columns
loss = loss.groupby('day').sum()

# Calculate the percentage of DC power converted to AC power
loss['losses'] = (loss['AC_POWER'] / loss['DC_POWER']) * 100

# Plot the losses
fig, ax = plt.subplots(figsize=(17, 5))
loss['losses'].plot(style='o--', ax=ax, label='Real Power')

# Plot styling
ax.set_title('% of DC Power Converted to AC Power', size=17)
ax.set_ylabel('DC Power Converted (%)', fontsize=14, color='red')
ax.axhline(loss['losses'].mean(), linestyle='--', color='gray', label='Mean')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("DC Power During the Day for All Sources")

# Create a copy of the data
sources = gen_1.copy()
sources['time'] = sources['DATE_TIME'].dt.time

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10))
sources.set_index('time').groupby('SOURCE_KEY')['DC_POWER'].plot(style='o', ax=ax, legend=True)

# Plot styling
ax.set_title('DC Power During the Day for All Sources', size=17)
ax.set_ylabel('DC POWER (kW)', color='navy', fontsize=17)

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("DC Power Distribution Across Sources")

# Create a copy of the data
dc_gen = gen_1.copy()
dc_gen['time'] = dc_gen['DATE_TIME'].dt.time

# Group by 'time' and 'SOURCE_KEY', then calculate the mean
dc_gen = dc_gen.groupby(['time', 'SOURCE_KEY'])['DC_POWER'].mean().unstack()

# Define the color palette
cmap = sns.color_palette("Spectral", n_colors=12)

# Create subplots
fig, ax = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(20, 6))

# Plot the first 11 sources
dc_gen.iloc[:, 0:11].plot(ax=ax[0], color=cmap)
ax[0].set_title('First 11 Sources')
ax[0].set_ylabel('DC POWER (kW)', fontsize=17, color='navy')

# Plot the last 11 sources
dc_gen.iloc[:, 11:22].plot(ax=ax[1], color=cmap)
ax[1].set_title('Last 11 Sources')

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Weather Data Analysis")

# Create subplots
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# Plot Irradiation
weather_data.plot(x='DATE_TIME', y='IRRADIATION', ax=ax[0], title="Irradiation (Weather Data)")

# Plot Ambient & Module Temperature
weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax[1], title="Ambient & Module Temperature (Weather Data)")

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("DC Power Converted with 2% Loss Assumption")

# Calculate DC power after conversion loss
gen_data['DC_POWER_CONVERTED'] = gen_data['DC_POWER'] * 0.98  # Assume 2% loss in conversion

# Create the plot
fig, ax = plt.subplots(figsize=(15, 5))
gen_data.plot(x='DATE_TIME', y='DC_POWER_CONVERTED', ax=ax, title="DC Power Converted")

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("DC Power Generated During Day Hours")

# Filter data for day hours (6 AM to 6 PM)
day_data_gen = gen_data[(gen_data['DATE_TIME'].dt.hour >= 6) & (gen_data['DATE_TIME'].dt.hour <= 18)]

# Create the plot
fig, ax = plt.subplots(figsize=(15, 5))
day_data_gen.plot(x='DATE_TIME', y='DC_POWER', ax=ax, title="DC Power Generated During Day Hours")

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Prepare Data for Time-Based Analysis")

# Copy generation data and extract time and date
temp1_gen = gen_1.copy()
temp1_gen['time'] = temp1_gen['DATE_TIME'].dt.time
temp1_gen['day'] = temp1_gen['DATE_TIME'].dt.date

# Copy sensor data and extract time and date
temp1_sens = sens_1.copy()
temp1_sens['time'] = temp1_sens['DATE_TIME'].dt.time
temp1_sens['day'] = temp1_sens['DATE_TIME'].dt.date

# Compute mean DC power grouped by time and day
cols = temp1_gen.groupby(['time', 'day'])['DC_POWER'].mean().unstack()

st.write("Data prepared successfully!")

st.subheader("Time-Based DC Power and Daily Yield Analysis")

# Create subplots for DC_POWER and DAILY_YIELD
fig, ax = plt.subplots(nrows=17, ncols=2, sharex=True, figsize=(20, 30))

# Plot DC Power
dc_power_plot = temp1_gen.groupby(['time', 'day'])['DC_POWER'].mean().unstack().plot(
    sharex=True, subplots=True, layout=(17, 2), figsize=(20, 30), ax=ax
)

# Plot Daily Yield
daily_yield_plot = temp1_gen.groupby(['time', 'day'])['DAILY_YIELD'].mean().unstack().plot(
    sharex=True, subplots=True, layout=(17, 2), figsize=(20, 20), style='-.', ax=ax
)

# Add titles and legends
i = 0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a, b].set_title(cols.columns[i], size=15)
        ax[a, b].legend(['DC_POWER', 'DAILY_YIELD'])
        i += 1

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Temperature Analysis Over Time")

# Create subplots for MODULE_TEMPERATURE and AMBIENT_TEMPERATURE
fig, ax = plt.subplots(nrows=17, ncols=2, sharex=True, figsize=(20, 30))

# Plot Module Temperature
module_temp_plot = temp1_sens.groupby(['time', 'day'])['MODULE_TEMPERATURE'].mean().unstack().plot(
    subplots=True, layout=(17, 2), figsize=(20, 30), ax=ax
)

# Plot Ambient Temperature
ambient_temp_plot = temp1_sens.groupby(['time', 'day'])['AMBIENT_TEMPERATURE'].mean().unstack().plot(
    subplots=True, layout=(17, 2), figsize=(20, 40), style='-.', ax=ax
)

# Add titles, legends, and threshold line
i = 0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a, b].axhline(50, color='r', linestyle='--', label='Threshold (50°C)')
        ax[a, b].set_title(cols.columns[i], size=15)
        ax[a, b].legend(['Module Temperature', 'Ambient Temperature', 'Threshold'])
        i += 1

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Worst Performing Source Analysis")

# Filter the worst-performing source
worst_source = gen_1[gen_1['SOURCE_KEY'] == 'bvBOhCH3iADSZry'].copy()
worst_source['time'] = worst_source['DATE_TIME'].dt.time
worst_source['day'] = worst_source['DATE_TIME'].dt.date

# Create subplots
fig, ax = plt.subplots(nrows=17, ncols=2, sharex=True, figsize=(20, 30))

# Plot DC Power for worst source
dc_power_plot = worst_source.groupby(['time', 'day'])['DC_POWER'].mean().unstack().plot(
    subplots=True, layout=(17, 2), figsize=(20, 30), ax=ax
)

# Plot Daily Yield for worst source
daily_yield_plot = worst_source.groupby(['time', 'day'])['DAILY_YIELD'].mean().unstack().plot(
    subplots=True, layout=(17, 2), figsize=(20, 30), ax=ax, style='-.'
)

# Add titles and legends
i = 0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a, b].set_title(cols.columns[i], size=15)
        ax[a, b].legend(['DC_POWER', 'DAILY_YIELD'])
        i += 1

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Inverter Performance Analysis")

# Calculate average DC power for each inverter
inverter_performance = gen_data.groupby('SOURCE_KEY')['DC_POWER'].mean().sort_values()

# Identify the underperforming inverter
underperforming_inverter = inverter_performance.idxmin()

# Display the result
st.write(f"**Underperforming Inverter:** {underperforming_inverter}")

# Plot inverter performance
fig, ax = plt.subplots(figsize=(12, 6))
inverter_performance.plot(kind='bar', ax=ax, color='navy')
ax.set_title("Average DC Power by Inverter")
ax.set_ylabel("DC Power (kW)")
ax.set_xlabel("Inverter (SOURCE_KEY)")
plt.xticks(rotation=90)

# Display the plot in Streamlit
st.pyplot(fig)
st.subheader("Module and Ambient Temperature (Weather Data)")

# Plot the temperature data
fig, ax = plt.subplots(figsize=(15, 5))
weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax, title="Module and Ambient Temperature (Weather Data)")

# Display the plot in Streamlit
st.pyplot(fig)

st.subheader("Inverter bvBOhCH3iADSZry Performance")

# Filter data for the specific inverter
inverter_data = gen_data[gen_data['SOURCE_KEY'] == 'bvBOhCH3iADSZry']

# Plot AC and DC power
fig, ax = plt.subplots(figsize=(15, 5))
inverter_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax, title="Inverter bvBOhCH3iADSZry")

# Display the plot in Streamlit
st.pyplot(fig)


# done




# Load data
df = gen_1.copy()
df = df.groupby('DATE_TIME').sum().reset_index()
df = df[['DATE_TIME', 'DAILY_YIELD']]
df.columns = ['ds', 'y']  # Prophet requires columns as 'ds' (datetime) and 'y' (value)

# Train Prophet model
model = Prophet()
model.fit(df)

# Create future dates for prediction (next 2 days)
future = model.make_future_dataframe(periods=96, freq='15min')

# Predict
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
model.plot_components(forecast)
plt.show()
