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
from pmdarima.arima import auto_arima
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

default_gen_data = pd.read_csv('Plant_1_Generation_Data.csv')
default_weather_data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

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

"""# Format 'DATE_TIME' column to datetime"""

gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M')
sens_1['DATE_TIME']= pd.to_datetime(sens_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

"""# Daily Yield & AC/DC Power from Generation Data"""

gen_data_daily = gen_data.set_index('DATE_TIME').resample('D').sum().reset_index()

"""# Plot Daily Yield and AC/DC Power"""

df_gen = gen_1.groupby('DATE_TIME').sum().reset_index()
df_gen['time'] = df_gen['DATE_TIME'].dt.time

# Create figure and axes
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# Daily yield plot
df_gen.plot(x='DATE_TIME', y='DAILY_YIELD', color='navy', ax=ax[0])
ax[0].set_title('Daily yield')
ax[0].set_ylabel('kW', color='navy', fontsize=17)

# AC & DC power plot
df_gen.set_index('time').drop('DATE_TIME', axis=1)[['AC_POWER', 'DC_POWER']].plot(style='o', ax=ax[1])
ax[1].set_title('AC power & DC power during day hours')

# Display in Streamlit
st.pyplot(fig)

# Create another figure for additional plots
fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# Daily and Total Yield plot
gen_data.plot(x='DATE_TIME', y=['DAILY_YIELD', 'TOTAL_YIELD'], ax=ax2[0], title="Daily and Total Yield (Generation Data)")

# AC Power & DC Power plot
gen_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax2[1], title="AC Power & DC Power (Generation Data)")

# Display the second figure in Streamlit
st.pyplot(fig2)

# Create a copy and extract the date
daily_gen = df_gen.copy()
daily_gen['date'] = daily_gen['DATE_TIME'].dt.date

# Group by 'date' and sum only the numerical columns
daily_gen = daily_gen.groupby('date').sum(numeric_only=True)

# Plot the daily and total yield
fig, ax = plt.subplots(ncols=2, dpi=100, figsize=(20, 5))
daily_gen['DAILY_YIELD'].plot(ax=ax[0], color='navy')
daily_gen['TOTAL_YIELD'].plot(kind='bar', ax=ax[1], color='navy')

fig.autofmt_xdate(rotation=45)
ax[0].set_title('Daily Yield')
ax[1].set_title('Total Yield')
ax[0].set_ylabel('kW', color='navy', fontsize=17)
plt.show()

# Group by 'DATE_TIME' and sum
df_sens = sens_1.groupby('DATE_TIME').sum().reset_index()
df_sens['time'] = df_sens['DATE_TIME'].dt.time

# Plotting
fig, ax = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(20, 5))

# Irradiation plot
df_sens.plot(x='time', y='IRRADIATION', ax=ax[0], style='o')

# Ambient and Module Temperature plot
df_sens.set_index('DATE_TIME').drop('time', axis=1)[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].plot(ax=ax[1])

# Setting titles and labels
ax[0].set_title('Irradiation during day hours')
ax[1].set_title('Ambient and Module Temperature')
ax[0].set_ylabel('W/m²', color='navy', fontsize=17)
ax[1].set_ylabel('°C', color='navy', fontsize=17)

plt.show()

"""# % of DC power converted to AC power"""

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
loss['losses'].plot(style='o--', figsize=(17, 5), label='Real Power')

# Plot styling
plt.title('% of DC power converted to AC power', size=17)
plt.ylabel('DC power converted (%)', fontsize=14, color='red')
plt.axhline(loss['losses'].mean(), linestyle='--', color='gray', label='mean')
plt.legend()
plt.show()

"""# DC Power"""

sources=gen_1.copy()
sources['time']=sources['DATE_TIME'].dt.time
sources.set_index('time').groupby('SOURCE_KEY')['DC_POWER'].plot(style='o',legend=True,figsize=(20,10))
plt.title('DC Power during day for all sources',size=17)
plt.ylabel('DC POWER ( kW )',color='navy',fontsize=17)
plt.show()

"""# DC POWER ( kW )"""

dc_gen=gen_1.copy()
dc_gen['time']=dc_gen['DATE_TIME'].dt.time
dc_gen=dc_gen.groupby(['time','SOURCE_KEY'])['DC_POWER'].mean().unstack()

cmap = sns.color_palette("Spectral", n_colors=12)

fig,ax=plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,6))
dc_gen.iloc[:,0:11].plot(ax=ax[0],color=cmap)
dc_gen.iloc[:,11:22].plot(ax=ax[1],color=cmap)

ax[0].set_title('First 11 sources')
ax[0].set_ylabel('DC POWER ( kW )',fontsize=17,color='navy')
ax[1].set_title('Last 11 sources')
plt.show()

"""# Irradiation, Ambient and Module Temperature from Weather Data"""

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
weather_data.plot(x='DATE_TIME', y='IRRADIATION', ax=ax[0], title="Irradiation (Weather Data)")
weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax[1], title="Ambient & Module Temperature (Weather Data)")
plt.show()

"""# Real DC power converted (DC Power efficiency)"""

gen_data['DC_POWER_CONVERTED'] = gen_data['DC_POWER'] * 0.98  # Assume 2% loss in conversion
fig, ax = plt.subplots(figsize=(15, 5))
gen_data.plot(x='DATE_TIME', y='DC_POWER_CONVERTED', ax=ax, title="DC Power Converted")
plt.show()

"""# DC Power generated during day hours (Generation Data)"""

day_data_gen = gen_data[(gen_data['DATE_TIME'].dt.hour >= 6) & (gen_data['DATE_TIME'].dt.hour <= 18)]
fig, ax = plt.subplots(figsize=(15, 5))
day_data_gen.plot(x='DATE_TIME', y='DC_POWER', ax=ax, title="DC Power Generated During Day Hours")
plt.show()

"""# DC Power And Daily Yield"""

temp1_gen=gen_1.copy()

temp1_gen['time']=temp1_gen['DATE_TIME'].dt.time
temp1_gen['day']=temp1_gen['DATE_TIME'].dt.date


temp1_sens=sens_1.copy()

temp1_sens['time']=temp1_sens['DATE_TIME'].dt.time
temp1_sens['day']=temp1_sens['DATE_TIME'].dt.date

# just for columns
cols=temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack()

ax =temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
temp1_gen.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,20),style='-.',ax=ax)

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
        i=i+1

plt.tight_layout()
plt.show()

"""# Module Temperature And Ambient Temperature"""

ax= temp1_sens.groupby(['time','day'])['MODULE_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,30))
temp1_sens.groupby(['time','day'])['AMBIENT_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,40),style='-.',ax=ax)

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].axhline(50)
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['Module Temperature','Ambient Temperature'])
        i=i+1

plt.tight_layout()
plt.show()

"""# DC_POWER And DAILY_YIELD"""

worst_source=gen_1[gen_1['SOURCE_KEY']=='bvBOhCH3iADSZry']
worst_source['time']=worst_source['DATE_TIME'].dt.time
worst_source['day']=worst_source['DATE_TIME'].dt.date

ax=worst_source.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
worst_source.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
        i=i+1

plt.tight_layout()
plt.show()

"""# Inverter Analysis (Generation Data)"""

inverter_performance = gen_data.groupby('SOURCE_KEY')['DC_POWER'].mean().sort_values()
print(f"Underperforming inverter: {inverter_performance.idxmin()}")

"""# Module temperature and Ambient Temperature on PLANT_1 (Weather Data)"""

fig, ax = plt.subplots(figsize=(15, 5))
weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax, title="Module and Ambient Temperature (Weather Data)")
plt.show()

"""# Inverter in action (Generation Data)"""

inverter_data = gen_data[gen_data['SOURCE_KEY'] == 'bvBOhCH3iADSZry']
fig, ax = plt.subplots(figsize=(15, 5))
inverter_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax, title="Inverter bvBOhCH3iADSZry")
plt.show()

"""# Forecasting with ARIMA (Generation Data)"""

df_daily_gen = gen_data_daily[['DATE_TIME', 'DAILY_YIELD']].set_index('DATE_TIME')

"""# Testing for stationarity"""

result = adfuller(df_daily_gen['DAILY_YIELD'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

"""# Splitting the dataset"""

train_gen, test_gen = train_test_split(df_daily_gen, test_size=0.2, shuffle=False)

"""# ARIMA model"""

arima_model_gen = ARIMA(train_gen['DAILY_YIELD'], order=(5, 1, 0))
arima_fit_gen = arima_model_gen.fit()
forecast_arima_gen = arima_fit_gen.forecast(steps=len(test_gen))
test_gen['Forecast_ARIMA'] = forecast_arima_gen

"""# Plot ARIMA Forecast"""

fig, ax = plt.subplots(figsize=(15, 5))
train_gen['DAILY_YIELD'].plot(ax=ax, label='Training Data')
test_gen['DAILY_YIELD'].plot(ax=ax, label='Test Data')
test_gen['Forecast_ARIMA'].plot(ax=ax, label='ARIMA Forecast')
plt.legend()
plt.show()

"""# SARIMA Model for Seasonal Data"""

sarima_model = SARIMAX(train_gen['DAILY_YIELD'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(test_gen))
test_gen['Forecast_SARIMA'] = sarima_forecast

"""# Plot SARIMA Forecast"""

plt.figure(figsize=(15, 5))
train_gen['DAILY_YIELD'].plot(label='Train')
test_gen['DAILY_YIELD'].plot(label='Test')
test_gen['Forecast_SARIMA'].plot(label='SARIMA Forecast')
plt.legend()
plt.title('SARIMA Model Forecast for Daily Yield (Generation Data)')
plt.show()

"""# SARIMAX vs ARIMA Comparison (Generation Data)"""

plt.figure(figsize=(15, 5))
plt.plot(test_gen.index, test_gen['DAILY_YIELD'], label='Actual Test Data')
plt.plot(test_gen.index, test_gen['Forecast_ARIMA'], label='ARIMA Forecast')
plt.plot(test_gen.index, test_gen['Forecast_SARIMA'], label='SARIMA Forecast')
plt.legend()
plt.title("ARIMA vs SARIMA Forecast Comparison (Generation Data)")
plt.savefig('first_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

"""# ARIMA Model"""

pred_gen=gen_1.copy()
pred_gen=pred_gen.groupby('DATE_TIME').sum()
pred_gen=pred_gen['DAILY_YIELD'][-288:].reset_index()
pred_gen.set_index('DATE_TIME',inplace=True)
pred_gen.head()

result = adfuller(pred_gen['DAILY_YIELD'])
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )

if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

train=pred_gen[:192]
test=pred_gen[-96:]
plt.figure(figsize=(15,5))
plt.plot(train,label='Train',color='navy')
plt.plot(test,label='Test',color='darkorange')
plt.title('Last 4 days of daily yield',fontsize=17)
plt.legend()
plt.show()

arima_model = auto_arima(train,start_p=0,d=1,start_q=0,max_p=4,max_d=4,max_q=4,start_P=0,D=1,start_Q=0,max_P=1,max_D=1,max_Q=1,m=96,seasonal=True,error_action='warn',trace=True,supress_warning=True,stepwise=True,random_state=20,n_fits=1)

future_dates = [test.index[-1] + DateOffset(minutes=x) for x in range(0,2910,15) ]

prediction=pd.DataFrame(arima_model.predict(n_periods=96),index=test.index)
prediction.columns=['predicted_yield']

fig,ax= plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(17,5))
ax[0].plot(train,label='Train',color='navy')
ax[0].plot(test,label='Test',color='darkorange')
ax[0].plot(prediction,label='Prediction',color='green')
ax[0].legend()
ax[0].set_title('Forecast on test set',size=17)
ax[0].set_ylabel('kW',color='navy',fontsize=17)


f_prediction=pd.DataFrame(arima_model.predict(n_periods=194),index=future_dates)
f_prediction.columns=['predicted_yield']
ax[1].plot(pred_gen,label='Original data',color='navy')
ax[1].plot(f_prediction,label='18th & 19th June',color='green')
ax[1].legend()
ax[1].set_title('Next days forecast',size=17)
plt.show()

arima_model.summary()
