import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Load data
@st.cache_data
def load_data():
    df_sales = pd.read_csv('train.csv')
    df_features = pd.read_csv('features.csv')
    df_stores = pd.read_csv('stores.csv')

    df = df_sales.merge(df_features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(df_stores, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.fillna(0)
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Walmart Sales Forecast")
store = st.sidebar.selectbox("Select Store", sorted(df['Store'].unique()))
dept = st.sidebar.selectbox("Select Department", sorted(df['Dept'].unique()))
model_choice = st.sidebar.selectbox("Select Model", ['SARIMA', 'ARIMA'])

st.title("Walmart Sales Forecast")
if st.button("Start Forecast"):
    # Filter data
    df_filtered = df[(df['Store'] == store) & (df['Dept'] == dept)].sort_values('Date')
    data = df_filtered[['Date', 'Weekly_Sales']].set_index('Date').resample('W').sum()
    data = data.fillna(0)

    train = data[:-12]
    y_val = data[-12:]

    forecast = None

    # Forecasting logic
    if model_choice == 'SARIMA':
        model = SARIMAX(train['Weekly_Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=12)
        forecast.index = y_val.index

    elif model_choice == 'ARIMA':
        model = ARIMA(train['Weekly_Sales'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)
        forecast.index = y_val.index

    # Remove any NaNs before displaying
    y_val = y_val.dropna()
    forecast = forecast.dropna()

    # Display forecast chart
    st.title(f"Forecast for Store {store} - Department {dept}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['Weekly_Sales'], label='Train')
    ax.plot(y_val.index, y_val['Weekly_Sales'], label='Actual')
    ax.plot(forecast.index, forecast, label='Forecast')
    ax.set_title(f"{model_choice} Forecast")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Display sales numbers
    st.subheader("Forecasted Weekly Sales Values")
    df_result = pd.DataFrame({
        'Date': forecast.index,
        'Forecasted_Sales': forecast.values,
        'Actual_Sales': y_val['Weekly_Sales'].values
    })
    st.dataframe(df_result.set_index('Date'))
