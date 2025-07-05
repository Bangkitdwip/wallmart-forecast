# 📊 Walmart Sales Forecasting App

A Streamlit web application to forecast weekly sales of Walmart stores and departments using multiple models including SARIMA, ARIMA, Random Forest, Linear Regression, and XGBoost.
🔗 [Lihat Aplikasinya di Streamlit](https://wallmart-forecast-wmjtf4m6sp66pxfh7tovu5.streamlit.app/)
---

## 🧰 Features

- Select Walmart **Store** and **Department**
- Choose forecasting model:  
  `SARIMA`, `ARIMA`, `Random Forest`, `Linear Regression`, or `XGBoost`
- Visualize historical vs forecasted weekly sales
- Display tabular weekly sales forecasts with actual comparisons

---

## 📁 Dataset Overview

This project uses three datasets:

- `train.csv` – Weekly sales data by store and department
- `features.csv` – External factors (Temperature, Fuel Price, CPI, etc.)
- `stores.csv` – Store characteristics (Type and Size)
