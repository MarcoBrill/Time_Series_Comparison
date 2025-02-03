import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Input: Time Series Data (Pandas DataFrame with 'date' and 'value' columns)
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

# Output: Evaluation Metrics (MSE, MAE) and Plots
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}")
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted')
    plt.title(f'{model_name} Forecast')
    plt.legend()
    plt.show()
    return mse, mae

# ARIMA Model
def arima_forecast(train, test, order=(1, 1, 1)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

# SARIMA Model
def sarima_forecast(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))
    return forecast

# Exponential Smoothing
def exp_smoothing_forecast(train, test):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

# Prophet Model
def prophet_forecast(train, test):
    train_df = train.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    return forecast['yhat'][-len(test):]

# Main Function
def main():
    # Load Data
    file_path = 'data/time_series_data.csv'  # Replace with your data file path
    data = load_data(file_path)

    # Split Data into Train and Test
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # ARIMA
    arima_pred = arima_forecast(train['value'], test['value'])
    arima_mse, arima_mae = evaluate_model(test['value'], arima_pred, 'ARIMA')

    # SARIMA
    sarima_pred = sarima_forecast(train['value'], test['value'])
    sarima_mse, sarima_mae = evaluate_model(test['value'], sarima_pred, 'SARIMA')

    # Exponential Smoothing
    es_pred = exp_smoothing_forecast(train['value'], test['value'])
    es_mse, es_mae = evaluate_model(test['value'], es_pred, 'Exponential Smoothing')

    # Prophet
    prophet_pred = prophet_forecast(train['value'], test['value'])
    prophet_mse, prophet_mae = evaluate_model(test['value'], prophet_pred, 'Prophet')

    # Compare Results
    results = pd.DataFrame({
        'Model': ['ARIMA', 'SARIMA', 'Exponential Smoothing', 'Prophet'],
        'MSE': [arima_mse, sarima_mse, es_mse, prophet_mse],
        'MAE': [arima_mae, sarima_mae, es_mae, prophet_mae]
    })
    print(results)

if __name__ == "__main__":
    main()
