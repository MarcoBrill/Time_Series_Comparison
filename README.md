# Time Series Analysis Comparison

This repository contains a Python script to compare different Time Series Analysis methodologies, including ARIMA, SARIMA, Exponential Smoothing, and Prophet. The script evaluates each model using Mean Squared Error (MSE) and Mean Absolute Error (MAE) and generates plots for visual comparison.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`

## Input
A CSV file (time_series_data.csv) containing:

date: The date in YYYY-MM-DD format.
value: The numeric value of the time series.

## Output
Plots of actual vs. predicted values for each model.

A table comparing the MSE and MAE of each model.

### How to Use

1. Replace `time_series_data.csv` with your dataset.
2. Run the script to compare the models and visualize the results.
3. Check the console for statistical results and the plots for visual comparisons.

Clone the repository:
   ```bash
   git clone https://github.com/yourusername/time-series-comparison.git
   cd time-series-comparison
