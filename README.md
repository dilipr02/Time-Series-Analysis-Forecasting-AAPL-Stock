[requirements.txt](https://github.com/user-attachments/files/23297562/requirements.txt)# Time-Series-Analysis-Forecasting-AAPL-Stock

Objective:
To analyze historical Apple Inc. (AAPL) stock prices, apply statistical and deep-learning forecasting techniques (ARIMA, SARIMA, Prophet, LSTM), and deliver a reproducible pipeline with visualizations and a Power BI dashboard for stakeholders.

Data & Preprocessing:

Source: Historical daily AAPL stock data (Yahoo Finance).

Preprocessing: removed header artifacts, parsed dates, converted types, forward-filled missing values, created Adj Close proxy and calculated log returns (ln(AdjClose_t / AdjClose_{t-1})). Log returns used for stationarity tests.

Exploratory Analysis & Stationarity:

Visualized closing price trend, volume, and log returns.

ADF test showed prices are non-stationary, log returns are stationary — leading to differencing or modeling returns for ARIMA-type models.

Models & Approach:

ARIMA: baseline autoregressive integrated moving average for short-term trends.

SARIMA: extends ARIMA with seasonal components (monthly seasonality).

Prophet: decomposable model capturing trend & seasonality robustly with minimal tuning.

LSTM: deep sequential model trained on scaled windows of historical prices; best for learning complex temporal patterns.

Evaluation & Results:

Metrics: RMSE, MAE, MAPE computed on hold-out test set (~20% of data).

Summary: LSTM achieved the lowest RMSE in experiments, followed by Prophet and SARIMA; ARIMA served as a baseline. Exact numbers are in outputs/model_results_summary.csv.

Visualizations & Dashboard:

Deliverables include time-series plots, returns histogram, volume charts, and model forecast overlays.

Power BI guide and theme included for interactive stakeholder dashboards (KPIs: Latest Close, 30-day MA, Volatility, Best Model RMSE).

Limitations & Next Steps:

Current models are univariate. Adding exogenous signals (news sentiment, macro indicators) could improve forecasting.

Implement walk-forward validation and hyperparameter tuning (GridSearch / Bayesian) for production readiness.

Deploy live dashboard with scheduled refresh for operational use.

[Uploading requirements.txt…]()
pandas
numpy
matplotlib
scikit-learn
statsmodels
prophet; python_version>='3.8'
tensorflow
python-pptx

