
# modeling_evaluation.py
import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

def evaluate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def run_statistical_models(cleaned_csv='data/AAPL_stock_data_cleaned.csv', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv, parse_dates=['Date']).sort_values('Date')
    close = df['Close'].reset_index(drop=True)
    train_size = int(len(close) * 0.8)
    train, test = close[:train_size], close[train_size:]
    results = {}
    try:
        from statsmodels.tsa.arima.model import ARIMA
        arima = ARIMA(train, order=(5,1,0)).fit()
        arima_fore = arima.forecast(steps=len(test))
        results['ARIMA'] = evaluate_metrics(test.values, arima_fore.values)
        pd.DataFrame({'Date': df['Date'].iloc[train_size:].values, 'ARIMA_Pred': arima_fore.values}).to_csv(os.path.join(save_dir,'arima_forecast.csv'), index=False)
    except Exception as e:
        results['ARIMA'] = {'error': str(e)}
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        sarima_fore = sarima.forecast(steps=len(test))
        results['SARIMA'] = evaluate_metrics(test.values, sarima_fore.values)
        pd.DataFrame({'Date': df['Date'].iloc[train_size:].values, 'SARIMA_Pred': sarima_fore.values}).to_csv(os.path.join(save_dir,'sarima_forecast.csv'), index=False)
    except Exception as e:
        results['SARIMA'] = {'error': str(e)}
    try:
        from prophet import Prophet
        prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(prophet_df.iloc[:train_size])
        future = m.make_future_dataframe(periods=len(test))
        fcst = m.predict(future)
        prophet_pred = fcst['yhat'].iloc[-len(test):].values
        results['Prophet'] = evaluate_metrics(test.values, prophet_pred)
        pd.DataFrame({'Date': df['Date'].iloc[train_size:].values, 'Prophet_Pred': prophet_pred}).to_csv(os.path.join(save_dir,'prophet_forecast.csv'), index=False)
    except Exception as e:
        results['Prophet'] = {'error': str(e)}
    pd.DataFrame(results).to_csv(os.path.join(save_dir,'model_results_summary.csv'))
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned', default='data/AAPL_stock_data_cleaned.csv')
    parser.add_argument('--outdir', default='outputs')
    args = parser.parse_args()
    print('Running statistical models...')
    res = run_statistical_models(cleaned_csv=args.cleaned, save_dir=args.outdir)
    print('Statistical results:', res)
