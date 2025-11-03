
# visualize_and_export.py
import os, pandas as pd, matplotlib.pyplot as plt

def create_visuals(cleaned_csv='data/AAPL_stock_data_cleaned.csv', outputs_dir='visuals'):
    os.makedirs(outputs_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv, parse_dates=['Date']).sort_values('Date')
    plt.figure(figsize=(10,4)); plt.plot(df['Date'], df['Close']); plt.title('AAPL Closing Price'); plt.xlabel('Date'); plt.ylabel('Price ($)')
    closing_path = os.path.join(outputs_dir,'closing_price.png'); plt.savefig(closing_path); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(df['Date'], df['Log_Return']); plt.title('AAPL Log Returns'); plt.xlabel('Date'); plt.ylabel('Log Return')
    lr_path = os.path.join(outputs_dir,'log_returns.png'); plt.savefig(lr_path); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(df['Date'], df['Volume']); plt.title('Volume'); plt.xlabel('Date'); plt.ylabel('Volume')
    vol_path = os.path.join(outputs_dir,'volume.png'); plt.savefig(vol_path); plt.close()
    print('Saved visuals to', outputs_dir)
    return {'closing': closing_path, 'log_returns': lr_path, 'volume': vol_path}

if __name__ == '__main__':
    create_visuals()
