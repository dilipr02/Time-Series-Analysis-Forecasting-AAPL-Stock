
# preprocessing.py
import pandas as pd
import numpy as np

def clean_stock_csv(input_path='data/AAPL_stock_data.csv', output_path='data/AAPL_stock_data_cleaned.csv'):
    df = pd.read_csv(input_path, header=None)
    try:
        pd.to_datetime(df.iloc[2,0])
        df = df.iloc[2:].reset_index(drop=True)
    except Exception:
        df = pd.read_csv(input_path)
        df.to_csv(output_path, index=False)
        return output_path
    if df.shape[1] >= 6:
        df = df.iloc[:, :6]
        df.columns = ['Date','Close','High','Low','Open','Volume']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['Close','High','Low','Open','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df['Adj Close'] = df.get('Adj Close', df['Close'])
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/AAPL_stock_data.csv')
    parser.add_argument('--output', default='data/AAPL_stock_data_cleaned.csv')
    args = parser.parse_args()
    out = clean_stock_csv(args.input, args.output)
    print('Saved cleaned CSV to:', out)
