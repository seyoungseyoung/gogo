import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import r2_score, mean_absolute_error

def basic_config():
    config = {
        'input_steps': 40,
        'hidden_dim': 256,
        'num_layers': 6,
        'batch_size': 32,
        'epochs': 500,
        'learning_rate': 3e-6,
        'val_ratio': 0,
        'test_ratio': 0,
        'target_col': 'SamsungE_Close',
        'date_col': 'Date',
        'shuffle_data': False,
        'weight_decay': 0,
        'seed': 123
    }
    return config

def load_csv(file_path: str):
    df = pd.read_csv(file_path)
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def extract_features(df):
    """
    뉴스와 비뉴스 피처를 자동 분리.
    """
    nonnews_cols = ['SamsungE_Open', 'SamsungE_High', 'SamsungE_Low', 'SamsungE_Close', 'SamsungE_Volume']
    news_cols = [col for col in df.columns if col.startswith("News_")]
    X_nonnews = df[nonnews_cols].values
    X_news = df[news_cols].values
    return X_nonnews, X_news

def create_sequences(X_nonnews, X_news, y, seq_length=30):
    X_nonnews_seq = []
    X_news_seq = []
    y_seq = []
    for i in range(seq_length, len(X_nonnews)):
        X_nonnews_seq.append(X_nonnews[i-seq_length:i])
        X_news_seq.append(X_news[i-seq_length:i])
        y_seq.append(y[i])
    return (
        np.array(X_nonnews_seq, dtype=np.float32),
        np.array(X_news_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32)
    )

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae

def save_plot(df, y_true, y_pred, date_col='Date', days=365, output_path=None):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    df_plot = df.copy()
    df_plot['Actual'] = y_true
    df_plot['Pred'] = y_pred

    last_date = df_plot[date_col].max()
    cutoff_date = last_date - pd.Timedelta(days=days)
    df_plot = df_plot[df_plot[date_col] >= cutoff_date]

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot[date_col], df_plot['Actual'], label='Actual')
    plt.plot(df_plot[date_col], df_plot['Pred'], label='Prediction')
    plt.title(f"Last {days} days comparison (Actual vs. Prediction)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def split_data(df, config):
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']
    date_col = config['date_col']

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    total_len = len(df)
    test_size = int(total_len * test_ratio)
    val_size = int(total_len * val_ratio)

    test_start = total_len - test_size
    val_start = test_start - val_size

    df_train = df.iloc[:val_start]
    df_val   = df.iloc[val_start:test_start] if val_size > 0 else pd.DataFrame()
    df_test  = df.iloc[test_start:] if test_size > 0 else pd.DataFrame()

    return df_train, df_val, df_test
