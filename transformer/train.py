import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os
import matplotlib.pyplot as plt
import random

from utils import *
from transformer import TransformerForecast

import sys
from tqdm import tqdm
from captum.attr import IntegratedGradients

# 로그 파일 이름에 오늘 날짜/시간/분 포함
log_filename = datetime.now().strftime("train_transformer_%Y%m%d_%H%M.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    filemode="w"
)
logger = logging.getLogger(__name__)

def train_main(file_path):
    config = basic_config()
    input_steps = config['input_steps']
    d_model = config['hidden_dim']   # Transformer의 d_model로 사용
    num_layers = config['num_layers']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    target_col = config['target_col']
    wd = config.get('weight_decay', 0)
    seed = config.get('seed', 42)

    logger.info(f"config: lr = {lr}, d_model = {d_model}, num_layers = {num_layers}")

    # 시드 고정: numpy, torch, random 및 cudnn
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device 설정: CUDA 또는 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1) 데이터 로드 및 전처리
    df = load_csv(file_path)
    df.dropna(inplace=True)  # 결측치 제거
    df_train, df_val, df_test = split_data(df, config)
    feature_cols = [c for c in df.columns if c not in [config['date_col'], target_col]]
    news_columns = [col for col in feature_cols if "News" in col]
    non_news_columns = [col for col in feature_cols if "News" not in col]

    print(f"Non-News Columns Number: {len(non_news_columns)}\tNews Columns Number: {len(news_columns)}")
    
    # 2) Train, Val 데이터 준비
    X_train_nonnews = df_train[non_news_columns].values.astype(np.float32)
    X_train_news = df_train[news_columns].values.astype(np.float32)
    y_train = df_train[target_col].values.astype(np.float32)
    if not df_val.empty:
        X_val_nonnews = df_val[non_news_columns].values.astype(np.float32)
        X_val_news = df_val[news_columns].values.astype(np.float32)
        y_val = df_val[target_col].values.astype(np.float32)
    else:
        X_val_nonnews, X_val_news, y_val = None, None, None

    # 입력 피처 스케일링
    input_scaler_nonnews = StandardScaler()
    input_scaler_news = StandardScaler()

    X_train_nonnews_scaled = input_scaler_nonnews.fit_transform(X_train_nonnews)
    X_train_news_scaled = input_scaler_news.fit_transform(X_train_news)
    if X_val_nonnews is not None:
        X_val_nonnews_scaled = input_scaler_nonnews.transform(X_val_nonnews)
        X_val_news_scaled = input_scaler_news.transform(X_val_news)
    else:
        X_val_nonnews_scaled, X_val_news_scaled = None, None

    # 타겟 스케일링
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    if y_val is not None:
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    else:
        y_val_scaled = None

    # 3) Transformer 입력 시퀀스 생성 (Train & Val)
    X_train_nonnews_seq, X_train_news_seq, y_train_seq = create_sequences(X_train_nonnews_scaled, X_train_news_scaled, y_train_scaled, seq_length=input_steps)
    if X_val_nonnews_scaled is not None and len(X_val_nonnews_scaled) >= input_steps:
        X_val_nonnews_seq, X_val_news_seq, y_val_seq = create_sequences(X_val_nonnews_scaled, X_val_news_scaled, y_val_scaled, seq_length=input_steps)
    else:
        X_val_nonnews_seq, X_val_news_seq, y_val_seq = None, None, None

    train_dataset = TensorDataset(torch.from_numpy(X_train_nonnews_seq), torch.from_numpy(X_train_news_seq), torch.from_numpy(y_train_seq))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                              num_workers=2, pin_memory=True)

    # 4) 모델 구성
    input_size = X_train_nonnews_seq.shape[2]
    news_dim = X_train_news_seq.shape[2]  # 각 시점별 뉴스 feature 수

    model = TransformerForecast(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        input_steps=input_steps,
        news_dim=news_dim,
        nhead=4,
        output_size=1,
        dropout=0.1
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # 각 임계값 별 체크포인트가 저장되었는지 추적하는 dictionary
    saved_checkpoints = {0.88: False, 0.92: False, 0.96: False, 0.99: False}
    
    # 5) 학습 루프
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        running_loss = 0.0
        for X_batch_nonnews_batch, X_batch_news_batch, y_batch in train_loader:
            X_batch_nonnews_batch = X_batch_nonnews_batch.to(device)
            X_batch_news_batch = X_batch_news_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch_nonnews_batch, X_batch_news_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        logger.info(f"Epoch [{epoch}/{epochs}] Train Loss: {epoch_loss:.6f}")

        # 20 에폭마다 혹은 마지막 에폭에 전체 데이터 평가 및 체크포인트 저장
        t = 20
        if epoch > 160: t = 5
        if epoch % t == 0 or epoch == epochs:
            df_plot_base = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
            X_plot_nonnews = df_plot_base[non_news_columns].values.astype(np.float32)
            X_plot_news = df_plot_base[news_columns].values.astype(np.float32)

            y_plot = df_plot_base[target_col].values.astype(np.float32)
            
            # 입력 피처 스케일링 for plotting
            X_plot_nonnews_scaled = input_scaler_nonnews.transform(X_plot_nonnews)
            X_plot_news_scaled = input_scaler_news.transform(X_plot_news)
            y_plot_scaled = y_scaler.transform(y_plot.reshape(-1, 1)).flatten()
            if len(X_plot_nonnews_scaled) >= input_steps:
                X_plot_nonnews_seq_scaled, X_plot_news_seq_scaled, y_plot_seq_scaled = create_sequences(X_plot_nonnews_scaled, X_plot_news_scaled, y_plot_scaled, seq_length=input_steps)
                model.eval()
                with torch.no_grad():
                    X_plot_nonnews_tensor = torch.from_numpy(X_plot_nonnews_seq_scaled).to(device)
                    X_plot_news_tensor = torch.from_numpy(X_plot_news_seq_scaled).to(device)
                    plot_preds_scaled = model(X_plot_nonnews_tensor, X_plot_news_tensor)
                plot_preds_scaled_np = plot_preds_scaled.cpu().numpy().reshape(-1, 1)
                # 역변환 후 평가 (원본 스케일)
                plot_preds_inv = y_scaler.inverse_transform(plot_preds_scaled_np).flatten()
                y_plot_seq_inv = y_scaler.inverse_transform(y_plot_seq_scaled.reshape(-1, 1)).flatten()
                r2_full, mae_full = evaluate(y_plot_seq_inv, plot_preds_inv)
                r2_scaled = r2_score(y_plot_seq_scaled, plot_preds_scaled_np.flatten())
                mae_scaled = mean_absolute_error(y_plot_seq_scaled, plot_preds_scaled_np.flatten())
                logger.info(f"--> Entire Data (Original) R2: {r2_full:.4f}, MAE: {mae_full:.4f}")
                logger.info(f"--> Entire Data (Scaled) R2: {r2_scaled:.4f}, MAE: {mae_scaled:.4f}")

                # 체크포인트 저장: 지정된 R² 임계값 이상일 경우
                for threshold in saved_checkpoints.keys():
                    if r2_full >= threshold and not saved_checkpoints[threshold]:
                        checkpoint_path = f"checkpoint_{threshold:.2f}.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        saved_checkpoints[threshold] = True
                        logger.info(f"Checkpoint saved: {checkpoint_path}")

    # 6) 테스트는 생략

    df_plot_base = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    X_nonnews_plot = df_plot_base[non_news_columns].values.astype(np.float32)
    X_news_plot = df_plot_base[news_columns].values.astype(np.float32)
    y_plot = df_plot_base[target_col].values.astype(np.float32)
    
    X_plot_nonnews_scaled = input_scaler_nonnews.transform(X_nonnews_plot)
    X_plot_news_scaled = input_scaler_news.transform(X_news_plot)
    y_plot_scaled = y_scaler.transform(y_plot.reshape(-1, 1)).flatten()

    if len(X_plot_nonnews_scaled) >= input_steps:
        X_plot_nonnews_seq, X_plot_news_seq, y_plot_seq_scaled = create_sequences(X_plot_nonnews_scaled, X_plot_news_scaled, y_plot_scaled, seq_length=input_steps)
        model.eval()
        with torch.no_grad():
            X_plot_nonnews_tensor = torch.from_numpy(X_plot_nonnews_seq).to(device)
            X_plot_news_tensor = torch.from_numpy(X_plot_news_seq).to(device)
            plot_preds_scaled = model(X_plot_nonnews_tensor, X_plot_news_tensor)
        plot_preds_scaled_np = plot_preds_scaled.cpu().numpy().reshape(-1, 1)
        plot_preds_inv = y_scaler.inverse_transform(plot_preds_scaled_np).flatten()
        y_plot_seq_inv = y_scaler.inverse_transform(y_plot_seq_scaled.reshape(-1, 1)).flatten()

        df_plot_for_plot = df_plot_base.iloc[input_steps:].copy()
        test_dates = df_plot_for_plot[config['date_col']].values

        # 원본 스케일 예측값 플랏
        save_plot(
            df_plot_for_plot,
            y_plot_seq_inv,
            plot_preds_inv,
            date_col=config['date_col'],
            days=730,
            output_path="last365_train_val_transformer.png"
        )
        logger.info("[INFO] Saved original-scale plot (Train+Val).")

        # 스케일된 값 플랏
        plt.figure(figsize=(10,6))
        plt.plot(test_dates, y_plot_seq_scaled, label='Scaled Actual')
        plt.plot(test_dates, plot_preds_scaled_np.flatten(), label='Scaled Prediction')
        plt.title('Last 730 days comparison (Scaled Actual vs. Prediction)')
        plt.xlabel('Date')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.tight_layout()
        plot_dir = "Plot"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        scaled_plot_filename = os.path.join(plot_dir, datetime.now().strftime("plot_transformer_scaled_%Y%m%d_%H%M.png"))
        plt.savefig(scaled_plot_filename)
        logger.info("Scaled plot saved to %s", scaled_plot_filename)
        plt.close()
    else:
        logger.info("Not enough data to create a sequence for plotting (Train+Val).")

    return model


def get_model(input_size=4, input_steps=40, news_dim=640):
    config = basic_config()
    d_model = config['hidden_dim']
    num_layers = config['num_layers']

    model = TransformerForecast(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        input_steps=input_steps,
        news_dim=news_dim,
        nhead=4,
        output_size=1,
        dropout=0.1
    )
    return model




def get_data(file_path):
    config = basic_config()
    input_steps = config['input_steps']
    d_model = config['hidden_dim']   # Transformer의 d_model로 사용
    num_layers = config['num_layers']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    target_col = config['target_col']
    wd = config.get('weight_decay', 0)
    seed = config.get('seed', 42)
    
    # 1) 데이터 로드 및 전처리
    df = load_csv(file_path)
    df.dropna(inplace=True)  # 결측치 제거
    
    df_train, df_val, df_test = split_data(df, config)
    feature_cols = [c for c in df.columns if c not in [config['date_col'], target_col]]
    news_columns = [col for col in feature_cols if "News" in col]
    non_news_columns = [col for col in feature_cols if "News" not in col]

    print(f"Non-News Columns Number: {len(non_news_columns)}\tNews Columns Number: {len(news_columns)}")
    
    # 2) Train, Val 데이터 준비
    X_train_nonnews = df_train[non_news_columns].values.astype(np.float32)
    X_train_news = df_train[news_columns].values.astype(np.float32)
    y_train = df_train[target_col].values.astype(np.float32)
    if not df_val.empty:
        X_val_nonnews = df_val[non_news_columns].values.astype(np.float32)
        X_val_news = df_val[news_columns].values.astype(np.float32)
        y_val = df_val[target_col].values.astype(np.float32)
    else:
        X_val_nonnews, X_val_news, y_val = None, None, None

    # 입력 피처 스케일링
    input_scaler_nonnews = StandardScaler()
    input_scaler_news = StandardScaler()

    X_train_nonnews_scaled = input_scaler_nonnews.fit_transform(X_train_nonnews)
    X_train_news_scaled = input_scaler_news.fit_transform(X_train_news)
    if X_val_nonnews is not None:
        X_val_nonnews_scaled = input_scaler_nonnews.transform(X_val_nonnews)
        X_val_news_scaled = input_scaler_news.transform(X_val_news)
    else:
        X_val_nonnews_scaled, X_val_news_scaled = None, None

    # 타겟 스케일링
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    if y_val is not None:
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    else:
        y_val_scaled = None

    # 3) Transformer 입력 시퀀스 생성 (Train & Val)
    X_train_nonnews_seq, X_train_news_seq, y_train_seq = create_sequences(X_train_nonnews_scaled, X_train_news_scaled, y_train_scaled, seq_length=input_steps)
    if X_val_nonnews_scaled is not None and len(X_val_nonnews_scaled) >= input_steps:
        X_val_nonnews_seq, X_val_news_seq, y_val_seq = create_sequences(X_val_nonnews_scaled, X_val_news_scaled, y_val_scaled, seq_length=input_steps)
    else:
        X_val_nonnews_seq, X_val_news_seq, y_val_seq = None, None, None
    
    return torch.from_numpy(X_train_news_seq)


def get_data_eval(file_path):
    config = basic_config()
    input_steps = config['input_steps']
    d_model = config['hidden_dim']   # Transformer의 d_model로 사용
    num_layers = config['num_layers']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    target_col = config['target_col']
    wd = config.get('weight_decay', 0)
    seed = config.get('seed', 42)
    
    # 1) 데이터 로드 및 전처리
    df = load_csv(file_path)
    df.dropna(inplace=True)  # 결측치 제거
    
    feature_cols = [c for c in df.columns if c not in [config['date_col'], target_col]]
    news_columns = [col for col in feature_cols if "News" in col]
    non_news_columns = [col for col in feature_cols if "News" not in col]

    print(f"Non-News Columns Number: {len(non_news_columns)}\tNews Columns Number: {len(news_columns)}")
    
    # 2) Train, Val 데이터 준비
    X_train_nonnews = df[non_news_columns].values.astype(np.float32)
    X_train_news = df[news_columns].values.astype(np.float32)
    y_train = df[target_col].values.astype(np.float32)

    # 입력 피처 스케일링
    input_scaler_nonnews = StandardScaler()
    input_scaler_news = StandardScaler()

    X_train_nonnews_scaled = input_scaler_nonnews.fit_transform(X_train_nonnews)
    X_train_news_scaled = input_scaler_news.fit_transform(X_train_news)

    # 타겟 스케일링
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    # 3) Transformer 입력 시퀀스 생성 (Train & Val)
    X_train_nonnews_seq, X_train_news_seq, y_train_seq = create_sequences(X_train_nonnews_scaled, X_train_news_scaled, y_train_scaled, seq_length=input_steps)
    
    return torch.from_numpy(X_train_news_seq), df, torch.from_numpy(X_train_nonnews_seq)

if __name__ == "__main__":
    file_path = "/home/kororu/seyoung/gogo/merge/final_shrink_128.csv"

    trained_model = train_main(file_path)
    torch.save(trained_model.state_dict(), "transformer_full_yonhaponly.pt")

    x_train_news_seq, df, x_train_nonnews_seq = get_data_eval(file_path)

    trained_model = get_model()
    trained_model.load_state_dict(torch.load("transformer_full_yonhaponly.pt"))
    trained_model = trained_model.to('cuda')
    mlp_model = trained_model.transformer_encoder.layers[3].mlp
    mlp_model.eval()

    all_results = []

    for num_index in tqdm(range(len(df) - 40 - 1)):
        x_sample = x_train_news_seq[num_index, :, :].to('cuda')  # [40, 640]
        x_sample = x_sample.view(1, -1)  # [1, 25600]
        date_val = df['Date'].iloc[num_index + 40 + 1]

        first_linear = mlp_model[0:2](x_sample)
        first_zero_indices = torch.nonzero(first_linear == 0, as_tuple=True)

        second_linear = mlp_model[0:4](x_sample)
        second_zero_indices = torch.nonzero(second_linear == 0, as_tuple=True)

        final_lst_abs = []
        final_lst_sum = []

        for i in range(5):
            mask = torch.ones_like(x_sample, device=x_sample.device)
            mask[:, i * 128:(i + 1) * 128] = 0
            x_modified = x_sample * mask

            output_1 = mlp_model[0](x_modified)
            output_1[first_zero_indices] = 0

            output_2 = mlp_model[2](output_1)
            output_2[second_zero_indices] = 0

            output_3 = mlp_model[4](output_2)

            final_lst_abs.append(round(torch.sum(torch.abs(output_3)).item(), 2))
            final_lst_sum.append(round(torch.sum(output_3).item(), 2))

        sorted_indices = (np.argsort(final_lst_sum)[::-1] + 1).tolist()

        all_results.append({
            'Date': date_val,
            'Final_Abs_Values': final_lst_abs,
            'Final_Sum_Values': final_lst_sum,
            'Sorted_List': sorted_indices
        })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv('transformer_all_results.csv', index=False)



    
    # first_linear = mlp_model[0:2](x_train_news_seq)
    # first_zero_indices = torch.nonzero(first_linear == 0, as_tuple=True)
    
    # second_linear = mlp_model[0:4](x_train_news_seq)
    # second_zero_indices = torch.nonzero(second_linear == 0, as_tuple=True)

    # third_linear = mlp_model[0:6](x_train_news_seq)
    # third_zero_indices = torch.nonzero(third_linear == 0, as_tuple=True)

    # final_lst_abs = []
    # final_lst_sum = []
    # for i in range(10):
        
    #     mask = torch.zeros_like(x_train_news_seq, device = x_train_news_seq.device)
    #     mask[:, i * 128: (i+1) * 128] = 1
    #     x_modified = x_train_news_seq * mask

    #     output_1 = mlp_model[0](x_modified)
    #     output_1[first_zero_indices] = 0

    #     output_2 = mlp_model[2](output_1)
    #     output_2[second_zero_indices] = 0

    #     output_3 = mlp_model[4](output_2)
    #     output_3[third_zero_indices] = 0

    #     output_4 = mlp_model[6](output_3)

    #     final_lst_abs.append(round(torch.sum(torch.abs(output_4)).item(), 2))
    #     final_lst_sum.append(round(torch.sum(output_4).item(), 2))
    # print("Final Abs Values :", final_lst_abs)
    # print("Sorted List :", np.argsort(final_lst_abs)[::-1] + 1)
