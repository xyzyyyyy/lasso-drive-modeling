# -*- coding: utf-8 -*-
"""
公平 GRU 基线模型（第二版：trial分组切分 + 正则化 + 早停）
论文对照实验：使用与 PI+GRU 相同的输入列 [τ_m, q_m, q̇_m]、相同目标、相同结构（hidden=64, layers=2）
仅缺少 Soft-Play 迟滞特征提取层，用于证明 Soft-Play 的增益

第二版改动：与 PI+GRU 同步采用 trial 分组切分 + Dropout + 早停
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_CSV, TEST_CSV, OUTPUT_DIR, INPUT_COLS, TARGET_COL,
    WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    BATCH_SIZE, LR, EPOCHS, SEED, DEVICE,
    DROPOUT, WEIGHT_DECAY, EARLY_STOP_PATIENCE, VAL_TRIAL_RATIO,
    BASELINE_MODEL_PATH, BASELINE_PLOT_PATH,
)
from utils import set_seed, metrics, load_trials, split_by_trial, plot_prediction


# ============================================================
# Dataset
# ============================================================
class CableDataset(Dataset):
    """纯 GRU 数据集：滑窗切片，无迟滞特征（单 trial 内）"""
    def __init__(self, X, y, window_size):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = window_size
        self.n_samples = len(X) - window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.w]              # (w, 3)
        target = self.y[idx + self.w]                  # 标量
        return x_seq, target.unsqueeze(-1)


class MultiTrialCableDataset(Dataset):
    """多 trial 纯 GRU 数据集：滑窗仅在同一 trial 内"""
    def __init__(self, trials, window_size):
        self.trials = trials
        self.w = window_size
        self.window_index = []
        for t_idx, (X_t, y_t) in enumerate(trials):
            n_win = max(0, len(X_t) - window_size)
            for s in range(n_win):
                self.window_index.append((t_idx, s))
        self.n_samples = len(self.window_index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t_idx, start = self.window_index[idx]
        X_t, y_t = self.trials[t_idx]
        x_seq = torch.as_tensor(X_t[start:start + self.w], dtype=torch.float32)
        tgt = torch.as_tensor(y_t[start + self.w], dtype=torch.float32).unsqueeze(-1)
        return x_seq, tgt


# ============================================================
# Model
# ============================================================
class CableGRU(nn.Module):
    """纯 GRU 基线：input=[τ_m, q_m, q̇_m] 3 通道（含 Dropout）"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += criterion(pred, yb).item() * len(xb)
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy().flatten()
            preds.extend(p)
            targets.extend(yb.numpy().flatten())
    return np.array(preds), np.array(targets)


# ============================================================
# Main
# ============================================================
def main(subset_rows=None, epochs=None):
    set_seed(SEED)
    device = torch.device(DEVICE)
    epochs = epochs or EPOCHS

    # 1. 按 trial 加载
    trials = load_trials(TRAIN_CSV, INPUT_COLS, TARGET_COL)
    if subset_rows:
        kept, cnt = [], 0
        for X_t, y_t in trials:
            kept.append((X_t, y_t))
            cnt += len(X_t)
            if cnt >= subset_rows:
                break
        trials = kept
        print(f"使用子集训练: {len(trials)} 个 trial, 约 {cnt} 行")

    # 2. 按 trial 分组切分
    tr_trials, val_trials = split_by_trial(trials, val_ratio=VAL_TRIAL_RATIO, seed=SEED)

    # 3. Dataset & Loader（多 trial 版本）
    ds_tr = MultiTrialCableDataset(tr_trials, WINDOW_SIZE)
    ds_val = MultiTrialCableDataset(val_trials, WINDOW_SIZE)
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 模型（含 Dropout + 权重衰减）
    model = CableGRU(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                     dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # 5. 训练（含早停）
    print(f"\n开始训练 baseline GRU v2 (epochs={epochs}, dropout={DROPOUT}, "
          f"wd={WEIGHT_DECAY}, patience={EARLY_STOP_PATIENCE})...", flush=True)
    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, loader_tr, opt, criterion, device)
        val_loss = validate(model, loader_val, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - train: {tr_loss:.6f}, val: {val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BASELINE_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  早停触发：Epoch {epoch+1}")
                break

    print(f"训练完成, 最佳验证损失: {best_val:.6f}")
    print(f"模型保存至: {BASELINE_MODEL_PATH}")

    # 6. 测试集评估
    print("\n加载测试集评估...")
    test_trials = load_trials(TEST_CSV, INPUT_COLS, TARGET_COL)
    ds_te = MultiTrialCableDataset(test_trials, WINDOW_SIZE)
    loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    preds, targets = evaluate(model, loader_te, device)
    m = metrics(preds, targets)
    print("\n========== Baseline GRU v2 测试结果 ==========")
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")

    # 7. 绘图
    plot_prediction(preds[:5000], targets[:5000], save_path=BASELINE_PLOT_PATH,
                    title=f"Baseline GRU v2 预测 (RMSE={m['RMSE']:.4f})")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="仅使用前 N 行训练（调试用，按 trial 取整）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖默认 epoch 数")
    args = parser.parse_args()
    main(subset_rows=args.subset, epochs=args.epochs)
