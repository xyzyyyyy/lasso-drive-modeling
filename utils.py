# -*- coding: utf-8 -*-
"""
工具函数：指标计算、绘图、种子设置、数据加载、trial划分
"""
import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(preds, targets):
    """计算 RMSE / MAXE / R² 三项指标
    preds, targets: 1D numpy array 或 1D tensor
    """
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    err = preds - targets
    rmse = float(np.sqrt(np.mean(err ** 2)))
    maxe = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"RMSE": rmse, "MAXE": maxe, "R2": r2}


def plot_training_loss(losses, val_losses, save_path):
    """绘制训练/验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='训练损失', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("训练损失曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def plot_prediction(preds, targets, time=None, save_path=None, title="预测 vs 真值"):
    """绘制时域预测对比图"""
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    if time is None:
        time = np.arange(len(preds))

    plt.figure(figsize=(14, 5))
    plt.plot(time, targets, label='真值 τ_l', linewidth=2, alpha=0.8)
    plt.plot(time, preds, label='预测 τ̂_l', linestyle='--', linewidth=1.5, alpha=0.9)
    plt.xlabel("时间 (s)")
    plt.ylabel("力矩 (N·m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"预测对比图已保存: {save_path}")
    plt.close()


def plot_hysteresis_loop(preds, targets, tau_m=None, save_path=None, title="迟滞环对比"):
    """绘制迟滞环（输入 τ_m vs 输出 τ_l）
    上：真值迟滞环；下：预测迟滞环
    """
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    if tau_m is None:
        print("警告: 未提供 τ_m，无法绘制迟滞环")
        return

    tau_m = np.asarray(tau_m).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(tau_m, targets, 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_title("真值迟滞环")
    axes[0].set_xlabel("输入力矩 τ_m (N·m)")
    axes[0].set_ylabel("输出力矩 τ_l (N·m)")
    axes[0].grid(True)

    axes[1].plot(tau_m, preds, 'r-', linewidth=1.5, alpha=0.7)
    axes[1].set_title("预测迟滞环")
    axes[1].set_xlabel("输入力矩 τ_m (N·m)")
    axes[1].set_ylabel("预测力矩 τ̂_l (N·m)")
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"迟滞环图已保存: {save_path}")
    plt.close()


# ============================================================
# 数据加载
# ============================================================

def load_raw(csv_path, input_cols, target_col, dtype=np.float32):
    """加载 CSV 原始数据（单一时序，不含 trial 检测）
    返回: time (T,), X (T, n_channels), y (T,)
    """
    import pandas as pd
    print(f"加载 CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=dtype)
    time = df.iloc[:, 0].values.astype(dtype)
    X = df.iloc[:, input_cols].values.astype(dtype)
    y = df.iloc[:, target_col].values.astype(dtype)
    print(f"  数据形状: X={X.shape}, y={y.shape}, 时长={time[-1]:.2f}s")
    return time, X, y


def load_trials(csv_path, input_cols, target_col, dtype=np.float32):
    """按 trial 检测并加载 CSV 数据

    返回: trials: List[(X_trial, y_trial)]，每个元素是一段独立实验
    trial 检测依据：Time 列单调性，diff(Time) < 0 即段边界
    """
    import pandas as pd
    print(f"加载 CSV (trial模式): {csv_path}")
    df = pd.read_csv(csv_path, dtype=dtype)
    time_col = df.iloc[:, 0].values.astype(dtype)
    X_all = df.iloc[:, input_cols].values.astype(dtype)
    y_all = df.iloc[:, target_col].values.astype(dtype)

    # 检测 trial 边界
    diffs = np.diff(time_col)
    boundary_idx = np.where(diffs < 0)[0]

    starts = [0] + (boundary_idx + 1).tolist()
    ends = boundary_idx.tolist() + [len(time_col) - 1]

    trials = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        X_trial = X_all[s:e+1]
        y_trial = y_all[s:e+1]
        trials.append((X_trial, y_trial))
        if i < 5 or i >= len(trials) - 2:
            print(f"  Trial {i}: rows {s}-{e}, length={e-s+1}")

    print(f"  共检测到 {len(trials)} 个独立 trial")
    return trials


# ============================================================
# 数据切分
# ============================================================

def split_temporal(X, y, split_ratio=0.8, val_ratio=0.1):
    """按时序切分（旧版本，用于简单对比）：train / val / test_internal
    返回: (X_tr, y_tr), (X_val, y_val), (X_te, y_te)
    """
    n = len(X)
    n_train_val = int(n * split_ratio)
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train_val], y[n_train:n_train_val]
    X_te, y_te = X[n_train_val:], y[n_train_val:]
    print(f"时序切分: 训练={n_train}, 验证={n_val}, 内部测试={n - n_train_val}")
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def split_by_trial(trials, val_ratio=0.2, seed=42):
    """按 trial 分组切分训练/验证集（核心改动）

    保证同一 trial 的所有样本不跨集合，防止数据泄漏。

    Args:
        trials: List[(X, y)]
        val_ratio: 验证集 trial 占比
        seed: 随机种子

    Returns:
        (tr_trials, val_trials)
    """
    rng = random.Random(seed)
    shuffled = list(trials)
    rng.shuffle(shuffled)

    n_val_trials = max(1, int(len(trials) * val_ratio))
    val_trials = shuffled[:n_val_trials]
    tr_trials = shuffled[n_val_trials:]

    print(f"按 trial 分组切分: 训练 trial={len(tr_trials)}, 验证 trial={len(val_trials)}")
    print(f"  训练样本总量: {sum(len(x) for x, _ in tr_trials)}")
    print(f"  验证样本总量: {sum(len(x) for x, _ in val_trials)}")
    return tr_trials, val_trials


def trials_to_arrays(trials):
    """把 trial 列表合并为整段数组（保留各段独立 X/y）
    注意：不跨越段边界——不同 trial 之间是独立段
    返回: (X_flat, y_flat, boundaries)
    """
    Xs = np.concatenate([x for x, _ in trials], axis=0)
    ys = np.concatenate([y for _, y in trials], axis=0)
    # boundaries: 每个 trial 的起始索引
    boundaries = [0]
    for x, _ in trials:
        boundaries.append(boundaries[-1] + len(x))
    return Xs, ys, boundaries


# ============================================================
# 早期版本兼容接口（保留，旧代码可继续用）
# ============================================================
def load_and_split(csv_path, input_cols, target_col, split_ratio=0.8, val_ratio=0.1):
    """加载 CSV + 时序切分（旧版兼容）"""
    _, X, y = load_raw(csv_path, input_cols, target_col)
    return split_temporal(X, y, split_ratio, val_ratio)
