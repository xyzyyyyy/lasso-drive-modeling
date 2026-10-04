# -*- coding: utf-8 -*-
"""
PI+GRU 测试脚本

功能：
1. 加载训练好的模型
2. 在测试集上评估 RMSE / MAXE / R²
3. 绘制时域预测对比图和迟滞环对比图
4. 支持与 baseline GRU 并列对比
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TEST_CSV, OUTPUT_DIR, INPUT_COLS, TARGET_COL,
    WINDOW_SIZE, N_CHANNELS, N_OPS_PER_CHAN, HIDDEN_SIZE, NUM_LAYERS,
    FC_HIDDEN, BETA_INIT, R_INIT_MAX, BATCH_SIZE, SEED, DEVICE,
    BEST_MODEL_PATH, PREDICTION_PLOT_PATH, HYSTERESIS_PLOT_PATH,
    ensure_output_dir,
)
from utils import set_seed, metrics, load_raw, plot_prediction, plot_hysteresis_loop
from soft_play import SoftPlay
from dataset import HysteresisDataset
from model import PIplusGRU


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch])
    fb = torch.stack([b[1] for b in batch])
    yb = torch.stack([b[2] for b in batch])
    return xb, fb, yb


def evaluate(model, loader, device):
    """在测试集上预测，返回 preds, targets"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, fb, yb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            pred = model(xb, fb).cpu().numpy().flatten()
            preds.extend(pred)
            targets.extend(yb.numpy().flatten())
    return np.array(preds), np.array(targets)


def main(model_path=None, subset_rows=None, compare_baseline=True):
    set_seed(SEED)
    device = torch.device(DEVICE)
    ensure_output_dir()
    model_path = model_path or BEST_MODEL_PATH

    print("=" * 60)
    print("PI+GRU 测试评估")
    print("=" * 60)

    # 1. 加载测试数据
    _, X_te, y_te = load_raw(TEST_CSV, INPUT_COLS, TARGET_COL)
    if subset_rows:
        X_te, y_te = X_te[:subset_rows], y_te[:subset_rows]
        print(f"使用子集测试: {subset_rows} 行")

    # 2. 构建模型并加载权重
    model = PIplusGRU(
        n_channels=N_CHANNELS, n_ops=N_OPS_PER_CHAN,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_hidden=FC_HIDDEN,
        beta_init=BETA_INIT, r_init_max=R_INIT_MAX,
    ).to(device)

    print(f"\n加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'beta' in ckpt:
        print(f"  训练时 β = {ckpt['beta']:.4f}")
    if 'r' in ckpt:
        r = ckpt['r']
        print(f"  训练时 r 范围 = [{r.min():.4f}, {r.max():.4f}], 均值 = {r.mean():.4f}")
    if 'val_loss' in ckpt:
        print(f"  训练时最佳验证损失 = {ckpt['val_loss']:.6f}")

    # 3. 用训练好的 r/β 重新预计算测试集特征
    print("\n预计算测试集 Soft-Play 特征...")
    ds_te = HysteresisDataset(X_te, y_te, WINDOW_SIZE, model.soft_play)
    loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn)
    print(f"  测试集样本数: {len(ds_te)}")

    # 4. 评估
    print("\n开始评估...")
    preds, targets = evaluate(model, loader_te, device)
    m = metrics(preds, targets)

    print("\n" + "=" * 50)
    print("PI+GRU 测试结果")
    print("=" * 50)
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")

    # 5. 绘图
    print("\n生成可视化...")
    # 时域预测对比（前 5000 个样本清晰可读）
    n_plot = min(5000, len(preds))
    plot_prediction(
        preds[:n_plot], targets[:n_plot],
        save_path=PREDICTION_PLOT_PATH,
        title=f"PI+GRU 预测 (RMSE={m['RMSE']:.4f}, R²={m['R2']:.4f})",
    )

    # 迟滞环（输入 τ_m vs 输出 τ_l）
    # 取与预测对齐的 τ_m（去掉前 WINDOW_SIZE 个）
    tau_m = X_te[WINDOW_SIZE:WINDOW_SIZE + len(preds), 0]  # τ_m 是第 0 个输入列
    plot_hysteresis_loop(
        preds[:n_plot], targets[:n_plot], tau_m[:n_plot],
        save_path=HYSTERESIS_PLOT_PATH,
        title=f"PI+GRU 迟滞环对比 (β={model.soft_play.beta.item():.2f})",
    )

    # 6. 可选：与 baseline GRU 对比
    if compare_baseline:
        baseline_path = os.path.join(OUTPUT_DIR, "baseline_gru.pth")
        if os.path.exists(baseline_path):
            print("\n对比 baseline GRU...")
            from baseline_gru import CableGRU, CableDataset
            baseline_model = CableGRU(input_size=3, hidden_size=HIDDEN_SIZE,
                                       num_layers=NUM_LAYERS).to(device)
            baseline_model.load_state_dict(
                torch.load(baseline_path, map_location=device, weights_only=False)
            )
            ds_base = CableDataset(X_te, y_te, WINDOW_SIZE)
            loader_base = DataLoader(ds_base, batch_size=BATCH_SIZE, shuffle=False)

            baseline_model.eval()
            base_preds, base_targets = [], []
            with torch.no_grad():
                for xb, yb in loader_base:
                    xb = xb.to(device)
                    p = baseline_model(xb).cpu().numpy().flatten()
                    base_preds.extend(p)
                    base_targets.extend(yb.numpy().flatten())
            base_preds = np.array(base_preds)
            base_targets = np.array(base_targets)
            m_base = metrics(base_preds, base_targets)

            print("\n" + "=" * 50)
            print("Baseline GRU 测试结果（对比）")
            print("=" * 50)
            for k, v in m_base.items():
                print(f"  {k}: {v:.6f}")

            # 改进幅度
            print("\n" + "-" * 50)
            print("PI+GRU 相对 Baseline GRU 的改进")
            print("-" * 50)
            for k in m:
                if k == "R2":
                    improve = m[k] - m_base[k]
                    print(f"  {k}: {m_base[k]:.6f} → {m[k]:.6f}  "
                          f"(提升 {improve:+.6f})")
                else:
                    improve = (m_base[k] - m[k]) / m_base[k] * 100
                    print(f"  {k}: {m_base[k]:.6f} → {m[k]:.6f}  "
                          f"(降低 {improve:.2f}%)")
        else:
            print("\n（未找到 baseline_gru.pth，跳过对比。请先运行 baseline_gru.py）")

    print("\n测试完成！")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径（默认 best_pi_gru.pth）")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--no-baseline", action="store_true",
                        help="跳过与 baseline 对比")
    args = parser.parse_args()
    main(model_path=args.model, subset_rows=args.subset,
         compare_baseline=not args.no_baseline)
