# -*- coding: utf-8 -*-
"""
PI+GRU v3 测试脚本

功能：
1. 加载 v3 训练权重（含物理标定字典 calib: k̂_τ/g0/F0_med/s_loop）
2. 测试集按 trial 重建物理上下文（修复 v2 把整段 CSV 当单一序列的
   跨 trial 记忆污染 bug）
3. 评估三组模型：v3 主干+残差 / 纯白箱(γ=1) / γ 校准白箱主干
4. 死区保持窗（换向瞬间）单独 RMSE/MAXE 指标
5. 逐 trial 迟滞环面积（耗散）误差
6. 时域对比图、迟滞环图（v3 命名）
7. 支持与 baseline GRU 并列对比
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TEST_CSV, OUTPUT_DIR, INPUT_COLS, TARGET_COL, FORCE_COLS,
    WINDOW_SIZE, N_CHANNELS, N_OPS_PER_CHAN, HIDDEN_SIZE, NUM_LAYERS,
    FC_HIDDEN, BETA_INIT, R_INIT_MAX, BATCH_SIZE, SEED, DEVICE,
    BEST_MODEL_V3_PATH, PREDICTION_PLOT_V3_PATH, HYSTERESIS_PLOT_V3_PATH,
    ensure_output_dir,
)
from utils import set_seed, metrics, load_trials_extended, plot_prediction, plot_hysteresis_loop
from physics_branch import build_phys_list
from dataset import MultiTrialPhysDataset
from model import PIplusGRUv3


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch])
    fb = torch.stack([b[1] for b in batch])
    pb = torch.stack([b[2] for b in batch])
    pe = torch.stack([b[3] for b in batch])
    yb = torch.stack([b[4] for b in batch])
    return xb, fb, pb, pe, yb


def evaluate(model, loader, device):
    """测试集前向，返回逐样本预测/真值/物理上下文"""
    model.eval()
    preds, targets, p_ends = [], [], []
    with torch.no_grad():
        for xb, fb, pb, pe, yb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            pb = pb.to(device)
            pe = pe.to(device)
            pred, _ = model(xb, fb, pb, pe)
            preds.append(pred.cpu())
            targets.append(yb)
            p_ends.append(pe)
    return (torch.cat(preds).numpy().flatten(),
            torch.cat(targets).numpy().flatten(),
            torch.cat(p_ends).numpy())


def report(name, m):
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print("=" * 50)
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")
    return m


def main(model_path=None, compare_baseline=True):
    set_seed(SEED)
    device = torch.device(DEVICE)
    ensure_output_dir()
    model_path = model_path or BEST_MODEL_V3_PATH

    print("=" * 60)
    print("PI+GRU v3 测试评估（物理主干 + 残差）")
    print("=" * 60)

    # 1. 按 trial 加载测试数据（含力传感列；trial 独立 → 记忆无污染）
    trials = load_trials_extended(TEST_CSV, INPUT_COLS, TARGET_COL, FORCE_COLS)

    # 2. 构建模型并加载权重（含物理标定字典）
    model = PIplusGRUv3(
        n_channels=N_CHANNELS, n_ops=N_OPS_PER_CHAN,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_hidden=FC_HIDDEN,
        beta_init=BETA_INIT, r_init_max=R_INIT_MAX,
    ).to(device)

    print(f"\n加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    calib = ckpt['calib']
    gk, gb = float(ckpt['gamma_k']), float(ckpt['gamma_b'])
    print(f"  训练时最佳验证损失 = {ckpt['val_loss']:.6f} (Epoch {ckpt['epoch']+1})")
    print(f"  标定: k̂_τ={calib['k_tau']:.4f}, g0={calib['g0']:.4f}, "
          f"F0_med={calib['F0_med']:.2f}N, s_loop={calib['s_loop']:+.0f}")
    print(f"  训练后 γ_k={gk:.4f}, γ_b={gb:.4f}, "
          f"β={ckpt['beta']:.3f}")

    # 3. 重建物理上下文 + 数据集（Soft-Play 阈值按测试 trial 的 F0 锚定）
    print("\n重建测试集物理上下文（逐 trial F0 → dz_scale 锚定）...")
    phys_list = build_phys_list(trials, calib)
    ds_te = MultiTrialPhysDataset(trials, phys_list, WINDOW_SIZE, model.soft_play)
    loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn)
    print(f"  测试集样本数: {len(ds_te)}")

    # 4. v3 模型评估
    print("\n开始评估...")
    preds, targets, pe = evaluate(model, loader_te, device)
    m_v3 = report("v3（物理主干 + PI+GRU 残差）测试结果", metrics(preds, targets))

    # 5. 纯白箱基线（无任何数据模型，只看物理）
    #    γ=1 版: y_phys = x_eff + branch·g（Play 状态机输出，即式4-19/4-23）
    wb_raw = pe[:, 0] + pe[:, 1] * pe[:, 2]
    #    γ 校准版: τ_l^phys = γ_k·x_eff + γ_b·g·branch（训练微调后的主干）
    wb_cal = gk * pe[:, 0] + gb * pe[:, 1] * pe[:, 2]
    m_wb = report("纯白箱基线（γ=1，式4-19/4-23 状态机）", metrics(wb_raw, targets))
    m_wbc = report("γ 校准白箱主干（无残差）", metrics(wb_cal, targets))

    # 6. 死区保持窗指标（换向瞬间，v2 的 MAXE 病灶）
    hold_mask = pe[:, 3] > 0.5
    print(f"\n{'=' * 50}")
    print(f"死区保持窗指标（hold 占比 {hold_mask.mean()*100:.1f}%）")
    print("=" * 50)
    if hold_mask.sum() > 0:
        for name, p in [("v3", preds), ("纯白箱γ=1", wb_raw), ("γ校准白箱", wb_cal)]:
            mh = metrics(p[hold_mask], targets[hold_mask])
            print(f"  {name}: RMSE={mh['RMSE']:.4f}, MAXE={mh['MAXE']:.4f}")
    else:
        print("  测试集无死区保持样本")

    # 7. 逐 trial 迟滞环面积（耗散）误差
    print(f"\n{'=' * 50}")
    print("逐 trial 迟滞环面积 A（耗散，真值 vs v3 预测）")
    print("=" * 50)
    trial_ids = np.array([t_idx for t_idx, _ in ds_te.window_index])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    errs = []
    for ti in range(len(trials)):
        msk = trial_ids == ti
        if msk.sum() < 100:
            continue
        xl = pe[msk, 4]
        A_true = trapz(targets[msk], xl)
        A_pred = trapz(preds[msk], xl)
        if abs(A_true) > 1e-8:
            rel = (A_pred - A_true) / abs(A_true) * 100
            errs.append(rel)
            print(f"  Trial {ti}: A_true={A_true:+.3f}, A_pred={A_pred:+.3f} "
                  f"({rel:+.1f}%)")
    if errs:
        print(f"  环面积相对误差: 均值 {np.mean(np.abs(errs)):.1f}%, "
              f"最大 {np.max(np.abs(errs)):.1f}%")

    # 8. 绘图
    print("\n生成可视化...")
    n_plot = min(5000, len(preds))
    plot_prediction(
        preds[:n_plot], targets[:n_plot],
        save_path=PREDICTION_PLOT_V3_PATH,
        title=f"PI+GRU v3 预测 (RMSE={m_v3['RMSE']:.4f}, R²={m_v3['R2']:.4f})",
    )
    tau_m = pe[:n_plot, 4] / calib['k_tau']   # x_line=k_τ·τ_m → τ_m
    plot_hysteresis_loop(
        preds[:n_plot], targets[:n_plot], tau_m,
        save_path=HYSTERESIS_PLOT_V3_PATH,
        title=f"v3 迟滞环对比 (γ_k={gk:.2f}, γ_b={gb:.2f})",
    )

    # 9. 可选：与 baseline GRU 对比（逐 trial 独立滑窗，无跨 trial 记忆污染）
    if compare_baseline:
        baseline_path = os.path.join(OUTPUT_DIR, "baseline_gru.pth")
        if os.path.exists(baseline_path):
            print("\n对比 baseline GRU...")
            from baseline_gru import CableGRU, MultiTrialCableDataset
            base_model = CableGRU(input_size=3, hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS).to(device)
            base_model.load_state_dict(
                torch.load(baseline_path, map_location=device, weights_only=False)
            )
            # 传二元组 trial 列表（baseline 接口约定）
            ds_base = MultiTrialCableDataset(
                [(X, y) for X, y, _ in trials], WINDOW_SIZE)
            loader_base = DataLoader(ds_base, batch_size=BATCH_SIZE, shuffle=False)

            base_model.eval()
            base_preds, base_targets = [], []
            with torch.no_grad():
                for xb, yb in loader_base:
                    xb = xb.to(device)
                    p = base_model(xb).cpu().numpy().flatten()
                    base_preds.extend(p)
                    base_targets.extend(yb.numpy().flatten())
            m_base = report("Baseline GRU 测试结果（对比）",
                            metrics(np.array(base_preds), np.array(base_targets)))

            print("\n" + "-" * 50)
            print("v3 相对 Baseline GRU 的改进")
            print("-" * 50)
            for k in m_v3:
                if k == "R2":
                    improve = m_v3[k] - m_base[k]
                    print(f"  {k}: {m_base[k]:.6f} → {m_v3[k]:.6f}  "
                          f"(提升 {improve:+.6f})")
                else:
                    improve = (m_base[k] - m_v3[k]) / m_base[k] * 100
                    print(f"  {k}: {m_base[k]:.6f} → {m_v3[k]:.6f}  "
                          f"(降低 {improve:.2f}%)")
        else:
            print("\n（未找到 baseline_gru.pth，跳过对比。请先运行 baseline_gru.py）")

    print("\n测试完成！")
    return m_v3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径（默认 best_pi_gru_v3.pth）")
    parser.add_argument("--no-baseline", action="store_true",
                        help="跳过与 baseline 对比")
    args = parser.parse_args()
    main(model_path=args.model, compare_baseline=not args.no_baseline)
