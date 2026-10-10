# -*- coding: utf-8 -*-
"""
PI+GRU 训练脚本（第三版 v3：白箱物理主干 + 残差学习 + 物理约束）

v3 相对 v2 的改动：
1. 数据加载含力传感列 → 逐 trial 预紧力 F0 估计（式4-16 静止段两绳各≈F0）
2. 全局标定：k̂_τ（OLS 斜率）、g0（可辨识 trial 扫描中位数）、s_loop（可辨识 trial 分支差符号）
3. 白箱 Play 状态机（数学上严格等价于式(4-19)/(4-23)）作为物理主干，
   γ_k∈(0,1.2]、γ_b≥0 为可微校准因子
4. PIplusGRUv3：GRU 只学残差 τ̂_res（fc2 零初始化 ⇒ 起点为纯白箱）
5. Soft-Play 阈值按 dz_scale=F0/F0_med 锚定物理死区（式4-19 b∝F0）
6. 复合损失：MSE + λ_t·(λ1·边界 + λ2·死区)，λ_t 从 epoch 3→10 线性 warmup
7. r/β/γ 更新（每 5 epoch × 3 步）：整 trial 序列端到端，含 λ3·耗散约束，
   时序约定与主循环一致（GRU 消费 [0..t-1] 预测 y[t]，物理在 t 取值）
8. v3.1: 残差头 LR 1e-3 余弦衰减至 1e-4（r/β/γ 保持小步慢调）、耐心 30、验证纯 MSE

CPU-only 策略 B 保持：主循环每 5 epoch 用当前 r/β 重算特征并缓存。
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_CSV, OUTPUT_DIR, INPUT_COLS, TARGET_COL, FORCE_COLS,
    WINDOW_SIZE, N_CHANNELS, N_OPS_PER_CHAN, HIDDEN_SIZE, NUM_LAYERS, FC_HIDDEN,
    BETA_INIT, R_INIT_MAX,
    BATCH_SIZE, EPOCHS, SEED, DEVICE,
    DROPOUT, WEIGHT_DECAY, VAL_TRIAL_RATIO,
    FEATURE_REFRESH_EPOCHS, SOFTPLAY_UPDATE_STEPS, GRAD_CLIP_NORM,
    LAMBDA_BOUND, LAMBDA_DEADZONE, LAMBDA_ENERGY,
    PHYS_WARMUP_EPOCHS, PHYS_WARMUP_FULL, LR_V3, LR_V3_MAIN, LR_V3_MIN,
    EARLY_STOP_PATIENCE_V3,
    BEST_MODEL_V3_PATH, FINAL_MODEL_V3_PATH, LOSS_PLOT_V3_PATH,
    ensure_output_dir,
)
from utils import set_seed, load_trials_extended, split_by_trial, plot_training_loss
from physics_branch import (
    calibrate_from_trials, build_phys_list, loss_bound, loss_deadzone, loss_energy,
)
from dataset import MultiTrialPhysDataset
from model import PIplusGRUv3


def lambda_warmup(epoch):
    """λ_t：前 PHYS_WARMUP_EPOCHS 个 epoch 纯 MSE，之后线性升至 PHYS_WARMUP_FULL"""
    if epoch < PHYS_WARMUP_EPOCHS:
        return 0.0
    if epoch >= PHYS_WARMUP_FULL:
        return 1.0
    return (epoch + 1 - PHYS_WARMUP_EPOCHS) / (PHYS_WARMUP_FULL - PHYS_WARMUP_EPOCHS)


def collate_fn(batch):
    """DataLoader collate: (x_window, feat_window, p_seq, p_end, target)"""
    xb = torch.stack([b[0] for b in batch])
    fb = torch.stack([b[1] for b in batch])
    pb = torch.stack([b[2] for b in batch])
    pe = torch.stack([b[3] for b in batch])
    yb = torch.stack([b[4] for b in batch])
    return xb, fb, pb, pe, yb


def train_one_epoch(model, loader, opt, criterion, device, lam_t, main_params):
    """主循环：MSE + λ_t·(λ1·边界 + λ2·死区)

    物理主干 γ detach（γ 只在 r/β/γ 更新阶段训练）；边界线用当前 γ 值。
    预缓存特征前向，GRU 每步反传。
    """
    model.train()
    total, n = 0.0, 0
    for xb, fb, pb, pe, yb in loader:
        xb = xb.to(device)
        fb = fb.to(device)
        pb = pb.to(device)
        pe = pe.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pred, resid = model(xb, fb, pb, pe)      # phys 主干 γ detach
        loss = criterion(pred, yb)
        if lam_t > 0.0:
            # λ1: 预测须落在加载/卸载线之间（式4-19/4-23，经 γ 校准）
            lb = loss_bound(pred, pe[:, 4:5], pe[:, 2:3],
                            model.calib.gamma_k, model.calib.gamma_b)
            # λ2: 死区保持窗内残差→0（直击换向瞬间 MAXE 病灶）
            ld = loss_deadzone(resid, pe[:, 3:4])
            loss = loss + lam_t * (LAMBDA_BOUND * lb + LAMBDA_DEADZONE * ld)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, GRAD_CLIP_NORM)
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


def validate(model, loader, criterion, device):
    """验证：纯 MSE（早停与选型依据，与 v2 口径一致）"""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, fb, pb, pe, yb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            pb = pb.to(device)
            pe = pe.to(device)
            yb = yb.to(device)
            pred, _ = model(xb, fb, pb, pe)
            total += criterion(pred, yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def update_physics_params(model, tr_trials, tr_phys, opt_sp, device, lam_t,
                          s_loop, n_steps=SOFTPLAY_UPDATE_STEPS, max_len=8192):
    """r/β/γ 端到端梯度更新（策略 B + 物理参数，GRU 冻结）

    随机选完整 trial（超长截断到 max_len，从起点截保持 Soft-Play 记忆正确），
    序列前向与主循环时序约定严格一致。
    损失 = MSE + λ_t·(λ1·边界 + λ2·死区 + λ3·耗散)。
    """
    model.eval()  # 关闭 dropout，梯度干净
    criterion = nn.MSELoss()
    sp_params = [model.soft_play.raw_r, model.soft_play.raw_beta,
                 model.calib.raw_gk, model.calib.raw_gb]
    for _ in range(n_steps):
        t_idx = int(np.random.randint(0, len(tr_trials)))
        X_t, y_t, _ = tr_trials[t_idx]
        ph = tr_phys[t_idx]
        T = min(len(X_t), max_len)
        if T < 2:
            continue

        x = torch.tensor(X_t[:T], dtype=torch.float32).unsqueeze(0).to(device)
        p_seq = torch.tensor(ph["p_seq"][:T], dtype=torch.float32).unsqueeze(0).to(device)
        x_eff = torch.tensor(ph["x_eff"][:T], dtype=torch.float32).reshape(T, 1).to(device)
        branch = torch.tensor(ph["branch"][:T], dtype=torch.float32).reshape(T, 1).to(device)
        x_line = torch.tensor(ph["x"][:T], dtype=torch.float32).to(device)
        hold = torch.tensor(ph["hold"][:T], dtype=torch.float32).to(device)
        y_seg = torch.tensor(y_t[:T], dtype=torch.float32).to(device)
        g = ph["g"]

        opt_sp.zero_grad()
        pred, resid = model.forward_sequence(x, p_seq, x_eff, branch, g)
        pred = pred[0, :, 0]     # (T-1,)
        resid = resid[0, :, 0]
        y_tgt = y_seg[1:]        # 目标时刻 t=1..T-1
        xl = x_line[1:]
        hd = hold[1:]

        loss = criterion(pred, y_tgt)
        if lam_t > 0.0:
            lb = loss_bound(pred, xl, g, model.calib.gamma_k, model.calib.gamma_b)
            ld = loss_deadzone(resid, hd)
            le = loss_energy(pred, xl, s_loop)
            loss = loss + lam_t * (LAMBDA_BOUND * lb + LAMBDA_DEADZONE * ld
                                   + LAMBDA_ENERGY * le)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sp_params, GRAD_CLIP_NORM)
        opt_sp.step()


def make_ckpt(model, epoch, val_loss, calib):
    return {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'beta': model.soft_play.beta.item(),
        'r': model.soft_play.r.detach().cpu().numpy(),
        'gamma_k': model.calib.gamma_k.item(),
        'gamma_b': model.calib.gamma_b.item(),
        # 标定字典：test.py 重建物理上下文用
        'calib': {
            'k_tau': float(calib['k_tau']),
            'g0': float(calib['g0']),
            'F0_med': float(calib['F0_med']),
            's_loop': float(calib['s_loop']),
        },
    }


def main(subset_rows=None, epochs=None, batch_size=None,
         refresh_epochs=None, quick_test=False):
    set_seed(SEED)
    device = torch.device(DEVICE)
    ensure_output_dir()

    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    refresh_epochs = refresh_epochs or FEATURE_REFRESH_EPOCHS

    # 1. 按 trial 加载数据（含力传感列）
    print("=" * 60)
    print("PI+GRU v3 训练（白箱物理主干 + 残差学习 + 物理约束）")
    print("=" * 60)
    trials = load_trials_extended(TRAIN_CSV, INPUT_COLS, TARGET_COL, FORCE_COLS)
    if subset_rows:
        kept, cnt = [], 0
        for t in trials:
            kept.append(t)
            cnt += len(t[0])
            if cnt >= subset_rows:
                break
        trials = kept
        print(f"使用子集训练: {len(trials)} 个 trial, 约 {cnt} 行")

    # 2. 全局物理标定（k̂_τ, g0, F0_med, s_loop）
    print("\n白箱物理标定...")
    calib = calibrate_from_trials(trials)
    print(f"  k̂_τ = {calib['k_tau']:.4f}（OLS 斜率）")
    print(f"  g0 = {calib['g0']:.4f} N·m（可辨识 trial {calib['n_ident']}/{len(trials)} "
          f"Play 拟合扫描最优 g 中位数，Play 半间隙）")
    print(f"  F0_med = {calib['F0_med']:.2f} N（逐trial范围 "
          f"[{calib['F0_list'].min():.1f}, {calib['F0_list'].max():.1f}] N）")
    print(f"  s_loop = {calib['s_loop']:+.0f}（可辨识 trial 分桶差中位数 "
          f"{calib['branch_gap']:+.3f}，理论 A=−2g·Δx<0 ⇒ −1）")
    print(f"  自洽性 2g/k_τ vs 式(4-21)换向死区: "
          f"{calib['consistency_2g_over_kt_vs_dzrev']:.3f}")

    # 3. trial 分组切分 + 逐 trial 物理上下文
    tr_trials, val_trials = split_by_trial(trials, val_ratio=VAL_TRIAL_RATIO, seed=SEED)
    tr_phys = build_phys_list(tr_trials, calib)
    val_phys = build_phys_list(val_trials, calib)

    # 4. 构建模型
    model = PIplusGRUv3(
        n_channels=N_CHANNELS, n_ops=N_OPS_PER_CHAN,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_hidden=FC_HIDDEN,
        beta_init=BETA_INIT, r_init_max=R_INIT_MAX,
        dropout=DROPOUT,
    ).to(device)
    main_params = list(model.gru.parameters()) + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())
    sp_params = [model.soft_play.raw_r, model.soft_play.raw_beta,
                 model.calib.raw_gk, model.calib.raw_gb]
    print(f"\n模型参数数: {sum(p.numel() for p in model.parameters())}")
    print(f"初始 γ_k={model.calib.gamma_k.item():.4f}, γ_b={model.calib.gamma_b.item():.4f}"
          f"（标定起点，γ_b≥0 结构性保证耗散非负）")
    print(f"初始 β={model.soft_play.beta.item():.4f}, r 范围="
          f"[{model.soft_play.r.min().item():.4f}, {model.soft_play.r.max().item():.4f}]")
    print(f"LR_main={LR_V3_MAIN}→{LR_V3_MIN}(cosine), LR_phys={LR_V3 * 0.5:.1e}, "
          f"Dropout={DROPOUT}, WeightDecay={WEIGHT_DECAY}, "
          f"EarlyStopPatience={EARLY_STOP_PATIENCE_V3}")
    print(f"λ1={LAMBDA_BOUND}, λ2={LAMBDA_DEADZONE}, λ3={LAMBDA_ENERGY}, "
          f"warmup {PHYS_WARMUP_EPOCHS}→{PHYS_WARMUP_FULL} epoch")

    # 两个优化器：GRU+FC 残差头 / r·β·γ 物理参数（小步慢调）
    # v3.1：残差头 LR 提至 1e-3（与 v2 相同）+ 余弦衰减；物理参数 LR 不变
    opt_main = torch.optim.Adam(main_params, lr=LR_V3_MAIN, weight_decay=WEIGHT_DECAY)
    opt_sp = torch.optim.Adam(sp_params, lr=LR_V3 * 0.5)
    sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_main, T_max=epochs, eta_min=LR_V3_MIN)
    criterion = nn.MSELoss()

    # 5. 多 trial 物理数据集（Soft-Play 特征阈值按 F0 锚定）
    print("\n初始化多trial物理数据集（含 Soft-Play 特征预计算）...")
    t0 = time.time()
    ds_tr = MultiTrialPhysDataset(tr_trials, tr_phys, WINDOW_SIZE, model.soft_play)
    ds_val = MultiTrialPhysDataset(val_trials, val_phys, WINDOW_SIZE, model.soft_play)
    print(f"  特征预计算耗时: {time.time() - t0:.1f}s")
    print(f"  训练集样本数: {len(ds_tr)}")
    print(f"  验证集样本数: {len(ds_val)}")

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # 6. 训练循环（λ warmup + 策略B + 早停）
    print(f"\n开始训练（epochs={epochs}, refresh 每 {refresh_epochs} epoch, "
          f"early_stop patience={EARLY_STOP_PATIENCE_V3}）...")
    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_counter = 0
    t_start = time.time()

    for epoch in range(epochs):
        lam_t = lambda_warmup(epoch)

        # 策略 B：先更新 r/β/γ，再用新参数重算缓存特征（修正 v2 的滞后一拍）
        if epoch > 0 and epoch % refresh_epochs == 0:
            print(f"  [Epoch {epoch}] 更新 r/β/γ + 刷新 Soft-Play 特征 "
                  f"(λ_t={lam_t:.2f})...")
            t_rf = time.time()
            update_physics_params(model, tr_trials, tr_phys, opt_sp, device,
                                  lam_t, calib['s_loop'])
            ds_tr.refresh_features()
            ds_val.refresh_features()
            print(f"    耗时 {time.time() - t_rf:.1f}s, "
                  f"β={model.soft_play.beta.item():.3f}, "
                  f"r 范围=[{model.soft_play.r.min().item():.4f}, "
                  f"{model.soft_play.r.max().item():.4f}], "
                  f"γ_k={model.calib.gamma_k.item():.4f}, "
                  f"γ_b={model.calib.gamma_b.item():.4f}")

        tr_loss = train_one_epoch(model, loader_tr, opt_main, criterion,
                                  device, lam_t, main_params)
        val_loss = validate(model, loader_val, criterion, device)
        sched_main.step()
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            elapsed = time.time() - t_start
            lr_now = opt_main.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs} - train: {tr_loss:.6f}, "
                  f"val: {val_loss:.6f}, λ_t: {lam_t:.2f}, lr: {lr_now:.2e}  "
                  f"[用时 {elapsed:.0f}s]", flush=True)

        # 最佳模型保存 + 早停判断（验证纯 MSE）
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(make_ckpt(model, epoch, val_loss, calib), BEST_MODEL_V3_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE_V3:
                print(f"\n  早停触发：验证损失连续 {EARLY_STOP_PATIENCE_V3} 个 epoch 未改善")
                print(f"  最佳验证损失: {best_val:.6f}")
                break

    total_time = time.time() - t_start
    print(f"\n训练完成！总用时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"最佳验证损失: {best_val:.6f}")
    print(f"模型保存至: {BEST_MODEL_V3_PATH}")

    # 保存最终模型
    torch.save(make_ckpt(model, epoch, val_loss, calib), FINAL_MODEL_V3_PATH)

    # 绘制训练曲线
    plot_training_loss(train_losses, val_losses, LOSS_PLOT_V3_PATH)

    # 打印训练后物理参数
    print("\n训练后物理参数:")
    print(f"  β: {model.soft_play.beta.item():.4f} (初始 {BETA_INIT})")
    print(f"  r 范围: [{model.soft_play.r.min().item():.4f}, "
          f"{model.soft_play.r.max().item():.4f}], "
          f"均值: {model.soft_play.r.mean().item():.4f}")
    print(f"  γ_k: {model.calib.gamma_k.item():.4f} (初始 1.0)")
    print(f"  γ_b: {model.calib.gamma_b.item():.4f} (初始 1.0，≥0 保证耗散非负)")

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="仅使用前 N 行训练（调试用，按 trial 取整）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--refresh", type=int, default=None,
                        help="特征刷新周期（epoch）")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式")
    args = parser.parse_args()

    if args.quick:
        main(subset_rows=50000, epochs=5, batch_size=128, refresh_epochs=1)
    else:
        main(subset_rows=args.subset, epochs=args.epochs,
             batch_size=args.batch, refresh_epochs=args.refresh)
