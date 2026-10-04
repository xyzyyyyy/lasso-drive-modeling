# -*- coding: utf-8 -*-
"""
PI+GRU 训练脚本（第二版：trial分组切分 + 正则化 + 早停）

改进点（相对第一版）：
1. 按 trial 分组切分训练/验证集（替代时序切分），防止同trial泄漏
2. 段边界处不生成伪样本（滑窗仅在同一trial内）
3. 加 Dropout + 权重衰减抑制过拟合
4. 早停（patience=10）在过拟合拐点及时刹车

CPU-only + 1.4M 样本 + Soft-Play 串行循环 = 性能瓶颈。采用：
- 主循环：每 5 epoch 用当前 r/β 重算特征并缓存（torch.no_grad()），GRU 每步反传
- r/β 更新：每 5 epoch 在小批量上重算带 grad 的特征做几步梯度更新
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
    TRAIN_CSV, OUTPUT_DIR, INPUT_COLS, TARGET_COL,
    WINDOW_SIZE, N_CHANNELS, N_OPS_PER_CHAN, HIDDEN_SIZE, NUM_LAYERS, FC_HIDDEN,
    BETA_INIT, R_INIT_MAX, EPS_LOG,
    BATCH_SIZE, LR, EPOCHS, SEED, DEVICE,
    DROPOUT, WEIGHT_DECAY, EARLY_STOP_PATIENCE, VAL_TRIAL_RATIO,
    FEATURE_REFRESH_EPOCHS, SOFTPLAY_UPDATE_STEPS, SOFTPLAY_UPDATE_BATCH,
    GRAD_CLIP_NORM, BEST_MODEL_PATH, FINAL_MODEL_PATH, LOSS_PLOT_PATH,
    ensure_output_dir,
)
from utils import set_seed, metrics, load_trials, split_by_trial, plot_training_loss
from soft_play import SoftPlay
from dataset import MultiTrialHysteresisDataset
from model import PIplusGRU


def train_one_epoch(model, loader, opt, criterion, device):
    """训练一个 epoch（GRU 参数更新，特征已预缓存）"""
    model.train()
    total = 0.0
    n = 0
    for xb, fb, yb in loader:
        xb = xb.to(device)
        fb = fb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        # 用预缓存的特征前向（不经过 soft_play，断开 r/β 反传图）
        pred = model(xb, fb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, fb, yb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            yb = yb.to(device)
            pred = model(xb, fb)
            total += criterion(pred, yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def update_softplay_params(model, tr_trials, opt_sp, criterion, device, n_steps=3):
    """对 r/β 做几步端到端梯度更新
    从训练 trial 中随机选一个完整 trial，重算带 grad 的 Soft-Play 特征
    用 forward_sequence 保留 Soft-Play 记忆连续性
    GRU 参数在此期间冻结
    """
    model.train()
    for _ in range(n_steps):
        # 随机选一个完整 trial
        t_idx = np.random.randint(0, len(tr_trials))
        X_t, y_t = tr_trials[t_idx]
        # 限制 trial 长度避免内存爆炸（最多取连续 4096 点）
        if len(X_t) > 4096:
            start = np.random.randint(0, len(X_t) - 4096)
            X_t = X_t[start:start + 4096]
            y_t = y_t[start:start + 4096]

        x_seg = torch.tensor(X_t, dtype=torch.float32).unsqueeze(0).to(device)
        y_seg = torch.tensor(y_t, dtype=torch.float32).to(device)

        opt_sp.zero_grad()
        # 端到端序列前向（特征现场算，保留 r/β 反传图）
        pred = model.forward_sequence(x_seg)  # (1, T, 1)
        pred = pred.squeeze(-1).squeeze(0)    # (T,)
        loss = criterion(pred, y_seg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.soft_play.raw_r, model.soft_play.raw_beta], GRAD_CLIP_NORM
        )
        opt_sp.step()


def collate_fn(batch):
    """DataLoader collate: (x_window, feat_window, target)"""
    xb = torch.stack([b[0] for b in batch])
    fb = torch.stack([b[1] for b in batch])
    yb = torch.stack([b[2] for b in batch])
    return xb, fb, yb


def main(subset_rows=None, epochs=None, batch_size=None,
         refresh_epochs=None, quick_test=False):
    set_seed(SEED)
    device = torch.device(DEVICE)
    ensure_output_dir()

    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    refresh_epochs = refresh_epochs or FEATURE_REFRESH_EPOCHS

    # 1. 按 trial 加载数据
    print("=" * 60)
    print("PI+GRU 训练（第二版：trial分组 + 正则化 + 早停）")
    print("=" * 60)
    trials = load_trials(TRAIN_CSV, INPUT_COLS, TARGET_COL)
    if subset_rows:
        # 调试用：只保留前几个 trial
        kept, cnt = [], 0
        for X_t, y_t in trials:
            kept.append((X_t, y_t))
            cnt += len(X_t)
            if cnt >= subset_rows:
                break
        trials = kept
        print(f"使用子集训练: {len(trials)} 个 trial, 约 {cnt} 行")

    # 2. 按 trial 分组切分训练/验证集（防止泄漏）
    tr_trials, val_trials = split_by_trial(trials, val_ratio=VAL_TRIAL_RATIO, seed=SEED)

    # 3. 构建模型（含 Dropout）
    model = PIplusGRU(
        n_channels=N_CHANNELS, n_ops=N_OPS_PER_CHAN,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_hidden=FC_HIDDEN,
        beta_init=BETA_INIT, r_init_max=R_INIT_MAX,
        dropout=DROPOUT,
    ).to(device)
    print(f"\n模型参数数: {sum(p.numel() for p in model.parameters())}")
    print(f"初始 β = {model.soft_play.beta.item():.4f}")
    print(f"初始 r 范围: [{model.soft_play.r.min().item():.4f}, "
          f"{model.soft_play.r.max().item():.4f}]")
    print(f"Dropout = {DROPOUT}, Weight Decay = {WEIGHT_DECAY}, "
          f"Early Stop Patience = {EARLY_STOP_PATIENCE}")

    # 两个优化器：GRU+FC 用主 lr + 权重衰减，r/β 用较小 lr（不加衰减）
    opt_main = torch.optim.Adam(
        list(model.gru.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    opt_sp = torch.optim.Adam(
        [model.soft_play.raw_r, model.soft_play.raw_beta],
        lr=LR * 0.5,  # r/β 更新慢一些更稳定
    )
    criterion = nn.MSELoss()

    # 4. 初始化多 trial 数据集（含 Soft-Play 特征预计算）
    print("\n初始化多trial数据集（含 Soft-Play 特征预计算）...")
    t0 = time.time()
    ds_tr = MultiTrialHysteresisDataset(tr_trials, WINDOW_SIZE, model.soft_play)
    ds_val = MultiTrialHysteresisDataset(val_trials, WINDOW_SIZE, model.soft_play)
    print(f"  特征预计算耗时: {time.time() - t0:.1f}s")
    print(f"  训练集样本数: {len(ds_tr)}")
    print(f"  验证集样本数: {len(ds_val)}")

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # 5. 训练循环（含早停）
    print(f"\n开始训练（epochs={epochs}, refresh 每 {refresh_epochs} epoch, "
          f"early_stop patience={EARLY_STOP_PATIENCE}）...")
    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_counter = 0   # 早停计数器
    t_start = time.time()

    for epoch in range(epochs):
        # 策略 B: 每 K epoch 用当前 r/β 重算特征并缓存
        if epoch > 0 and epoch % refresh_epochs == 0:
            print(f"  [Epoch {epoch}] 刷新 Soft-Play 特征...")
            t_rf = time.time()
            ds_tr.refresh_features()
            ds_val.refresh_features()
            print(f"    特征重算耗时: {time.time() - t_rf:.1f}s, "
                  f"β={model.soft_play.beta.item():.3f}, "
                  f"r 范围=[{model.soft_play.r.min().item():.4f}, "
                  f"{model.soft_play.r.max().item():.4f}]")

            # 端到端更新 r/β
            for sp_step in range(SOFTPLAY_UPDATE_STEPS):
                update_softplay_params(model, tr_trials, opt_sp, criterion, device, n_steps=1)

        # 主循环：训练 GRU+FC（特征已缓存）
        tr_loss = train_one_epoch(model, loader_tr, opt_main, criterion, device)
        val_loss = validate(model, loader_val, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1}/{epochs} - train: {tr_loss:.6f}, "
                  f"val: {val_loss:.6f}  [用时 {elapsed:.0f}s]", flush=True)

        # 最佳模型保存 + 早停判断
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'beta': model.soft_play.beta.item(),
                'r': model.soft_play.r.detach().cpu().numpy(),
            }, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  早停触发：验证损失连续 {EARLY_STOP_PATIENCE} 个 epoch 未改善")
                print(f"  最佳验证损失: {best_val:.6f} (Epoch {epoch+1-EARLY_STOP_PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"\n训练完成！总用时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"最佳验证损失: {best_val:.6f}")
    print(f"模型保存至: {BEST_MODEL_PATH}")

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'beta': model.soft_play.beta.item(),
        'r': model.soft_play.r.detach().cpu().numpy(),
    }, FINAL_MODEL_PATH)

    # 绘制训练曲线
    plot_training_loss(train_losses, val_losses, LOSS_PLOT_PATH)

    # 打印训练后参数
    print(f"\n训练后 Soft-Play 参数:")
    print(f"  β: {model.soft_play.beta.item():.4f} (初始 {BETA_INIT})")
    print(f"  r 范围: [{model.soft_play.r.min().item():.4f}, "
          f"{model.soft_play.r.max().item():.4f}]")
    print(f"  r 均值: {model.soft_play.r.mean().item():.4f}")

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
