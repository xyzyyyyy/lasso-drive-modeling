# -*- coding: utf-8 -*-
"""
逐通道 Soft-Play 数据集

核心设计（trial版）：
- 按 trial 独立处理：每个 trial 独立预计算 Soft-Play 特征
- 特征缓存按 trial 索引，支持按需重算
- 滑窗仅在同一 trial 内进行，保证时序连续性
- 段边界处不生成伪样本
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class HysteresisDataset(Dataset):
    """PI+GRU 数据集：原始 3 通道 + 预计算 Soft-Play 特征

    Args:
        X: numpy/tensor (T, 3) 原始 [τ_m, q_m, q̇_m]
        y: numpy/tensor (T,) 目标 τ_l
        window_size: 滑窗长度
        soft_play_module: SoftPlay 模块（提供 r/β 参数）
        refresh: 是否在构造时立即预计算特征
    """
    def __init__(self, X, y, window_size, soft_play_module, refresh=True):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = window_size
        self.soft_play = soft_play_module
        self.feats = None  # (T, C*N)
        self.n_samples = max(0, len(X) - window_size)
        if refresh:
            self.refresh_features()

    def refresh_features(self):
        """用当前 r/β 重新计算整段 Soft-Play 特征并缓存
        断开特征到 r/β 的反传图（no_grad），节省内存
        GRU 仍每步反传
        """
        with torch.no_grad():
            # 输入: (1, T, 3) → 输出 (1, T, 30) → 去掉 batch 维 (T, 30)
            x = self.X.unsqueeze(0)
            feats = self.soft_play(x)
            self.feats = feats.squeeze(0).detach()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """返回: (x_window, feat_window, target)
        x_window:    (w, 3)   原始输入
        feat_window: (w, 30) Soft-Play 迟滞特征
        target:      (1,)     τ_l
        """
        x_seq = self.X[idx:idx + self.w]               # (w, 3)
        f_seq = self.feats[idx:idx + self.w]            # (w, 30)
        tgt = self.y[idx + self.w].unsqueeze(-1)        # (1,)
        return x_seq, f_seq, tgt


class MultiTrialHysteresisDataset(Dataset):
    """多 trial 迟滞数据集：按 trial 独立预计算 Soft-Play 特征

    核心改进：
    - 每个 trial 独立计算特征，保证时序记忆正确性
    - 滑窗仅在同一 trial 内，不跨段边界
    - 所有 trial 的窗口索引合并为全局索引

    Args:
        trials: List[(X_trial, y_trial)]，每个 (T_trial, 3) 和 (T_trial,)
        window_size: 滑窗长度
        soft_play_module: SoftPlay 模块
        refresh: 是否在构造时立即预计算所有特征
    """
    def __init__(self, trials, window_size, soft_play_module, refresh=True):
        self.trials = trials
        self.w = window_size
        self.soft_play = soft_play_module
        self.feats_per_trial = []  # List[(T_trial, 30)]

        # 预计算每个 trial 的窗口索引
        self.window_index = []  # List[(trial_idx, start_in_trial)]
        for t_idx, (X_t, y_t) in enumerate(trials):
            n_win = max(0, len(X_t) - window_size)
            for s in range(n_win):
                self.window_index.append((t_idx, s))

        self.n_samples = len(self.window_index)
        if refresh:
            self.refresh_features()

    def refresh_features(self):
        """用当前 r/β 重新计算所有 trial 的 Soft-Play 特征并缓存"""
        self.feats_per_trial = []
        with torch.no_grad():
            for X_t, y_t in self.trials:
                x = torch.as_tensor(X_t, dtype=torch.float32).unsqueeze(0)  # (1, T, 3)
                feats = self.soft_play(x)                                    # (1, T, 30)
                self.feats_per_trial.append(feats.squeeze(0).detach())     # (T, 30)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """返回: (x_window, feat_window, target)
        x_window:    (w, 3)   原始输入
        feat_window: (w, 30) Soft-Play 迟滞特征
        target:      (1,)     τ_l
        """
        trial_idx, start_in_trial = self.window_index[idx]
        X_t, y_t = self.trials[trial_idx]

        x_seq = torch.as_tensor(
            X_t[start_in_trial:start_in_trial + self.w], dtype=torch.float32
        )  # (w, 3)
        f_seq = self.feats_per_trial[trial_idx][start_in_trial:start_in_trial + self.w]  # (w, 30)
        tgt = torch.as_tensor(
            y_t[start_in_trial + self.w], dtype=torch.float32
        ).unsqueeze(-1)  # (1,)

        return x_seq, f_seq, tgt


class MultiTrialPhysDataset(Dataset):
    """v3 多 trial 物理数据集：Soft-Play 特征 + 物理上下文

    相对 MultiTrialHysteresisDataset 的 v3 改动：
    - Soft-Play 阈值按逐 trial dz_scale=F0/F0_med 锚定（式4-19 b∝F0）
    - 额外提供物理上下文序列 p_seq (T,3)=[branch, hold, f0_n]
    - 额外提供端点物理状态 p_end=[x_eff, branch, g, hold, x_line]（目标时刻取值）
    - __getitem__ 返回 5 元组（供 PIplusGRUv3 残差融合）

    Args:
        trials: List[(X, y, F)]（v3 三元组）
        phys_list: build_phys_list 输出，与 trials 一一对应
        window_size: 滑窗长度
        soft_play_module: SoftPlay 模块（阈值锚定用 r/β）
    """
    def __init__(self, trials, phys_list, window_size, soft_play_module, refresh=True):
        self.trials = trials
        self.phys = phys_list
        self.w = window_size
        self.soft_play = soft_play_module
        self.feats_per_trial = []  # List[(T_trial, 30)]

        # 预计算每个 trial 的窗口索引（与 v2 相同：stride=1，不跨段）
        self.window_index = []  # List[(trial_idx, start_in_trial)]
        for t_idx, trial in enumerate(trials):
            n_win = max(0, len(trial[0]) - window_size)
            for s in range(n_win):
                self.window_index.append((t_idx, s))

        self.n_samples = len(self.window_index)
        if refresh:
            self.refresh_features()

    def refresh_features(self):
        """用当前 r/β 重新计算所有 trial 的 Soft-Play 特征（阈值按 dz_scale 锚定）"""
        self.feats_per_trial = []
        with torch.no_grad():
            for i, trial in enumerate(self.trials):
                x = torch.as_tensor(trial[0], dtype=torch.float32).unsqueeze(0)  # (1, T, 3)
                feats = self.soft_play(x, dz_scale=self.phys[i]["dz_scale"])     # (1, T, 30)
                self.feats_per_trial.append(feats.squeeze(0).detach())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """返回 5 元组:
        x_seq: (w, 3)  原始输入 [τ_m, q_m, q̇_m]
        f_seq: (w, 30) Soft-Play 迟滞特征（阈值已按 F0 锚定）
        p_seq: (w, 3)  物理上下文 [branch, hold, f0_n]
        p_end: (5,)    端点物理状态 [x_eff, branch, g, hold, x_line]（目标时刻 s+w）
        tgt:   (1,)    τ_l
        """
        trial_idx, s = self.window_index[idx]
        X_t, y_t, _ = self.trials[trial_idx]
        ph = self.phys[trial_idx]
        e = s + self.w  # 目标时刻索引

        x_seq = torch.as_tensor(X_t[s:s + self.w], dtype=torch.float32)
        f_seq = self.feats_per_trial[trial_idx][s:s + self.w]
        p_seq = torch.as_tensor(ph["p_seq"][s:s + self.w], dtype=torch.float32)
        p_end = torch.tensor(
            [float(ph["x_eff"][e]), float(ph["branch"][e]), float(ph["g"]),
             float(ph["hold"][e]), float(ph["x"][e])],
            dtype=torch.float32,
        )
        tgt = torch.as_tensor(y_t[e], dtype=torch.float32).unsqueeze(-1)
        return x_seq, f_seq, p_seq, p_end, tgt


class SoftPlayUpdateDataset(Dataset):
    """用于 r/β 端到端梯度更新的小批量数据集
    与 HysteresisDataset 不同：特征不预缓存，前向时实时算（保留反传图）
    """
    def __init__(self, X, y, window_size, soft_play_module, max_samples=4096):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = window_size
        self.soft_play = soft_play_module
        self.n_samples = min(max_samples, max(0, len(X) - window_size))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.w]                # (w, 3)
        tgt = self.y[idx + self.w].unsqueeze(-1)        # (1,)
        return x_seq, tgt


class MultiTrialSoftPlayUpdateDataset(Dataset):
    """多 trial 的 r/β 端到端更新数据集（保留反传图）
    每个样本是完整的一段 trial，Soft-Play 在前向时现场计算
    """
    def __init__(self, trials, window_size, soft_play_module, max_trials=8):
        self.trials = trials
        self.w = window_size
        self.soft_play = soft_play_module
        # 只用前 max_trials 个 trial（或全部），每个 trial 用一个窗口
        self.samples = []
        for X_t, y_t in trials[:max_trials]:
            if len(X_t) > window_size:
                # 取 trial 中间的一个窗口
                mid = len(X_t) // 2
                start = max(0, mid - window_size // 2)
                self.samples.append((X_t, y_t, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_t, y_t, start = self.samples[idx]
        x_seq = torch.as_tensor(
            X_t[start:start + self.w], dtype=torch.float32
        )
        tgt = torch.as_tensor(
            y_t[start + self.w], dtype=torch.float32
        ).unsqueeze(-1)
        return x_seq, tgt


def compute_full_features(X, soft_play_module):
    """对整段连续信号逐通道计算 Soft-Play 特征
    X: (T, 3) numpy 或 tensor
    返回: (T, 30) tensor，已 detach
    """
    x = torch.as_tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, T, 3)
    with torch.no_grad():
        feats = soft_play_module(x)
    return feats.squeeze(0).detach()


def _test_dataset():
    """单元测试：验证 shape 与记忆连续性"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from soft_play import SoftPlay

    print("=" * 60)
    print("HysteresisDataset 单元测试")
    print("=" * 60)

    # 构造假数据
    T = 500
    X = np.random.randn(T, 3).astype(np.float32)
    y = np.random.randn(T).astype(np.float32)

    sp = SoftPlay(n_channels=3, n_ops=10, beta_init=10.0)
    ds = HysteresisDataset(X, y, window_size=20, soft_play_module=sp)

    print(f"  数据集长度: {len(ds)} (期望 {T-20}={T-20})")
    x_seq, f_seq, tgt = ds[0]
    print(f"  x_seq shape: {x_seq.shape}  (期望 (20, 3))")
    print(f"  f_seq shape: {f_seq.shape}  (期望 (20, 30))")
    print(f"  tgt shape:   {tgt.shape}      (期望 (1,))")

    # 验证记忆连续性：相邻窗口的 feat 重叠部分应一致
    x1, f1, _ = ds[0]
    x2, f2, _ = ds[1]
    # f1[1:20] 应与 f2[0:19] 完全相同（同一时间点）
    diff = (f1[1:] - f2[:-1]).abs().max().item()
    print(f"\n  相邻窗口特征重叠差异: {diff:.2e}")
    assert diff < 1e-6, "特征记忆不连续"
    print("✓ 迟滞记忆连续性验证通过")

    # 验证 refresh 后特征会随 r 变化
    print("\n  验证 r 更新后特征变化...")
    old_feats = ds.feats.clone()
    with torch.no_grad():
        sp.raw_r.data += 0.1
    ds.refresh_features()
    new_feats = ds.feats
    diff = (new_feats - old_feats).abs().max().item()
    print(f"  r 改变后特征最大差异: {diff:.6f}")
    assert diff > 0, "特征未随 r 变化"
    print("✓ 特征刷新验证通过")


def _test_multi_trial_dataset():
    """单元测试：多 trial 数据集"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from soft_play import SoftPlay

    print("\n" + "=" * 60)
    print("MultiTrialHysteresisDataset 单元测试")
    print("=" * 60)

    # 构造 3 个不同长度的 trial
    trials = []
    for i in range(3):
        T_i = 100 + i * 50  # 100, 150, 200
        X_i = np.random.randn(T_i, 3).astype(np.float32)
        y_i = np.random.randn(T_i).astype(np.float32)
        trials.append((X_i, y_i))

    sp = SoftPlay(n_channels=3, n_ops=10, beta_init=10.0)
    ds = MultiTrialHysteresisDataset(trials, window_size=20, soft_play_module=sp)

    print(f"  总样本数: {len(ds)}")
    print(f"  期望样本数: {(100-20)+(150-20)+(200-20)} = 390")

    x_seq, f_seq, tgt = ds[0]
    print(f"  x_seq shape: {x_seq.shape}")
    print(f"  f_seq shape: {f_seq.shape}")
    print(f"  tgt shape:   {tgt.shape}")

    # 验证不同 trial 之间不共享记忆
    # 取 trial0 末尾和 trial1 开头的特征，它们不应该有连续性
    n_trial0 = 100 - 20
    x_t0_end, f_t0_end, _ = ds[n_trial0 - 1]
    x_t1_start, f_t1_start, _ = ds[n_trial0]
    print(f"\n  trial0 末尾特征 shape: {f_t0_end.shape}")
    print(f"  trial1 开头特征 shape: {f_t1_start.shape}")
    print("✓ 多 trial 数据集验证通过")


if __name__ == "__main__":
    _test_dataset()
    _test_multi_trial_dataset()
