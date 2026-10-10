# -*- coding: utf-8 -*-
"""
PI+GRU 耦合迟滞模型（论文四层架构）

1. 输入层:        x(t) = [τ_m, q_m, q̇_m]，三通道
2. 迟滞特征提取层: Soft-Play 逐通道建模 → (B, T, 30)
3. GRU 时序建模层: 双层 64 隐藏单元，输入 (B, T, 33)，层间 Dropout
4. 全连接层:      ReLU(W_fc·h+b_fc) → W_out·y+b_out → τ̂_l

第二版改进：GRU 层间加入 Dropout 抑制过拟合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_play import SoftPlay
from physics_branch import PhysicsCalib


class PIplusGRU(nn.Module):
    """PI+GRU 完整模型

    Args:
        n_channels: 输入通道数（论文=3）
        n_ops: 每通道 Soft-Play 算子数（论文=10）
        hidden_size: GRU 隐藏单元数（论文=64）
        num_layers: GRU 层数（论文=2）
        fc_hidden: 全连接中间层维度（论文=64）
        beta_init: Soft-Play 平滑系数 β 初始值（论文=10）
        r_init_max: 阈值 r 初始化上界
        dropout: GRU 层间 dropout 率（第二版新增）
    """
    def __init__(self, n_channels=3, n_ops=10, hidden_size=64,
                 num_layers=2, fc_hidden=64, beta_init=10.0, r_init_max=1.0,
                 dropout=0.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_ops = n_ops

        # 层1: Soft-Play 迟滞特征提取
        self.soft_play = SoftPlay(
            n_channels=n_channels,
            n_ops=n_ops,
            beta_init=beta_init,
            r_init_max=r_init_max,
        )

        # 层3: 双层 GRU（第二版：层间加 dropout）
        gru_input = n_channels + n_channels * n_ops  # 3 + 30 = 33
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # 层间 dropout
        )

        # 层4: 全连接（含 ReLU 中间层）
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.dropout = nn.Dropout(dropout)   # FC 前 dropout
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x, feat=None):
        """前向传播
        Args:
            x:    (B, T, C) 原始输入
            feat: (B, T, C*N) 预计算的 Soft-Play 特征（None 则现场算）
        Returns:
            pred: (B, 1) τ̂_l
        """
        if feat is None:
            # 现场算（小批量或 r/β 端到端更新）
            feat = self.soft_play(x)
        # 拼接原始输入与迟滞特征: (B, T, C + C*N)
        h = torch.cat([x, feat], dim=-1)
        # GRU 取末时刻
        out, _ = self.gru(h)
        last = out[:, -1, :]               # (B, hidden)
        # ReLU 中间层 + Dropout + 线性输出
        y = F.relu(self.fc1(last))
        y = self.dropout(y)
        pred = self.fc2(y)
        return pred

    def forward_with_softplay(self, x):
        """端到端前向（保留 r/β 反传图），用于 r/β 更新
        x: (B, T, C)
        返回: (B, 1) 末时刻预测
        """
        feat = self.soft_play(x)
        return self.forward(x, feat)

    def forward_sequence(self, x):
        """对整个序列每个时刻都输出预测（保留 r/β 反传图）
        用于 r/β 端到端更新，保留 Soft-Play 记忆连续性
        x: (1, T, C)  单连续序列
        返回: (1, T, 1)
        """
        feat = self.soft_play(x)             # (1, T, C*N)
        h = torch.cat([x, feat], dim=-1)      # (1, T, C+C*N)
        out, _ = self.gru(h)                  # (1, T, hidden)
        y = F.relu(self.fc1(out))            # (1, T, fc_hidden)
        pred = self.fc2(y)                    # (1, T, 1)
        return pred


class PIplusGRUv3(nn.Module):
    """v3：白箱物理主干 + PI+GRU 残差（式(4-19)/(4-23) 进入数据模型）

    架构：
    1. 物理主干（不可训练部分为数据标定量）:
           τ_l^phys = γ_k·x_eff + γ_b·g·branch
       其中 x_eff/branch/g 来自白箱 Play 状态机（physics_branch.play_state_machine，
       严格等价于论文式(4-19)/(4-23) 两条线性分支 + 死区保持），
       γ_k∈(0,1.2]、γ_b≥0 为可微校准因子（PhysicsCalib）
    2. Soft-Play 特征层：阈值按逐 trial dz_scale=F0/F0_med 锚定物理死区
    3. GRU 输入 = [τ_m,q_m,q̇_m](3) + Soft-Play(30) + 物理上下文(3) = 36 维
    4. fc2 零初始化 ⇒ 训练起点 τ̂ = τ_l^phys（纯白箱），GRU 只学残差 τ̂_res

    Args 同 PIplusGRU。
    """
    def __init__(self, n_channels=3, n_ops=10, hidden_size=64,
                 num_layers=2, fc_hidden=64, beta_init=10.0, r_init_max=1.0,
                 dropout=0.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_ops = n_ops

        # 层1a: Soft-Play 迟滞特征（阈值由 dz_scale 外部锚定）
        self.soft_play = SoftPlay(
            n_channels=n_channels, n_ops=n_ops,
            beta_init=beta_init, r_init_max=r_init_max,
        )
        # 层1b: 白箱物理主干校准因子 γ_k, γ_b
        self.calib = PhysicsCalib()

        # 层3: GRU（输入 3 + 30 + 3 = 36 维）
        gru_input = n_channels + n_channels * n_ops + 3
        self.gru = nn.GRU(
            input_size=gru_input, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 层4: 残差头（fc2 零初始化 → 起点为纯白箱）
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def physics_trunk(self, p_end, detach_gamma=False):
        """物理主干 τ_l^phys = γ_k·x_eff + γ_b·g·branch
        p_end: (B, 5)=[x_eff, branch, g, hold, x_line]
        detach_gamma=True 时 γ 视为常数（主循环用，γ 只在 r/β/γ 更新阶段训练）
        """
        x_eff, branch, g = p_end[:, 0:1], p_end[:, 1:2], p_end[:, 2:3]
        if detach_gamma:
            gk = self.calib.gamma_k.detach()
            gb = self.calib.gamma_b.detach()
            return gk * x_eff + gb * g * branch
        return self.calib.physics_end(x_eff, branch, g)

    def _residual(self, x, feat, p_seq):
        """GRU 残差头：返回 (B, 1) 残差 τ̂_res"""
        h = torch.cat([x, feat, p_seq], dim=-1)
        out, _ = self.gru(h)
        y = F.relu(self.fc1(out[:, -1, :]))
        y = self.dropout(y)
        return self.fc2(y)

    def forward(self, x, feat, p_seq, p_end, live_physics=False):
        """前向传播（窗口模式）
        Args:
            x:    (B, T, C) 原始输入
            feat: (B, T, C*N) 预缓存 Soft-Play 特征
            p_seq: (B, T, 3) 物理上下文 [branch, hold, f0_n]
            p_end: (B, 5) 端点物理状态 [x_eff, branch, g, hold, x_line]
            live_physics: True 时物理主干保留 γ 反传图（r/β/γ 更新阶段用）
        Returns:
            (pred, resid): τ̂_l = τ_l^phys + τ̂_res
        """
        resid = self._residual(x, feat, p_seq)
        phys = self.physics_trunk(p_end, detach_gamma=not live_physics)
        return phys + resid, resid

    def forward_sequence(self, x, p_seq, x_eff, branch, g, feat=None):
        """序列前向（r/β/γ 端到端更新用，保留全部反传图）

        时序约定与主循环严格一致：GRU 消费输入 [0..t-1]，预测目标 y[t]，
        物理主干在目标时刻 t 取值（Play 算子因果依赖当前输入，无泄漏）。
        Soft-Play/GRU 从序列起点连续演化（记忆正确）。

        Args:
            x: (1, T, C) 单连续序列（从 trial 起点开始）
            p_seq: (1, T, 3) 物理上下文
            x_eff: (T, 1) 有效输入序列（全 trial 状态机）
            branch: (T, 1) 分支序列
            g: float 迟滞半宽
            feat: 可选，外部预计算的 Soft-Play 特征 (1, T, C*N)
        Returns:
            (pred, resid): (1, T-1, 1)，对应目标时刻 t=1..T-1
        """
        if feat is None:
            feat = self.soft_play(x)               # live r/β
        h = torch.cat([x, feat, p_seq], dim=-1)   # (1, T, 36)
        out, _ = self.gru(h[:, :-1, :])            # GRU 消费 [0..t-1] → (1, T-1, hidden)
        resid = self.fc2(F.relu(self.fc1(out)))   # (1, T-1, 1)
        # 物理主干在目标时刻 t=1..T-1 取值
        phys = self.calib.physics_end(x_eff[1:], branch[1:],
                                      x.new_tensor(float(g)))  # (T-1, 1)
        return phys.unsqueeze(0) + resid, resid


def _test_model():
    """单元测试：验证前向 shape 和反传"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("PIplusGRU 单元测试")
    print("=" * 60)

    model = PIplusGRU(
        n_channels=3, n_ops=10,
        hidden_size=64, num_layers=2, fc_hidden=64,
        beta_init=10.0, r_init_max=1.0,
    )

    # 前向 shape 测试
    B, T, C = 4, 20, 3
    x = torch.randn(B, T, C)
    pred = model(x)
    print(f"  输入 shape:  {x.shape}")
    print(f"  输出 shape:  {pred.shape}  (期望 ({B}, 1))")
    assert pred.shape == (B, 1), "输出 shape 不匹配"
    print("✓ 前向 shape 验证通过")

    # 反传测试
    target = torch.randn(B, 1)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    r_grad = model.soft_play.raw_r.grad
    beta_grad = model.soft_play.raw_beta.grad
    print(f"\n  loss = {loss.item():.6f}")
    if r_grad is not None:
        print(f"  raw_r.grad max|grad| = {r_grad.abs().max().item():.6f}")
    if beta_grad is not None:
        print(f"  raw_beta.grad = {beta_grad.item():.6f}")
    print("✓ 反传验证通过")

    # 参数统计
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数数:     {n_params}")
    print(f"  可训练参数数: {n_trainable}")

    # forward_with_softplay 测试
    pred2 = model.forward_with_softplay(x)
    print(f"\n  forward_with_softplay 输出 shape: {pred2.shape}")
    assert pred2.shape == (B, 1)
    print("✓ 端到端前向验证通过")


if __name__ == "__main__":
    _test_model()
