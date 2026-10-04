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
