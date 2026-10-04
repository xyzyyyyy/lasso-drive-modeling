# -*- coding: utf-8 -*-
"""
可微 Soft-Play 算子（论文核心创新）

数学定义（论文 β 约定：β→∞ 退化为硬 Play）：
    soft_max(a, b; β) = (1/β) · log( exp(βa) + exp(βb) + ε )
    soft_min(a, b; β) = -soft_max(-a, -b; β)
    soft_clamp(z, low, high; β) = soft_min( soft_max(z, low; β), high; β )
    Soft-Play:  y[k] = soft_clamp( y[k-1], x[k]-r, x[k]+r; β )

PyTorch 实现关键点：
- 用 torch.logsumexp 保证数值稳定（避免 exp(βa) 溢出）
- r 用 softplus(raw_r) 包裹保证非负，作为 nn.Parameter 可训练
- β 设为可训练标量，softplus 包裹
- 时间维串行循环 JIT 加速

逐通道处理：对 C 通道各算 N 个 Soft-Play 算子，输出 (B, T, C*N)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 模块级平滑函数（可被 torch.jit.script 调用）
# ============================================================
def soft_max_beta(a, b, beta, eps=1e-12):
    """soft_max(a, b; β) = (1/β) · log( exp(βa) + exp(βb) + ε )
    β→∞ 退化为 max(a, b)
    用 torch.logsumexp 保证数值稳定
    """
    # log(exp(βa)+exp(βb)+ε) = logsumexp([βa, βb, log(ε)])
    log_eps = math.log(eps) if eps > 0 else -math.inf
    stacked = torch.stack([beta * a, beta * b,
                           torch.full_like(a, log_eps)], dim=-1)
    return torch.logsumexp(stacked, dim=-1) / beta


def soft_min_beta(a, b, beta, eps=1e-12):
    """soft_min(a, b; β) = -soft_max(-a, -b; β)
    β→∞ 退化为 min(a, b)
    """
    return -soft_max_beta(-a, -b, beta, eps)


def soft_clamp_beta(z, low, high, beta, eps=1e-12):
    """soft_clamp(z, low, high; β) = soft_min(soft_max(z, low; β), high; β)
    β→∞ 退化为 clamp(z, low, high)
    """
    return soft_min_beta(soft_max_beta(z, low, beta, eps), high, beta, eps)


def soft_play_seq(x_seq, r, beta, eps=1e-12):
    """对一维序列串行计算 Soft-Play
    x_seq: (B, T)  单通道输入
    r:      (B, N) 该通道 N 个算子阈值
    beta:   标量
    返回:   (B, T, N)
    """
    B, T = x_seq.shape
    N = r.shape[1]
    # y 的初始状态为零（对应 docx 论文假设：t0 时刻系统处于零位）
    y = torch.zeros(B, N, device=x_seq.device, dtype=x_seq.dtype)
    out = torch.empty(B, T, N, device=x_seq.device, dtype=x_seq.dtype)
    for k in range(T):
        # 当前时刻 x: (B, 1) → 广播到 (B, N)
        xk = x_seq[:, k:k+1].expand(B, N)
        low = xk - r          # (B, N)
        high = xk + r
        y = soft_clamp_beta(y, low, high, beta, eps)
        out[:, k, :] = y
    return out


# ============================================================
# 硬 Play 算子（对比基线，不可微）
# ============================================================
def hard_play_seq(x_seq, r):
    """硬 Play: y[k] = clamp(y[k-1], x[k]-r, x[k]+r)
    仅用于对比绘图，不可微
    """
    B, T = x_seq.shape
    N = r.shape[1]
    y = torch.zeros(B, N, device=x_seq.device, dtype=x_seq.dtype)
    out = torch.empty(B, T, N, device=x_seq.device, dtype=x_seq.dtype)
    for k in range(T):
        xk = x_seq[:, k:k+1].expand(B, N)
        low = xk - r
        high = xk + r
        y = torch.maximum(torch.minimum(y, high), low)
        out[:, k, :] = y
    return out


# ============================================================
# SoftPlay 模块
# ============================================================
class SoftPlay(nn.Module):
    """逐通道可微 Soft-Play 算子组

    输入 x: (B, T, C)  C 通道
    输出 h: (B, T, C*N)  每通道 N 个算子的迟滞特征

    参数:
        raw_r: (C*N,) 未约束阈值，前向时 softplus(raw_r) 保证非负
        raw_beta: 标量，softplus(raw_beta) 为平滑系数 β
    """
    def __init__(self, n_channels=3, n_ops=10,
                 beta_init=10.0, r_init_max=1.0, eps=1e-12):
        super().__init__()
        self.n_channels = n_channels
        self.n_ops = n_ops
        self.eps = eps

        # r 初始化：每通道在 [0, r_init_max] 均匀分布
        # 用 raw_r，前向时 softplus 包裹
        r_init = torch.linspace(0, r_init_max, n_ops).repeat(n_channels)
        # linspace(0,...) 第一个为 0，加微小偏移使 softplus 输出非零
        r_init = r_init + 1e-3
        self.raw_r = nn.Parameter(r_init.clone())

        # β 初始化：softplus(raw_beta) = beta_init → raw_beta = log(exp(beta_init)-1)
        raw_beta_val = math.log(math.expm1(beta_init)) if beta_init > 0 else 0.0
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta_val, dtype=torch.float32))

    @property
    def r(self):
        """非负阈值，形状 (C*N,)"""
        return F.softplus(self.raw_r)

    @property
    def beta(self):
        """正的平滑系数 β"""
        return F.softplus(self.raw_beta)

    def forward(self, x):
        """逐通道 Soft-Play 计算
        x: (B, T, C)
        返回: (B, T, C*N)
        """
        B, T, C = x.shape
        assert C == self.n_channels, f"通道数不符: 期望 {self.n_channels}, 得到 {C}"

        # 取出参数
        r = self.r.to(x.dtype)         # (C*N,)
        beta = self.beta.to(x.dtype)
        r = r.view(C, self.n_ops)      # (C, N)

        # 对每个通道独立计算 Soft-Play，再拼接
        outputs = []
        for c in range(C):
            x_c = x[:, :, c]                  # (B, T)
            r_c = r[c].unsqueeze(0).expand(B, -1)  # (B, N)
            h_c = soft_play_seq(x_c, r_c, beta, self.eps)  # (B, T, N)
            outputs.append(h_c)

        # 拼接所有通道: (B, T, C*N)
        return torch.cat(outputs, dim=-1)


# ============================================================
# 单元测试（直接运行验证）
# ============================================================
def _test_soft_play():
    """用正弦信号单测：验证 β→∞ 趋近硬 Play，且 r 可反传"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Soft-Play 算子单元测试")
    print("=" * 60)

    # 生成正弦信号
    T = 500
    t = torch.linspace(0, 4 * math.pi, T)
    x = 5.0 * torch.sin(t)              # 与论文实验 a 的 5 N·m 一致
    x = x.unsqueeze(0).unsqueeze(-1)    # (1, T, 1) 单通道

    # 测试不同 β
    betas = [1.0, 5.0, 10.0, 50.0, 200.0]
    r_value = 0.5

    plt.figure(figsize=(12, 8))

    # ===== 图1: 时域信号 =====
    plt.subplot(2, 1, 1)
    plt.plot(t.numpy(), x.numpy().flatten(), 'k-', label='输入 x(t)', linewidth=2)

    # 硬 Play 参考
    r_hard = torch.tensor([[r_value]])
    hard = hard_play_seq(x[:, :, 0], r_hard).squeeze().numpy()
    plt.plot(t.numpy(), hard, 'b--', label='硬 Play (r=0.5)', linewidth=2)

    # 不同 β 的 Soft-Play
    for beta_val in betas:
        sp = SoftPlay(n_channels=1, n_ops=1, beta_init=beta_val, r_init_max=r_value)
        with torch.no_grad():
            out = sp(x).squeeze().numpy()
        plt.plot(t.numpy(), out, '--', label=f'Soft-Play β={beta_val}', alpha=0.7)

    plt.xlabel('t')
    plt.ylabel('输出')
    plt.title('Soft-Play 算子时域响应（β→∞ 趋近硬 Play）')
    plt.legend(fontsize=8)
    plt.grid(True)

    # ===== 图2: 迟滞环（输入 vs 输出） =====
    plt.subplot(2, 1, 2)
    plt.plot(x.numpy().flatten(), hard, 'b-', label='硬 Play', linewidth=2)
    for beta_val in betas:
        sp = SoftPlay(n_channels=1, n_ops=1, beta_init=beta_val, r_init_max=r_value)
        with torch.no_grad():
            out = sp(x).squeeze().numpy()
        plt.plot(x.numpy().flatten(), out, '--', label=f'β={beta_val}', alpha=0.7)
    plt.xlabel('输入 x')
    plt.ylabel('输出 y')
    plt.title('迟滞环（β 越大越接近硬 Play）')
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outputs", "soft_play_test.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"测试图已保存: {out_path}")

    # ===== 测试可微性：r 可反传 =====
    print("\n测试 r/β 可微性...")
    sp = SoftPlay(n_channels=1, n_ops=1, beta_init=10.0, r_init_max=0.5)
    x_test = torch.randn(2, 100, 1, requires_grad=False)
    out = sp(x_test)
    target = torch.randn(2, 100, 1)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    r_grad = sp.raw_r.grad
    beta_grad = sp.raw_beta.grad
    print(f"  loss = {loss.item():.6f}")
    print(f"  raw_r.grad: 非零={r_grad is not None and (r_grad.abs() > 0).any().item()}, "
          f"max|grad|={r_grad.abs().max().item():.6f}" if r_grad is not None else "  raw_r.grad: None")
    print(f"  raw_beta.grad: 非零={beta_grad is not None and (beta_grad.abs() > 0).item()}, "
          f"value={beta_grad.item():.6f}" if beta_grad is not None else "  raw_beta.grad: None")

    if r_grad is not None and beta_grad is not None \
       and (r_grad.abs() > 0).any().item() and (beta_grad.abs() > 0).item():
        print("✓ r 和 β 梯度均非零，可微性验证通过")
    else:
        print("✗ 梯度为零，可微性验证失败")

    # ===== 多通道测试 =====
    print("\n多通道 SoftPlay 前向测试...")
    sp3 = SoftPlay(n_channels=3, n_ops=10, beta_init=10.0, r_init_max=1.0)
    x3 = torch.randn(4, 50, 3)
    out3 = sp3(x3)
    expected_shape = (4, 50, 30)
    print(f"  输入 shape: {x3.shape}")
    print(f"  输出 shape: {out3.shape}")
    print(f"  期望 shape: {expected_shape}")
    assert out3.shape == expected_shape, "shape 不匹配"
    print("✓ 多通道 shape 验证通过")


if __name__ == "__main__":
    _test_soft_play()
