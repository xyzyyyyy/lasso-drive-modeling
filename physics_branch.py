# -*- coding: utf-8 -*-
"""
v3 白箱物理分支：双向套索力矩传递模型（式(4-10)~(4-24) 进入数据模型）

核心数学事实（v3 的理论基石，论文未做）：
    式(4-19) 加载:  τ_l = k_τ·τ_m + b   （b = 2F0(ξa-ξb)rl/(χa+χb) < 0，因 ξa-ξb<0）
    式(4-23) 卸载:  τ_l = k_τ·τ_m - b
    令 g = |b|（迟滞半宽）、x = k_τ·τ_m，两条线合并为一个 Play 动力学：
        τ_l(t) = clamp( τ_l(t-1), x(t)-g, x(t)+g )
    即 白箱模型 ≡ 半间隙 g 的硬 Play 算子。由此自洽地得到：
        启动死区(式4-22): Δτ_m = g/k_τ = 2·F0·rm·|ψ|
        换向死区(式4-21): Δτ_m = 2g/k_τ = 4·F0·rm·|ψ|
    —— 论文 P-I 模型"物理意义明确"的严格版本。

标定策略（台架常数不可精确已知的补偿；经逐 trial 诊断修订）：
    - k_τ：训练数据 OLS 直接标定（τ_l ~ τ_m 斜率）
    - g：仅用"可辨识" trial（g 扫描有内部最优且 RMSE 较线性模型显著
      下降）的 Play 拟合最优 g 取中位数标定 g0；逐 trial 按
      g_i = g0·F0_i/F0_med 缩放（式(4-19) 中 b ∝ F0，物理保证逐 trial
      相对比值；θ/EA/ks/μ 的绝对误差被数据标定吸收，不影响随 F0 变化律）
    - s_loop：可辨识 trial 的方向分桶残差差中位数判号（本系统全部>0 ⇒
      A=−2g·Δx<0 耗散 ⇒ −1；晚段漂移 trial 的分桶差转负系台架状态
      退化伪影，不参与判号）
    - γ_k∈(0,1.2]、γ_b≥0(softplus)：可微校准因子，训练微调；
      γ_b≥0 是结构性物理约束——迟滞环耗散恒非负。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 台架常数（默认取论文值；绝对误差由数据标定吸收，见模块头注释）
# ============================================================
PHYS_MU = 0.2        # 摩擦系数 μ（式4-10）
PHYS_THETA = math.pi  # 套管弯曲包角 θ (rad)
PHYS_EA = 200.0       # 绳索抗拉刚度 EA (N)
PHYS_KS = 3.069e4     # 串联弹簧刚度 ks (N/m)
PHYS_RM = 0.02        # 电机端线轮半径 rm (m)
PHYS_RL = 0.02        # 负载端线轮半径 rl (m)
PHYS_L = 0.5          # 套管路径长度 L (m)


def lasso_physics_scalars(F0, mu=PHYS_MU, theta=PHYS_THETA, EA=PHYS_EA,
                          ks=PHYS_KS, rm=PHYS_RM, rl=PHYS_RL, L=PHYS_L):
    """式(4-10)~(4-24)：给定预紧力 F0，返回全部白箱标量

    ξ_a = e^{-μθ}, ξ_b = e^{μθ}                      （式4-10）
    γ_a = La(1-ξ_a)/(μθa), γ_b = -Lb(1-ξ_b)/(μθb)     （式4-14）
    χ_a = γ_a/L + 4EA/(ksL), χ_b = γ_b/L - 4EA/(ksL)  （式4-16）
    ψ   = (ξ_a-ξ_b)/(χ_b·ξ_a + χ_a·ξ_b)               （式4-21）
    k_τ = (χ_b·ξ_a+χ_a·ξ_b)·rl/((χ_a+χ_b)·rm)         （式4-19 系数）
    b   = 2F0(ξ_a-ξ_b)·rl/(χ_a+χ_b)                   （式4-19 截距，<0）
    启动死区 = 2F0·rm·|ψ|（式4-22），换向死区 = 4F0·rm·|ψ|（式4-21）
    """
    xi_a = math.exp(-mu * theta)
    xi_b = math.exp(+mu * theta)
    gamma_a = L * (1.0 - xi_a) / (mu * theta)
    gamma_b = -L * (1.0 - xi_b) / (mu * theta)
    chi_a = gamma_a / L + 4.0 * EA / (ks * L)
    chi_b = gamma_b / L - 4.0 * EA / (ks * L)
    psi = (xi_a - xi_b) / (chi_b * xi_a + chi_a * xi_b)
    k_tau = (chi_b * xi_a + chi_a * xi_b) * rl / ((chi_a + chi_b) * rm)
    b = 2.0 * F0 * (xi_a - xi_b) * rl / (chi_a + chi_b)
    return {
        "xi_a": xi_a, "xi_b": xi_b,
        "gamma_a": gamma_a, "gamma_b": gamma_b,
        "chi_a": chi_a, "chi_b": chi_b,
        "psi": psi,
        "k_tau": k_tau,       # 传动增益
        "b": b,              # 式(4-19) 截距（负值）
        "g": abs(b),         # 迟滞半宽（Play 半间隙）
        "dz_start": 2.0 * F0 * rm * abs(psi),   # 式(4-22) 启动死区
        "dz_rev": 4.0 * F0 * rm * abs(psi),     # 式(4-21) 换向死区
    }


# ============================================================
# 白箱状态机（等价于硬 Play + 式4-19/4-23 分解）
# ============================================================
def play_state_machine(x, g):
    """逐 trial 白箱状态机

    τ_l(t) = clamp(τ_l(t-1), x-g, x+g)，x = k_τ·τ_m（标定后）
    分解为 τ_l = x_eff + branch·g：
      branch=-1: 加载线 τ_l = x - g   （式4-19，b<0）
      branch=+1: 卸载线 τ_l = x + g   （式4-23）
      hold=1: 死区保持（启动/换向，式4-21/4-22），x_eff 冻结于反转点

    Args:
        x: (T,) 标定输入 k_τ·τ_m
        g: 迟滞半宽（标量，逐 trial 由 F0 缩放）
    Returns:
        y (T,) 物理输出; branch (T,) ∈ {-1,+1}; hold (T,) ∈ {0,1};
        x_eff (T,) 有效输入（保持期冻结）
    """
    T = len(x)
    y = np.empty(T, dtype=np.float64)
    branch = np.empty(T, dtype=np.float64)
    hold = np.empty(T, dtype=np.float64)
    x_eff = np.empty(T, dtype=np.float64)

    y_prev = 0.0          # 论文假设：trial 起始零位（τ_m=0, τ_l=0）
    br = -1.0             # 初始分支（加载线方向）
    # 恒等式 y = x_eff + branch·g 在初始死区保持段也须成立：
    # y=0, br=-1 ⇒ x_eff = y_prev - br·g = g（初始零位不在分支线上，由恒等式定义）
    x_frozen = y_prev - br * g

    for k in range(T):
        lo = x[k] - g
        hi = x[k] + g
        if y_prev < lo:            # 输出跟随加载线（下边界）
            y_prev = lo
            br = -1.0
            x_frozen = float(x[k])
            holding = False
        elif y_prev > hi:          # 输出跟随卸载线（上边界）
            y_prev = hi
            br = +1.0
            x_frozen = float(x[k])
            holding = False
        else:                      # 死区保持：输出冻结
            holding = True
        y[k] = y_prev
        branch[k] = br
        hold[k] = 1.0 if holding else 0.0
        x_eff[k] = x_frozen if holding else float(x[k])

    return y, branch, hold, x_eff


# ============================================================
# 逐 trial 预紧力估计（来自力传感器列）
# ============================================================
def estimate_f0(F_pair, tau_m, quiet_tau=0.15, head_s=3.0, dt=0.002,
                min_pts=50, fallback_rows=200):
    """估计单 trial 预紧力 F0

    物理依据：静止时两绳各承受预紧力 F0（式4-16: Fa1·χa+Fb1·χb=2F0 在 τ_m=0
    附近两绳拉力均≈F0）。取 trial 前 head_s 秒内 |τ_m|<quiet_tau 的样本，
    F0 = 两绳力各自中位数的均值。

    Args:
        F_pair: (T, 2) [Load Cell1 Force, Load Cell2 Force] (N)
        tau_m:  (T,) 电机输入力矩 (N·m)
    """
    T = len(tau_m)
    n_head = max(int(head_s / dt), 1)
    seg = slice(0, min(n_head, T))
    mask = np.abs(tau_m[seg]) < quiet_tau
    if mask.sum() < min_pts:
        n_fb = min(fallback_rows, T)
        mask = np.ones(n_fb, dtype=bool)
        seg = slice(0, n_fb)
    f1 = F_pair[seg][:, 0][mask]
    f2 = F_pair[seg][:, 1][mask]
    if len(f1) == 0 or len(f2) == 0:
        return float(np.median(F_pair[:fallback_rows].mean(axis=1)))
    return float(0.5 * (np.median(f1) + np.median(f2)))


# ============================================================
# 全局标定（训练开始时执行一次）
# ============================================================
def calibrate_from_trials(trials_ext, g_subsample=5):
    """从训练数据标定白箱常数

    标定方法（经真实数据逐 trial 诊断确定）：
    - k̂_τ：pooled OLS 斜率（迟滞环上下对称，±b 抵消；与 (k,g) 双参数
      扫描最优点一致）
    - g0：逐 trial 1D 扫描 Play 拟合 RMSE 最优 g，仅取"可辨识" trial 的
      中位数。可辨识 = 扫描存在内部最优（argmin 不在网格端点）且
      rmse_opt ≤ 0.85·rmse(g=网格下限)。
      诊断结论：55 个训练 trial 仅前段 9 个可辨识（g_opt 0.5~0.7，
      中位 0.6，与 μ/θ/EA 台架常数理论值 ≈0.62 一致）；trial 20 之后
      台架状态漂移（逐 trial OLS 斜率 1.0→0.57、分桶差转负），扫描对
      g 单调、g 不可辨识——全体中位数触网格下限 0.1，已弃用。
    - s_loop：耗散方向由逐 trial 方向分桶残差差（branch_gap，全分辨率）
      在可辨识 trial 上的中位数判定（健康 trial 全部 gap>0 ⇒ 加载走
      下线 ⇒ 耗散环 A=−2g·Δx<0 ⇒ −1，与理论一致）。全体 trial 的
      分桶差被晚段漂移污染（中位 −0.24 ⇒ 误判 +1），已弃用。

    Args:
        trials_ext: List[(X_trial, y_trial, F_trial)]，
                    X (T,3)=[τ_m,q_m,q̇_m]，y (T,)=τ_l，F (T,2) 力传感
        g_subsample: g 扫描的抽样间隔（Play 率无关；实测 ss=1/2/5/10
          的扫描 RMSE 差异 < 0.002）
    Returns:
        dict: k_tau、g0、g_opt_list（逐trial最优g）、n_ident、g_ident_list、
              F0_med、F0_list、s_loop（本系统=−1）、branch_gap（可辨识
              trial 分桶差中位数）、phys_med、consistency
    """
    tau_m_all = np.concatenate([X[:, 0] for X, _, _ in trials_ext])
    taul_all = np.concatenate([y for _, y, _ in trials_ext])
    # k̂_τ：迟滞环上下对称，pooled OLS 斜率不受 ±b 抵消影响
    k_tau = float(np.polyfit(tau_m_all, taul_all, 1)[0])

    # 逐 trial：g 扫描（子采样提速）+ 方向分桶残差（全分辨率）
    g_grid = np.arange(0.1, 3.01, 0.1)
    g_opts, gap_list, ident_mask = [], [], []
    for X, y, _ in trials_ext:
        x = k_tau * X[::g_subsample, 0]
        ys = y[::g_subsample]
        rmses = np.empty(len(g_grid))
        for gi, g_try in enumerate(g_grid):
            yp, _, _, _ = play_state_machine(x, float(g_try))
            rmses[gi] = np.sqrt(np.mean((yp - ys) ** 2))
        i_opt = int(np.argmin(rmses))
        g_opts.append(float(g_grid[i_opt]))
        # 可辨识：内部最优 + 相对线性模型（g→网格下限）RMSE 显著下降
        ident_mask.append(bool(0 < i_opt < len(g_grid) - 1
                               and rmses[i_opt] <= 0.85 * rmses[0]))

        # 分桶差：加载(dx>0)残差 < 卸载(dx<0)残差 ⇒ gap>0 ⇒ 加载走下线
        xf = k_tau * X[:, 0]
        dx = np.diff(xf, prepend=xf[0])
        r = y - xf
        mv = np.abs(dx) > 1e-4
        ld = r[mv & (dx > 0)]
        ul = r[mv & (dx < 0)]
        gap_list.append(float(np.mean(ul) - np.mean(ld))
                        if len(ld) > 100 and len(ul) > 100 else np.nan)

    g_opts = np.array(g_opts)
    gap_list = np.array(gap_list)
    ident_mask = np.array(ident_mask)
    n_ident = int(ident_mask.sum())

    # g0：可辨识 trial 中位数（≥3 个才可信；否则退回全体中位数）
    if n_ident >= 3:
        g0 = float(np.median(g_opts[ident_mask]))
    else:
        g0 = float(np.median(g_opts))

    # s_loop：可辨识 trial 分桶差中位数判号（>0 ⇒ −1，与理论 A<0 一致）
    if n_ident >= 1:
        gap_use = float(np.nanmedian(gap_list[ident_mask]))
    else:
        gap_use = float(np.nanmean(gap_list)) if np.isfinite(gap_list).any() else 0.0
    s_loop = -1.0 if gap_use > 0 else 1.0

    F0_list = np.array([estimate_f0(F, X[:, 0]) for X, _, F in trials_ext])
    F0_med = float(np.median(F0_list))

    # 物理自洽性诊断：2g/k_τ ≡ 换向死区（式4-21，论文常数代入应≈1）
    phys_med = lasso_physics_scalars(F0_med)
    consistency = (2.0 * phys_med["g"] / phys_med["k_tau"]) / phys_med["dz_rev"]

    return {
        "k_tau": k_tau,
        "g0": g0,
        "g_opt_list": g_opts,
        "n_ident": n_ident,
        "g_ident_list": g_opts[ident_mask] if n_ident else g_opts,
        "F0_med": F0_med,
        "F0_list": F0_list,
        "s_loop": s_loop,
        "branch_gap": float(gap_use),
        "phys_med": phys_med,
        "consistency_2g_over_kt_vs_dzrev": consistency,
    }


def build_phys_list(trials_ext, calib):
    """逐 trial 预计算物理上下文（train/val/test 共用）

    每个 trial 产出：
      f0, g（=g0·F0/F0_med，式4-19 的 b∝F0）, x（=k_τ·τ_m）,
      y_phys（白箱输出）, branch, hold, x_eff, p_seq (T,3)=[branch,hold,f0_n],
      dz_scale（=F0/F0_med，Soft-Play 阈值锚定比例）
    """
    phys_list = []
    for X_t, y_t, F_t in trials_ext:
        f0 = estimate_f0(F_t, X_t[:, 0])
        g = calib["g0"] * f0 / calib["F0_med"]
        x = calib["k_tau"] * X_t[:, 0]
        y_phys, branch, hold, x_eff = play_state_machine(x, g)
        f0_n = f0 / calib["F0_med"] - 1.0
        p_seq = np.stack([branch, hold, np.full(len(x), f0_n)], axis=1)
        phys_list.append({
            "f0": f0, "g": g, "x": x, "y_phys": y_phys,
            "branch": branch, "hold": hold, "x_eff": x_eff,
            "p_seq": p_seq.astype(np.float32),
            "dz_scale": f0 / calib["F0_med"],
        })
    return phys_list


# ============================================================
# 可微校准因子（结构性物理约束）
# ============================================================
class PhysicsCalib(nn.Module):
    """γ_k∈(0,1.2]：传动增益校准；γ_b≥0：迟滞半宽校准（softplus 保证非负）

    γ_b≥0 的结构意义：迟滞环面积 ∝ γ_b·g（式4-19/4-23 两线间距），
    非负 ⇒ 模型永远不可能学出"负耗散"（产能量）的迟滞环——
    能量约束以参数化方式结构性成立，比软惩罚更强。
    初始化 γ_k=γ_b=1.0（标定值即起点）。
    """
    def __init__(self):
        super().__init__()
        # γ_k = 1.2·sigmoid(raw) = 1.0 ⇒ raw = logit(1/1.2) = ln 5
        self.raw_gk = nn.Parameter(torch.tensor(float(math.log(5.0))))
        # γ_b = softplus(raw) = 1.0 ⇒ raw = ln(e-1)
        self.raw_gb = nn.Parameter(torch.tensor(float(math.log(math.e - 1.0))))

    @property
    def gamma_k(self):
        return 1.2 * torch.sigmoid(self.raw_gk)

    @property
    def gamma_b(self):
        return F.softplus(self.raw_gb)

    def physics_end(self, x_eff, branch, g):
        """窗口端点物理主干：τ_l^phys = γ_k·x_eff + γ_b·g·branch
        x_eff (B,1), branch (B,1), g (B,1) —— 也可广播到 (B,T)
        """
        return self.gamma_k * x_eff + self.gamma_b * g * branch


# ============================================================
# 物理约束损失助手
# ============================================================
def loss_bound(pred, x_line, g_end, gamma_k, gamma_b):
    """边界约束（λ1）：预测必须落在加载/卸载线之间
    式(4-19)/(4-23)：k_τ·τ_m - g ≤ τ_l ≤ k_τ·τ_m + g（经 γ 校准）
    """
    lo = gamma_k * x_line - gamma_b * g_end
    hi = gamma_k * x_line + gamma_b * g_end
    return (F.relu(pred - hi) + F.relu(lo - pred)).mean()


def loss_deadzone(resid, hold):
    """死区约束（λ2）：保持窗内（式4-21/4-22）物理输出冻结，
    残差应趋零 ⇒ 罚 hold·resid²（直击换向瞬间 MAXE 病灶）"""
    return (hold * resid ** 2).mean()


def loop_area(pred, x_line):
    """迟滞环面积（梯形）：A = Σ 0.5(τ̂_k+τ̂_{k+1})·(x_{k+1}-x_k)
    耗散环要求 s_loop·A ≥ 0（本系统 s_loop=−1，即 A≤0：A=−2g·Δx<0）"""
    if pred.numel() < 2:
        return pred.new_zeros(())
    return torch.sum(0.5 * (pred[1:] + pred[:-1]) * (x_line[1:] - x_line[:-1]))


def loss_energy(pred, x_line, s_loop):
    """耗散约束（λ3）：relu(-s_loop·A/box)²，box 归一化防止量纲爆炸"""
    A = loop_area(pred, x_line)
    box = ((x_line.max() - x_line.min()) * (pred.max() - pred.min())).clamp(min=1.0)
    return (F.relu(-s_loop * A / box)) ** 2


# ============================================================
# 自测
# ============================================================
def _test():
    print("=" * 60)
    print("physics_branch 自测")
    print("=" * 60)

    # 1) 标量自洽性：2g/k_τ ≡ 换向死区（式4-21）
    sc = lasso_physics_scalars(F0=30.0)
    ratio = (2.0 * sc["g"] / sc["k_tau"]) / sc["dz_rev"]
    print(f"  一致性 2g/k_τ vs dz_rev: ratio={ratio:.6f}（应≈1）")
    assert abs(ratio - 1.0) < 1e-9, "式(4-21) 自洽性失败"
    print(f"  k_τ={sc['k_tau']:.4f}, b={sc['b']:.4f}(<0), g={sc['g']:.4f}, "
          f"启动死区={sc['dz_start']:.4f} N·m")

    # 2) 状态机 = 硬 Play + 分解恒等式
    rng = np.random.default_rng(0)
    T = 4000
    x = np.cumsum(rng.normal(0, 0.01, T))          # 随机游走
    g = 0.3
    y, br, hold, x_eff = play_state_machine(x, g)
    # 分解恒等式：y = x_eff + branch·g
    err = np.max(np.abs(y - (x_eff + br * g)))
    print(f"  分解恒等式 max|y-(x_eff+branch·g)| = {err:.2e}（应≈0）")
    assert err < 1e-10
    # 硬 Play 等价性（参考实现）
    y_ref = 0.0
    bad = 0
    for k in range(T):
        y_ref = min(max(y_ref, x[k] - g), x[k] + g)
        bad += abs(y_ref - y[k]) > 1e-12
    print(f"  硬 Play 等价性：不一致点数 = {bad}（应=0）")
    assert bad == 0
    # 保持期输出冻结
    dh = np.abs(np.diff(y))
    assert (dh[hold[1:] == 1] == 0).all(), "保持期输出不应变化"
    print("  保持期输出冻结 ✓")

    # 3) PhysicsCalib 梯度流
    calib = PhysicsCalib()
    xe = torch.randn(8, 1, requires_grad=True)
    brn = torch.where(torch.randn(8, 1) > 0, 1.0, -1.0)
    gt = torch.full((8, 1), 0.3)
    out = calib.physics_end(xe, brn, gt)
    out.sum().backward()
    print(f"  γ_k={calib.gamma_k.item():.4f}(应=1), γ_b={calib.gamma_b.item():.4f}(应=1), "
          f"raw_gk.grad={calib.raw_gk.grad.item():.2e}, "
          f"raw_gb.grad={calib.raw_gb.grad.item():.2e}")
    assert abs(calib.gamma_k.item() - 1.0) < 1e-5
    assert abs(calib.gamma_b.item() - 1.0) < 1e-5
    assert calib.raw_gk.grad is not None and calib.raw_gb.grad is not None

    # 4) 损失助手
    p = torch.tensor([0.0, 1.0, 2.0])
    xl = torch.tensor([0.0, 1.0, 2.0])
    print(f"  loss_energy(无损环)={loss_energy(p, xl, 1.0).item():.4f}（应=0）")
    assert loss_energy(p, xl, 1.0).item() == 0.0
    print("✓ physics_branch 全部自测通过")


if __name__ == "__main__":
    _test()
