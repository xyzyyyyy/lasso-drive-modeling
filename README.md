# PI+GRU 耦合迟滞模型论文复现

> **当前分支：`v3-graybox-softplay`（第三版 · 灰箱模型）**
> 白箱 Soft-Play 物理主干（k_τ OLS 标定 + g 逐 trial 缩放）+ PI+GRU 残差头（fc2 零初始化）+ 三条物理约束损失。
>
> **版本切换：**
> | 分支 | 版本 | 架构 |
> |------|------|------|
> | `v2-pi-gru`（= `main`） | 第二版 | 纯数据驱动：PI 算子库 + GRU |
> | `v3-graybox-softplay` | 第三版 | 灰箱：白箱物理主干 + PI+GRU 残差 |

按论文文档严格复现 PI+GRU 端到端可微迟滞建模。

## 目录结构

```
PI_GRU_SoftPlay/
├── data/                       # 训练与测试数据
│   ├── data_traindata.csv      # 168MB 训练集
│   └── data_testdata.csv       # 35MB 测试集
├── config.py                   # 全局配置
├── soft_play.py                # 可微 Soft-Play 算子（核心）
├── dataset.py                  # 数据集与特征预计算
├── model.py                    # PI+GRU 完整模型
├── train.py                    # 训练脚本（策略 B）
├── test.py                     # 测试与可视化
├── baseline_gru.py             # 纯 GRU 公平基线
├── utils.py                    # 工具函数
├── outputs/                    # 训练产物
└── README.md
```

## 快速开始

### 1. 验证模块单元测试

```bash
python soft_play.py     # Soft-Play 算子可微性测试
python dataset.py       # 数据集记忆连续性测试
python model.py         # 模型前向反传测试
```

### 2. 训练 Baseline GRU（公平基线）

```bash
python baseline_gru.py                          # 全量训练 100 epoch
python baseline_gru.py --subset 100000 --epochs 20   # 调试：子集 + 少 epoch
```

### 3. 训练 PI+GRU（论文模型）

```bash
python train.py                                # 全量训练 100 epoch（策略 B）
python train.py --quick                         # 快速测试（5万行, 5 epoch）
python train.py --subset 100000 --epochs 20     # 子集调试
```

### 4. 测试评估与可视化

```bash
python test.py                  # 评估 + 自动对比 baseline
python test.py --no-baseline    # 仅评估 PI+GRU
```

## 核心设计

### 1. 可微 Soft-Play 算子

数学定义（论文 β 约定：β→∞ 退化为硬 Play）：

```
soft_max(a, b; β) = (1/β) · log( exp(βa) + exp(βb) + ε )
soft_min(a, b; β) = -soft_max(-a, -b; β)
soft_clamp(z, low, high; β) = soft_min( soft_max(z, low; β), high; β )
Soft-Play:  y[k] = soft_clamp( y[k-1], x[k]-r, x[k]+r; β )
```

- 用 `torch.logsumexp` 保证数值稳定
- r 用 `softplus(raw_r)` 保证非负，作为 `nn.Parameter` 可训练
- β 同样可训练，初始值 10.0

### 2. 逐通道建模

对 τ_m / q_m / q̇_m 三通道各算 10 个 Soft-Play 算子，输出 (B, T, 30) 全局迟滞特征向量。

### 3. 训练策略 B（CPU 性能权衡）

CPU-only + 1.4M 样本 + Soft-Play 串行循环 = 性能瓶颈：

- 每 5 epoch 用当前 r/β 重算特征并缓存（`torch.no_grad()`），GRU 每步反传
- 每 5 epoch 在小批量上重算带 grad 的特征做 3 步 r/β 梯度更新
- 90% 训练步只需前向缓存特征，10× 加速

### 4. 全段预计算保留记忆

Soft-Play 算子有时序记忆性（y[k] 依赖 y[k-1]），在 `__init__` 时对整段连续信号预计算特征，避免窗口边界状态断裂。

## 论文复现目标

| 指标 | 现有 PI+GRU | 论文报告 | 本方案目标 |
|------|------------|---------|-----------|
| RMSE | 0.582 | 0.2216 | ≤ 0.25 |
| R² | — | 0.9872 | > 0.95 |
| MAXE | — | 0.4309 | < 1.0 |

## 与原 Hysteresis 项目的差异

| 项 | 原 Hysteresis 代码 | 本方案 |
|----|---------------------|--------|
| 迟滞算子 | 经典 Play（不可微，离线计算） | Soft-Play（可微，端到端训练） |
| 通道处理 | 单通道 | 三通道逐通道建模 |
| 输入列 | `[1,2,3,7,8]` 5 列 | `[3,1,2]` 3 列（τ_m, q_m, q̇_m） |
| GRU 隐藏 | 32 | 64（论文设定） |
| FC 中间层 | 无 | ReLU + 64 维中间层 |
| β 约定 | tau→0 退化为 max（反向） | β→∞ 退化为硬 Play（论文约定） |

## 依赖

- Python 3.10+
- PyTorch 2.0+（CPU-only 亦可）
- numpy, pandas, matplotlib
