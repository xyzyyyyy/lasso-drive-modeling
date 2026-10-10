# -*- coding: utf-8 -*-
"""
PI+GRU 耦合迟滞模型 - 全局配置
论文复现：套索驱动外骨骼迟滞建模
"""
import os

# ============================================================
# 路径配置（相对本文件所在目录）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "data_traindata.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "data_testdata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ============================================================
# CSV 列索引（0-based，对应 dataset.py 输出格式）
# ============================================================
COL_TIME = 0      # Time (s)
COL_QM = 1        # Motor1 Position (rad)  ← q_m 电机角度
COL_QDM = 2       # Motor1 Velocity (rad/s) ← q̇_m 电机角速度
COL_TAUM = 3      # Motor1 Torque (N·m)    ← τ_m 电机输入力矩
COL_TAUL = 11     # Load Cell Torque (N·m) ← τ_l 输出力矩（预测目标）

# 论文要求的三通道输入：[τ_m, q_m, q̇_m]
INPUT_COLS = [COL_TAUM, COL_QM, COL_QDM]
TARGET_COL = COL_TAUL

# ============================================================
# 模型超参（严格按论文）
# ============================================================
WINDOW_SIZE = 20          # 滑窗长度（200Hz × 0.1s）
SPLIT_RATIO = 0.8         # 训练/内部测试切分
VAL_RATIO = 0.10          # 验证集占训练集比例（时序末尾切）

N_CHANNELS = 3            # 输入通道数（τ_m, q_m, q̇_m）
N_OPS_PER_CHAN = 10       # 每通道 Soft-Play 算子数（论文设定）
HIDDEN_SIZE = 64          # GRU 隐藏单元数（论文设定）
NUM_LAYERS = 2            # GRU 层数（论文设定）
FC_HIDDEN = 64            # 全连接中间层维度

# Soft-Play 参数
BETA_INIT = 10.0          # 平滑系数 β（β→∞ 退化为硬 Play，初始较大保持迟滞锐度）
R_INIT_MAX = 1.0          # 阈值 r 初始化上界（[0, R_INIT_MAX] 均匀分布）
EPS_LOG = 1e-12           # logsumexp 数值稳定常数

# ============================================================
# 训练超参（严格按论文）
# ============================================================
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
SEED = 42
DEVICE = "cpu"            # CPU-only 环境

# ============================================================
# 第二版改进：正则化 + 早停 + trial分组切分
# ============================================================
# trial 分组切分（替代原时序切分，防止同trial泄漏）
VAL_TRIAL_RATIO = 0.2     # 验证集 trial 占比（55 trial → 11 个验证）

# 正则化
DROPOUT = 0.1             # GRU 层间 dropout
WEIGHT_DECAY = 1e-4       # Adam 权重衰减

# 早停
EARLY_STOP_PATIENCE = 10  # 验证集连续10个epoch无改善则停止

# 策略 B：r/β 延迟更新
FEATURE_REFRESH_EPOCHS = 5    # 每 5 epoch 重算 Soft-Play 特征
SOFTPLAY_UPDATE_STEPS = 3     # 每 5 epoch 在小批量上做几步 r/β 梯度更新
SOFTPLAY_UPDATE_BATCH = 4096  # r/β 更新时使用的小批量样本数

# 梯度裁剪（防止 β 过大梯度爆炸）
GRAD_CLIP_NORM = 5.0

# ============================================================
# 第三版（v3）：白箱物理主干 + PI+GRU 残差 + 物理约束
# ============================================================
# 力传感器列（Load Cell1/2 Force, N）——逐 trial F0 估计的数据源
FORCE_COLS = [9, 10]

# 物理约束损失权重
LAMBDA_BOUND = 0.1         # λ1: 边界约束（预测须落在加载/卸载线之间，式4-19/4-23）
LAMBDA_DEADZONE = 0.5      # λ2: 死区保持约束（hold 窗残差→0，直击换向 MAXE 病灶）
LAMBDA_ENERGY = 0.05       # λ3: 耗散约束（迟滞环面积符号，仅 r/β/γ 更新阶段）

# λ warmup：前 3 epoch 纯 MSE，epoch 10 线性升至全权重
PHYS_WARMUP_EPOCHS = 3
PHYS_WARMUP_FULL = 10

# v3 训练超参（残差学习起点为标定白箱，用更小 lr + 更长耐心）
# 第一轮教训：best val 出现在 epoch 2，后续波谷（epoch 11-12，0.295）距其仅
# 差 0.4% 时 patience 15 已耗尽 → 模型欠训练（残差头只训了 3 个 epoch）。
# patience 提至 30（约 +2.5h），确保 warmup（epoch 10）后仍有充分改进机会。
LR_V3 = 3e-4
EARLY_STOP_PATIENCE_V3 = 30
# 第三轮（v3.1）：残差头提速 + 余弦衰减。第二轮证明 patience 不是瓶颈
# （30 epoch 未超 epoch2 的 0.2938，v2 best 也是 epoch3/val 0.2930——共同
# 域漂移天花板）；差异在于 v3 的 best 存档时残差头只训了 2 epoch（LR 3e-4、
# fc2 零初始化），而 v2 的 best 是 LR 1e-3 充分训练的 → 残差头 LR 提至 1e-3
# （与 v2 相同），余弦衰减至 1e-4；r/β/γ 物理参数仍小步慢调不变。
LR_V3_MAIN = 1e-3
LR_V3_MIN = 1e-4

# v3 输出文件名
BEST_MODEL_V3_PATH = os.path.join(OUTPUT_DIR, "best_pi_gru_v3.pth")
FINAL_MODEL_V3_PATH = os.path.join(OUTPUT_DIR, "final_pi_gru_v3.pth")
LOSS_PLOT_V3_PATH = os.path.join(OUTPUT_DIR, "training_loss_v3.png")
PREDICTION_PLOT_V3_PATH = os.path.join(OUTPUT_DIR, "prediction_vs_truth_v3.png")
HYSTERESIS_PLOT_V3_PATH = os.path.join(OUTPUT_DIR, "hysteresis_loop_v3.png")
PHYS_STATS_CSV_PATH = os.path.join(OUTPUT_DIR, "physics_stats.csv")

# ============================================================
# 输出文件名
# ============================================================
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_pi_gru.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_pi_gru.pth")
BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_gru.pth")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss.png")
PREDICTION_PLOT_PATH = os.path.join(OUTPUT_DIR, "prediction_vs_truth.png")
HYSTERESIS_PLOT_PATH = os.path.join(OUTPUT_DIR, "hysteresis_loop.png")
BASELINE_PLOT_PATH = os.path.join(OUTPUT_DIR, "baseline_prediction.png")


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
