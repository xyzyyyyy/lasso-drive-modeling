# -*- coding: utf-8 -*-
"""
v3 物理标定统计（训练前诊断）

输出：
1. 逐 trial F0 分布表（力传感估计，式4-16 静止段）
2. 全局标定：k̂_τ（OLS）、g0（可辨识 trial 扫描中位数）、s_loop（可辨识 trial 分支差符号）
3. hold（死区保持）占比
4. g_data（数据标定迟滞半宽）vs g_physics（论文式4-19 台架常数预测）
5. 自洽性：2g/k_τ vs 式(4-21) 换向死区
结果存 outputs/physics_stats.csv
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_CSV, TEST_CSV, INPUT_COLS, TARGET_COL, FORCE_COLS,
    OUTPUT_DIR, PHYS_STATS_CSV_PATH, ensure_output_dir,
)
from utils import load_trials_extended
from physics_branch import calibrate_from_trials, build_phys_list, lasso_physics_scalars


def stats_for(csv_path, tag):
    print("\n" + "=" * 60)
    print(f"物理标定统计 [{tag}]: {os.path.basename(csv_path)}")
    print("=" * 60)

    trials = load_trials_extended(csv_path, INPUT_COLS, TARGET_COL, FORCE_COLS)
    calib = calibrate_from_trials(trials)
    phys_list = build_phys_list(trials, calib)

    k_tau, g0, F0_med = calib['k_tau'], calib['g0'], calib['F0_med']
    print(f"\n全局标定:")
    print(f"  k̂_τ = {k_tau:.4f}")
    print(f"  g0 = {g0:.4f} N·m（可辨识 trial {calib['n_ident']}/{len(trials)} "
          f"扫描最优 g 中位数；g_ident 范围 [{calib['g_ident_list'].min():.2f}, "
          f"{calib['g_ident_list'].max():.2f}]；全体 g_opt 范围 "
          f"[{calib['g_opt_list'].min():.2f}, {calib['g_opt_list'].max():.2f}]，"
          f"散布反映率相关摩擦与晚段台架漂移）")
    print(f"  F0_med = {F0_med:.2f} N")
    print(f"  s_loop = {calib['s_loop']:+.0f}（可辨识 trial 分桶残差差中位数 "
          f"branch_gap={calib['branch_gap']:+.3f}，>0 ⇒ 加载走下线 ⇒ 耗散 A<0）")
    print(f"  自洽性 2g/k_τ vs 式(4-21)换向死区: "
          f"{calib['consistency_2g_over_kt_vs_dzrev']:.3f}")

    # 逐 trial 表
    print(f"\n{'Trial':>5} {'T':>7} {'F0(N)':>7} {'g_data':>7} {'g_phys':>7} "
          f"{'dz_rev_data':>11} {'dz_rev_phys':>11} {'hold%':>6} {'A_true':>8}")
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rows = []
    for i, ph in enumerate(phys_list):
        g_phys = lasso_physics_scalars(ph['f0'])['g']
        hold_pct = float(ph['hold'].mean() * 100)
        A_true = trapz(trials[i][1], ph['x'])
        rows.append({
            'split': tag, 'trial': i, 'T': len(trials[i][0]),
            'F0_N': round(ph['f0'], 2),
            'g_data': round(ph['g'], 4),
            'g_phys': round(g_phys, 4),
            'dz_rev_data': round(2 * ph['g'] / k_tau, 4),
            'dz_rev_phys': round(4 * ph['f0'] * 0.02 * abs(calib['phys_med']['psi']), 4),
            'hold_pct': round(hold_pct, 2),
            'loop_area_true': round(float(A_true), 4),
        })
        print(f"{i:>5} {len(trials[i][0]):>7} {ph['f0']:>7.1f} {ph['g']:>7.3f} "
              f"{g_phys:>7.3f} {2*ph['g']/k_tau:>11.3f} "
              f"{4*ph['f0']*0.02*abs(calib['phys_med']['psi']):>11.3f} "
              f"{hold_pct:>6.1f} {A_true:>8.3f}")

    # 汇总
    F0s = np.array([ph['f0'] for ph in phys_list])
    holds = np.array([ph['hold'].mean() for ph in phys_list])
    print(f"\n汇总: {len(trials)} trials, "
          f"F0 ∈ [{F0s.min():.1f}, {F0s.max():.1f}] N (中位 {np.median(F0s):.1f}), "
          f"hold 占比均值 {holds.mean()*100:.1f}%")
    return rows


def main():
    ensure_output_dir()
    all_rows = []
    all_rows += stats_for(TRAIN_CSV, "train")
    if os.path.exists(TEST_CSV):
        all_rows += stats_for(TEST_CSV, "test")

    # 存 CSV
    import csv
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(PHYS_STATS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n统计已保存: {PHYS_STATS_CSV_PATH}")


if __name__ == "__main__":
    main()
