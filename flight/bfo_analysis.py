"""
BFO 信号序列分析 —— 差分模式识别。

不需要精确的绝对 BFO 计算（校准参数未知）。
只使用 BFO 序列的差分（一阶/二阶）提取运动学特征。

核心逻辑：
  BFO 一阶差分 = 多普勒加速度 ≈ 飞机径向速度变化率
  BFO 二阶差分 = 多普勒急动度 ≈ 飞机转弯/变速特征
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from known_data import PINGS


def bfo_sequence_analysis():
    """BFO 时间序列的差分分析"""
    print("BFO 信号序列分析 (差分模式识别)")
    print("=" * 60)

    t = np.array([p.t_hours for p in PINGS])
    bfo = np.array([p.bfo_hz for p in PINGS])
    std = np.array([p.bfo_std for p in PINGS])
    dt = np.diff(t)
    dbfo = np.diff(bfo)
    dbfo_dt = dbfo / dt  # 一阶差分：多普勒变化率 (Hz/hr)
    d2bfo = np.diff(dbfo)
    d2bfo_dt = d2bfo / dt[1:]  # 二阶差分：多普勒急动度 (Hz/hr²)

    print(f"\n{'区间':20s} {'Δt(hr)':8s} {'ΔBFO(Hz)':10s} {'BFO变化率':12s} {'解释':30s}")
    print("-" * 80)
    for i in range(len(dt)):
        direction = "远离卫星加速" if dbfo[i] < 0 else "远离卫星减速"
        magnitude = "显著" if abs(dbfo_dt[i]) > 40 else "温和" if abs(dbfo_dt[i]) > 15 else "轻微"
        rate_str = f"{magnitude}{direction}"
        interval = f"{PINGS[i].label}→{PINGS[i+1].label[:4]}"
        print(f"{interval:20s} {dt[i]:>7.2f} {dbfo[i]:>+8.1f} {dbfo_dt[i]:>+8.1f} {rate_str:30s}")

    print(f"\n{'转折点':20s} {'急动度':12s} {'物理含义':30s}")
    print("-" * 62)
    for i in range(len(d2bfo)):
        transition = f"{PINGS[i+1].label[:6]}→{PINGS[i+2].label[:4]}"
        jerk_text = "转向/变速" if abs(d2bfo_dt[i]) > 20 else "均匀运动"
        print(f"{transition:20s} {d2bfo_dt[i]:>+8.1f} {jerk_text:30s}")

    # ── 核心发现：第 7 次 ping 的异常模式 ──
    print("\n" + "=" * 60)
    print("核心发现：第 7 次 ping 异常")
    print("=" * 60)

    # 前 5 个区间（1→6th ping）：持续单调下降
    # BFO 变化率 ≈ -50 Hz/hr（稳定匀速南下）
    trend_before = np.mean(dbfo_dt[:5])
    trend_after = dbfo_dt[5]  # 6→7th ping 的 BFO 变化率
    print(f"\n  1→6 次 ping 的 BFO 平均变化率: {trend_before:+.1f} Hz/hr")
    print(f"  6→7 次 ping 的 BFO 变化率:      {trend_after:+.1f} Hz/hr")
    print(f"  变化率差: {trend_after - trend_before:+.1f} Hz/hr")

    # ΔBFO/Δt 如果从负变正，说明加速度方向改变了
    if trend_after > trend_before:
        print(f"\n  → BFO 变化率从 {trend_before:+.0f} 突变到 {trend_after:+.0f} Hz/hr")
        print(f"  → 飞机径向速度变化方向发生改变")
        print(f"  → 最可能原因：飞机在做机动（转弯/减速/下降）")

    # BFO 二阶差分异常检测
    final_jerk = d2bfo_dt[-1]
    mean_jerk = np.mean(d2bfo_dt[:-1]) if len(d2bfo_dt) > 1 else 0
    print(f"\n  末段急动度（6→7 次 ping）: {final_jerk:+.1f} Hz/hr²")
    print(f"  之前平均急动度:             {mean_jerk:+.1f} Hz/hr²")
    jerk_ratio = abs(final_jerk / mean_jerk) if abs(mean_jerk) > 1 else 999
    if jerk_ratio > 3:
        print(f"  → 末段急动度异常（{jerk_ratio:.0f}× 高于均值）")
        print(f"  → 最后时刻机动力度远超前 6 小时")

    # ── BFO 拟合分析 ──
    print("\n" + "=" * 60)
    print("BFO 时间序列模式匹配")
    print("=" * 60)

    # 假设 1：匀速飞行（BFO 线性下降）
    coeffs_linear = np.polyfit(t[:6], bfo[:6], 1)
    bfo_pred_linear = np.polyval(coeffs_linear, t)
    rmse_linear = np.sqrt(np.mean((bfo_pred_linear - bfo) ** 2))
    print(f"\n  模型 A: 匀速飞行（线性拟合，前 6 次）")
    print(f"    斜率: {coeffs_linear[0]:+.1f} Hz/hr")
    print(f"    7 次 ping RMSE: {rmse_linear:.1f} Hz")
    print(f"    第 7 次残差: {bfo[-1] - bfo_pred_linear[-1]:+.1f} Hz")
    print(f"    解释: {'第 7 次 BFO 与线性外推无显著差异' if abs(bfo[-1] - bfo_pred_linear[-1]) < 15 else '第 7 次明显偏离线性趋势'}")

    # 假设 2：匀速减速（BFO 二次下降）
    coeffs_quad = np.polyfit(t[:6], bfo[:6], 2)
    bfo_pred_quad = np.polyval(coeffs_quad, t)
    rmse_quad = np.sqrt(np.mean((bfo_pred_quad - bfo) ** 2))
    print(f"\n  模型 B: 匀减速（二次拟合，前 6 次）")
    print(f"    二次项: {coeffs_quad[0]:+.4f} Hz/hr²")
    print(f"    7 次 ping RMSE: {rmse_quad:.1f} Hz")
    print(f"    第 7 次残差: {bfo[-1] - bfo_pred_quad[-1]:+.1f} Hz")

    # 拟合质量对比
    print(f"\n  模型选择: {'线性模型更优' if rmse_linear < rmse_quad else '二次模型更优'}")

    return {
        't': t, 'bfo': bfo, 'std': std,
        'dbfo_dt': dbfo_dt, 'd2bfo_dt': d2bfo_dt,
        'trend_before': trend_before,
        'trend_after': trend_after,
        'final_jerk': final_jerk,
        'coeffs_linear': coeffs_linear,
        'coeffs_quad': coeffs_quad,
    }


if __name__ == '__main__':
    result = bfo_sequence_analysis()
