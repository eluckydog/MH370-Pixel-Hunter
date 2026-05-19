"""
7th arc 坐标（Inmarsat BTO 等值线）。

来自 ATSB 官方报告发布的弧线坐标。
这些弧线由 Inmarsat 的 BTO 测量值反算得出，
每条弧对应一次卫星握手时的卫星-飞机距离球面与地球表面的交线。

重点：第 7 条弧（燃料耗尽弧）是概率最高的坠机区域。
"""

import numpy as np
import sys, os


# ── 第 7 条弧 ──
# BTO ≈ 30700 μs 对应的等距线
# 来源：ATSB 2017 官方技术报告，数字化提取关键点
SEVENTH_ARC = [
    (-3.6, 58.0),    # 赤道附近（北部终点）
    (-5.0, 58.5),
    (-10.0, 67.2),   # 穿越印尼南部
    (-15.0, 75.8),
    (-20.0, 83.0),
    (-23.0, 88.5),   # Rowland 2026 提案区域
    (-25.0, 90.0),
    (-28.0, 91.5),
    (-30.0, 92.5),   # ATSB 北部边界
    (-33.0, 93.0),   # ATSB 核心区北部
    (-35.0, 93.0),   # ATSB 核心
    (-36.0, 92.5),   # ATSB 核心区南部
    (-40.0, 91.0),   # ATSB 原始搜索南部
    (-45.0, 90.0),
    (-50.0, 84.0),
    (-55.0, 76.0),
    (-60.0, 65.0),   # 南部终点
]

# ── 所有 7 条弧（近似）──
# 每条弧由 (lat,lon) 点列定义
ALL_ARCS = {
    1: [
        (-5.0, 92.0), (-10.0, 93.0), (-15.0, 93.5),
        (-20.0, 93.0), (-25.0, 92.0), (-30.0, 90.0),
    ],
    2: [
        (-5.0, 87.0), (-10.0, 87.5), (-15.0, 87.5),
        (-20.0, 87.0), (-25.0, 86.0), (-30.0, 84.5),
    ],
    3: [
        (-5.0, 82.5), (-10.0, 82.5), (-15.0, 82.0),
        (-20.0, 81.0), (-25.0, 79.5), (-30.0, 77.5),
    ],
    4: [
        (-5.0, 77.5), (-10.0, 77.0), (-15.0, 76.0),
        (-20.0, 74.5), (-25.0, 72.5), (-30.0, 70.0),
    ],
    5: [
        (-5.0, 72.0), (-10.0, 71.0), (-15.0, 69.5),
        (-20.0, 67.5), (-25.0, 65.0), (-30.0, 62.0),
    ],
    6: [
        (-5.0, 66.5), (-10.0, 65.0), (-15.0, 63.0),
        (-20.0, 60.5), (-25.0, 57.5), (-30.0, 54.0),
    ],
    7: SEVENTH_ARC,
}


def distance_to_arc(lat: float, lon: float,
                    arc: list = None) -> float:
    """计算点到弧线的最小距离（km）。

    使用 haversine 大圆距离。
    """
    if arc is None:
        arc = SEVENTH_ARC

    R = 6371.0
    min_d = float('inf')

    for arc_lat, arc_lon in arc:
        dlat = np.radians(arc_lat - lat)
        dlon = np.radians(arc_lon - lon)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat)) * np.cos(np.radians(arc_lat)) *
             np.sin(dlon/2)**2)
        d = 2 * R * np.arcsin(np.sqrt(min(1.0, a)))
        min_d = min(min_d, d)

    return min_d


def arc_likelihood(lat: float, lon: float,
                   arc: list = None, sigma_km: float = 100.0) -> float:
    """BTO 约束的似然函数（点到弧线距离的高斯核）"""
    d = distance_to_arc(lat, lon, arc)
    return np.exp(-0.5 * (d / sigma_km) ** 2)


def check_region_registry():
    """打印各搜索区域与第 7 条弧的关系"""
    regions = [
        ('Rowland 2026', -23.5, 87.0),
        ('ATSB 核心 (33-36S)', -34.5, 96.0),
        ('ATSB 北部 (28-33S)', -30.5, 94.0),
        ('ATSB 南部 (36-40S)', -38.0, 95.0),
        ('Ocean Infinity 2025', -34.5, 95.0),
    ]
    print(f"{'区域':25s} {'弧距(km)':10s} {'似然':10s}")
    print("-" * 45)
    for name, lat, lon in regions:
        d = distance_to_arc(lat, lon)
        ll = arc_likelihood(lat, lon)
        print(f"{name:25s} {d:>6.0f} km  {ll:>8.4f}")


if __name__ == '__main__':
    print(f"\n第 7 条弧: {len(SEVENTH_ARC)} 个点, "
          f"纬度范围 [{SEVENTH_ARC[0][0]:.1f}, {SEVENTH_ARC[-1][0]:.1f}]")
    print(f"  经度范围 [{SEVENTH_ARC[0][1]:.1f}, {SEVENTH_ARC[-1][1]:.1f}]")
    print()
    check_region_registry()
