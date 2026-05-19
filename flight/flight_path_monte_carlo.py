"""
Phase 1: 蒙特卡洛飞行路径模拟。

从最后雷达位置开始，采样可能的飞行路径，
用与第 7 条弧的距离加权，输出燃料耗尽点的概率分布。

不使用 BTO 物理反算（卫星-飞机三维距离中含 ~35,786 km 卫星高度，
BTO 是差分测量非绝对距离），改用 ATSB 发布的弧线几何。
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from known_data import LAST_RADAR, DEBRIS_FINDS
from seventh_arc import SEVENTH_ARC, distance_to_arc, arc_likelihood

# ── 地球常量 ──
R_EARTH_KM = 6371.0

# ── B777-200ER 性能参数 ──
B777 = {
    'cruise_mach_min': 0.76,
    'cruise_mach_max': 0.88,
    'mach_to_knot': 667.47,
    'fuel_capacity_kg': 49100,
    'fuel_burn_rate_kg_hr': 6400,
    'max_endurance_hours': 7.5,
}

# ── 蒙特卡洛参数 ──
MC_DEFAULTS = {
    'n_samples': 100000,
    'heading_mean': 185,
    'heading_std': 15,
    'speed_mach_mean': 0.84,
    'speed_mach_std': 0.04,
    'fuel_minutes': 370,
    'fuel_std': 15,
}

# ── 搜索区域定义 ──
SEARCH_ZONES = [
    ('Rowland 2026', -24, -23),
    ('ATSB north (28-33S)', -33, -28),
    ('ATSB core (33-36S)', -36, -33),
    ('ATSB south (36-40S)', -40, -36),
    ('South of 40S', -60, -40),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine 大圆距离（km）"""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(min(1.0, a)))


def destination_from_bearing(lat, lon, bearing_deg, dist_km):
    """从起点沿大圆方位角走 dist_km"""
    d = dist_km / R_EARTH_KM
    brg = np.radians(bearing_deg)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) +
                     np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def mach_to_speed_kms(mach):
    """Mach → km/s（@35,000ft）"""
    return mach * B777['mach_to_knot'] * 1.852 / 3600


def sample_flight_path(n_samples=None, **kwargs):
    """采样 N 条飞行路径。

    从最后雷达位置出发，采样航向、速度、飞行时间，
    用第 7 条弧约束 + 地理约束计算权重。

    Returns
    -------
    dict with keys: lats, lons, headings, speeds_knots, hours, weight
    """
    if n_samples is None:
        n_samples = MC_DEFAULTS['n_samples']

    # 采样航向（度）
    headings = np.random.normal(
        kwargs.get('heading_mean', MC_DEFAULTS['heading_mean']),
        kwargs.get('heading_std', MC_DEFAULTS['heading_std']),
        n_samples
    )
    headings = np.clip(headings, 125, 245)

    # 采样巡航速度
    mach = np.random.normal(
        kwargs.get('speed_mach_mean', MC_DEFAULTS['speed_mach_mean']),
        kwargs.get('speed_mach_std', MC_DEFAULTS['speed_mach_std']),
        n_samples
    )
    mach = np.clip(mach, B777['cruise_mach_min'], B777['cruise_mach_max'])
    speeds_kms = mach_to_speed_kms(mach)

    # 采样燃料耗尽时间（分钟），从 18:22 UTC 起算
    flight_minutes = np.random.normal(
        kwargs.get('fuel_minutes', MC_DEFAULTS['fuel_minutes']),
        kwargs.get('fuel_std', MC_DEFAULTS['fuel_std']),
        n_samples
    )
    flight_minutes = np.clip(flight_minutes, 300, 420)

    # 起点
    start_lat, start_lon = LAST_RADAR['lat'], LAST_RADAR['lon']

    # 计算终点 + 权重
    end_lats = np.zeros(n_samples)
    end_lons = np.zeros(n_samples)
    weights = np.ones(n_samples)

    for i in range(n_samples):
        h = headings[i]
        dist_km = speeds_kms[i] * flight_minutes[i] * 60
        lat2, lon2 = destination_from_bearing(start_lat, start_lon, h, dist_km)
        end_lats[i] = lat2
        end_lons[i] = lon2

        # 权重：第 7 条弧距离约束
        d_arc = distance_to_arc(lat2, lon2, SEVENTH_ARC)
        w = np.exp(-0.5 * (d_arc / 150.0) ** 2)

        # 地理约束：应落在南印度洋
        if lat2 > -10:
            w *= 0.01  # 赤道以北基本不可能
        if lon2 < 50 or lon2 > 120:
            w *= 0.01  # 印度洋范围外

        weights[i] = w

    return {
        'lats': end_lats,
        'lons': end_lons,
        'headings': headings,
        'speeds_knots': mach * B777['mach_to_knot'],
        'hours': flight_minutes / 60.0,
        'weight': weights,
    }


def compute_probability_grid(lats, lons, weights,
                             res_deg=0.5,
                             lat_range=(-60, -10),
                             lon_range=(50, 120)):
    """加权样本 → 网格概率密度"""
    lat_bins = np.arange(lat_range[0], lat_range[1] + res_deg, res_deg)
    lon_bins = np.arange(lon_range[0], lon_range[1] + res_deg, res_deg)

    hist, _, _ = np.histogram2d(
        lats, lons, bins=(lat_bins, lon_bins), weights=weights)

    total = np.sum(hist)
    if total > 0:
        hist = hist / total

    grid_lats = (lat_bins[:-1] + lat_bins[1:]) / 2
    grid_lons = (lon_bins[:-1] + lon_bins[1:]) / 2

    return {
        'prob': hist,
        'lats': grid_lats,
        'lons': grid_lons,
        'lat_edges': lat_bins,
        'lon_edges': lon_bins,
    }


def run_simulation(n_samples=50000):
    """完整模拟 + 报告"""
    print("=" * 55)
    print("  MH370 蒙特卡洛飞行路径模拟 (Phase 1)")
    print(f"  采样数: {n_samples}")
    print("=" * 55)

    result = sample_flight_path(n_samples)
    grid = compute_probability_grid(
        result['lats'], result['lons'], result['weight'])

    # 统计
    w_total = np.sum(result['weight'])
    print(f"\n  有效权重和: {w_total:.2f}")
    print(f"  纬度范围: [{result['lats'].min():.1f}, {result['lats'].max():.1f}]")
    print(f"  经度范围: [{result['lons'].min():.1f}, {result['lons'].max():.1f}]")
    print(f"  航向范围: [{result['headings'].min():.1f}, {result['headings'].max():.1f}] deg")
    print(f"  飞行时间: [{result['hours'].min():.1f}, {result['hours'].max():.1f}] hrs")

    # 各区域概率质量
    print(f"\n  概率质量分布 (第 7 条弧约束):")
    print(f"  {'区域':30s} {'概率':>10s}")
    print(f"  {'-'*42}")
    for name, lat_min, lat_max in SEARCH_ZONES:
        mask = (result['lats'] >= lat_min) & (result['lats'] < lat_max)
        prob = np.sum(result['weight'][mask])
        if w_total > 0:
            prob_pct = prob / w_total * 100
        else:
            prob_pct = 0.0
        print(f"  {name:30s} {prob_pct:>6.1f}%")

    # 最高概率网格
    max_idx = np.unravel_index(np.argmax(grid['prob']), grid['prob'].shape)
    max_prob = grid['prob'][max_idx]
    if np.isfinite(max_prob) and max_prob > 0:
        lat_s = f"{abs(grid['lats'][max_idx[0]]):.1f}S"
        lon_s = f"{abs(grid['lons'][max_idx[1]]):.1f}E"
        print(f"\n  最高概率网格: ({lat_s}, {lon_s})  p={max_prob:.6f}")

    # 与已知残骸的兼容性
    print(f"\n  已知残骸 (向后兼容性):")
    for d in DEBRIS_FINDS:
        dd = distance_to_arc(d['lat'], d['lon'], SEVENTH_ARC)
        print(f"    {d['date']:15s} arc_dist={dd:.0f}km  {d['desc'][:25]}")

    return result, grid


if __name__ == '__main__':
    result, grid = run_simulation(50000)
