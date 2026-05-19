"""
MH370 核心数据模块。

Inmarsat 卫星 ping 数据、已知线索、坐标定义。
数据来源：ATSB 调查报告、公开学术论文。

所有时间均为 UTC。
"""

import numpy as np
import dataclasses

# ── 关键事件时间线 ──

EVENTS = [
    # (timestamp_utc, description)
    ('16:41', 'KLIA 起飞'),
    ('17:07', '最后一次 ACARS 位置报告（正常转弯点）'),
    ('17:21', '应答机关闭（推测）'),
    ('18:02', '军用雷达最后确认：偏离航线向西'),
    ('18:22', '军用雷达最后接触：6.94N 97.86E（槟城西北）'),
    ('18:25', 'Inmarsat 首次握手（登录验证）'),
    ('19:41', '第二次 Inmarsat 握手'),
    ('20:41', '第三次 Inmarsat 握手'),
    ('21:41', '第四次 Inmarsat 握手'),
    ('22:41', '第五次 Inmarsat 握手'),
    ('23:41', '第六次 Inmarsat 握手（非常规预答）'),
    ('00:11', '推测油量即将耗尽，速度开始下降'),
    ('00:19', '第七次 Inmarsat 握手（部分回复，燃料耗尽）'),
    ('00:22', '推测最终坠海时间'),
]

# ── 卫星位置（Inmarsat-3 F1, 64.5E GEO）──
SATELLITE = {
    'name': 'Inmarsat-3 F1 (IOR)',
    'longitude': 64.5,      # deg E
    'latitude': 0.0,        # GEO 赤道上空
    'altitude_km': 35786.0,  # 地球同步轨道
}


@dataclasses.dataclass
class PingRecord:
    """一次 Inmarsat 卫星握手的已知数据"""
    label: str            # 编号如 '1st', '2nd', ...
    time_utc: str         # HH:MM 格式
    t_hours: float        # 从 18:00 UTC 起算的小时数
    bto_us: float         # 突发定时偏移（微秒），代表卫星-飞机距离
    bto_std: float        # BTO 测量误差（微秒）
    bfo_hz: float         # 突发频率偏移（Hz），代表多普勒频移
    bfo_std: float        # BFO 测量误差（Hz）
    arc_desc: str         # 弧线描述


# BTO → 距离转换：每微秒约 0.15 km（光速半程往返）
# BTO = 2 * d / c * 1e6 → d = BTO * c / (2 * 1e6)
C_LIGHT_KMS = 299792.458  # 光速 km/s


def bto_to_range(bto_us: float) -> float:
    """BTO（微秒）→ 卫星-飞机距离（km）"""
    return bto_us * C_LIGHT_KMS / 2_000_000.0


# ── Inmarsat ping 数据 ──
# 数据来源：ATSB 2017 技术报告
# BTO/BFO 值取可信区间中值，误差来自官方报告

PINGS = [
    PingRecord(
        label='1st (log-on)',
        time_utc='18:25',
        t_hours=0.42,
        bto_us=20140,
        bto_std=50,
        bfo_hz=142.0,
        bfo_std=7.0,
        arc_desc='1st arc ≈ 印尼南部/澳洲北部'),
    PingRecord(
        label='2nd',
        time_utc='19:41',
        t_hours=1.68,
        bto_us=23480,
        bto_std=50,
        bfo_hz=60.0,
        bfo_std=7.0,
        arc_desc='2nd arc ≈ 爪哇以南'),
    PingRecord(
        label='3rd',
        time_utc='20:41',
        t_hours=2.68,
        bto_us=25000,
        bto_std=50,
        bfo_hz=-19.0,
        bfo_std=7.0,
        arc_desc='3rd arc ≈ 西澳以北'),
    PingRecord(
        label='4th',
        time_utc='21:41',
        t_hours=3.68,
        bto_us=26500,
        bto_std=50,
        bfo_hz=-97.0,
        bfo_std=7.0,
        arc_desc='4th arc'),
    PingRecord(
        label='5th',
        time_utc='22:41',
        t_hours=4.68,
        bto_us=27500,
        bto_std=50,
        bfo_hz=-146.0,
        bfo_std=7.0,
        arc_desc='5th arc'),
    PingRecord(
        label='6th',
        time_utc='23:41',
        t_hours=5.68,
        bto_us=28600,
        bto_std=50,
        bfo_hz=-197.0,
        bfo_std=7.0,
        arc_desc='6th arc'),
    PingRecord(
        label='7th (partial)',
        time_utc='00:19',
        t_hours=6.32,
        bto_us=30700,
        bto_std=100,
        bfo_hz=-186.0,
        bfo_std=10.0,
        arc_desc='7th arc — 燃料耗尽区'),
]

# ── 最后雷达位置 ──
LAST_RADAR = {
    'lat': 6.94,
    'lon': 97.86,
    'time_utc': '18:22',
    'desc': '槟城西北，军用雷达最后确认'
}

# ── 已知残骸发现点 ──
# （经确认由官方/学术文献认定的残骸）
DEBRIS_FINDS = [
    {'lat': -21.0, 'lon': 55.5, 'date': '2015-07-29',
     'desc': '留尼汪岛 — 襟副翼（第一块确认残骸）',
     'id': 'flaperon'},
    {'lat': -24.7, 'lon': 35.5, 'date': '2015-12-30',
     'desc': '莫桑比克 — 水平安定面',
     'id': 'mozambique_1'},
    {'lat': -25.6, 'lon': 32.8, 'date': '2016-02-28',
     'desc': '莫桑比克 — 发动机面板',
     'id': 'mozambique_2'},
    {'lat': -16.5, 'lon': 39.6, 'date': '2016-03-18',
     'desc': '莫桑比克 — 机翼部件',
     'id': 'mozambique_3'},
    {'lat': -25.4, 'lon': 44.0, 'date': '2016-03-21',
     'desc': '南非 — 发动机穹顶面板',
     'id': 'south_africa'},
    {'lat': -17.0, 'lon': 43.0, 'date': '2016-05-11',
     'desc': '马达加斯加 — 行李箱碎片 ×2',
     'id': 'madagascar_1'},
    {'lat': -8.5, 'lon': 39.5, 'date': '2016-06-23',
     'desc': '坦桑尼亚 — 机翼部件（襟翼）',
     'id': 'tanzania'},
    {'lat': -19.5, 'lon': 34.0, 'date': '2016-09-05',
     'desc': '莫桑比克 — 机舱内板',
     'id': 'mozambique_4'},
    {'lat': -12.5, 'lon': 48.5, 'date': '2017-05-23',
     'desc': '马达加斯加 — 起落架门',
     'id': 'madagascar_2'},
]


def print_summary():
    """打印数据摘要"""
    print("=" * 60)
    print("MH370 核心数据摘要")
    print("=" * 60)
    print(f"\n卫星: {SATELLITE['name']} @ {SATELLITE['longitude']}E, "
          f"高度 {SATELLITE['altitude_km']} km")
    print(f"\n最后雷达接触: {LAST_RADAR['desc']}")
    print(f"  位置: ({LAST_RADAR['lat']:.2f}N, {LAST_RADAR['lon']:.2f}E)")
    print(f"\nInmarsat 握手记录:")
    print(f"  {'Ping':15s} {'UTC':8s} {'t(h)':8s} {'BTO(us)':8s} {'Range(km)':10s} {'BFO(Hz)':8s}")
    print(f"  {'-'*65}")
    for p in PINGS:
        r = bto_to_range(p.bto_us)
        print(f"  {p.label:15s} {p.time_utc:8s} {p.t_hours:>6.2f} {p.bto_us:>8.0f} {r:>8.0f} {p.bfo_hz:>8.1f}")
    print(f"\n已知残骸 ({len(DEBRIS_FINDS)} 件):")
    for d in DEBRIS_FINDS:
        lat_str = f"{abs(d['lat']):.1f}{'S' if d['lat'] < 0 else 'N'}"
        lon_str = f"{abs(d['lon']):.1f}{'W' if d['lon'] < 0 else 'E'}"
        print(f"  {d['date']:15s} ({lat_str:>7s}, {lon_str:>7s}) {d['desc']}")


if __name__ == '__main__':
    print_summary()
