"""
OSCAR 卫星海表流数据加载器。

从 ERDDAP API 拉取 OSCAR（Ocean Surface Current Analysis Real-time）数据。
OSCAR 是卫星测高/散射计融合产品，1/3° 全球格点，5 天复合。

用于 MH370 残骸漂移模拟。
"""

import numpy as np
import requests
import sys, os
from datetime import datetime, timezone, timedelta

BASE_URL = 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplOscar'

# MH370 漂移模拟所需区域
# 影响区域从 ~34S 93E 向西到非洲（~30-120E, 18-45S）
DRIFT_REGION = {
    'lat_min': -45.0,
    'lat_max': -15.0,
    'lon_min': 30.0,
    'lon_max': 120.0,
}

SURFACE_DEPTH = 15.0  # OSCAR 的海表深度（固定值，m）


def query_currents(lat_min=-45, lat_max=-15,
                   lon_min=30, lon_max=120,
                   time_start='2014-03-01',
                   time_end='2016-12-31',
                   timeout=120):
    """从 ERDDAP 查询 OSCAR 海表流数据。

    Returns
    -------
    dict with 'u', 'v', 'lat', 'lon', 'time'
    """
    t0 = datetime.fromisoformat(time_start).replace(tzinfo=timezone.utc).timestamp()
    t1 = datetime.fromisoformat(time_end).replace(tzinfo=timezone.utc).timestamp()

    url = (
        f'{BASE_URL}.csv'
        f'?u[({t0:.0f}):1:({t1:.0f})][0:1:0]'
        f'[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})],'
        f'v[({t0:.0f}):1:({t1:.0f})][0:1:0]'
        f'[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]'
    )

    print(f'拉取 OSCAR 数据: {time_start} → {time_end}')
    print(f'  区域: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]')
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    print(f'  响应: {r.status_code}, {len(r.text)/1e6:.1f} MB')

    if r.status_code != 200:
        raise IOError(f'OSCAR API 返回 {r.status_code}')

    lines = r.text.strip().split('\n')
    # 跳过 CSV 头两行
    data_lines = lines[2:]

    result = {
        'u': {},
        'v': {},
    }
    count = 0
    for line in data_lines:
        parts = line.split(',')
        if len(parts) < 6:
            continue
        t_str, depth_str, lat_str, lon_str, u_val, v_val = parts[:6]
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            u = float(u_val)
            v = float(v_val)
        except ValueError:
            continue

        t_key = t_str[:10]  # 只取日期

        if t_key not in result['u']:
            result['u'][t_key] = {}
            result['v'][t_key] = {}
        key = (lat, lon)
        if key not in result['u'][t_key]:
            result['u'][t_key][key] = u
            result['v'][t_key][key] = v
            count += 1

    print(f'  解析完成: {len(result["u"])} 天, {count} 个格点')
    return result


def get_closest_current(data, lat, lon, date_str):
    """获取指定位置和日期的最接近海表流"""
    if date_str not in data.get('u', {}):
        # 找最近的日期
        all_dates = sorted(data.get('u', {}).keys())
        if not all_dates:
            return 0.0, 0.0
        date_str = min(all_dates, key=lambda d: abs(
            (datetime.fromisoformat(d) - datetime.fromisoformat(date_str)).days))

    keys = list(data['u'][date_str].keys())
    if not keys:
        return 0.0, 0.0

    # 找最近格点
    best_key = min(keys, key=lambda k: (k[0]-lat)**2 + (k[1]-lon)**2)
    return data['u'][date_str][best_key], data['v'][date_str][best_key]


def drifter_simulation(data, start_lat, start_lon, start_date,
                       duration_days=365, dt_days=5):
    """用 OSCAR 海表流数据模拟漂流轨迹。

    简单的欧拉前向积分，无风驱和 Stokes drift 修正。

    Parameters
    ----------
    data : dict
        OSCAR 数据
    start_lat, start_lon : float
        起点经纬度
    start_date : str
        ISO 格式日期
    duration_days : int
        漂流总天数
    dt_days : float
        积分步长

    Returns
    -------
    lats, lons, dates : np.ndarray
    """
    n_steps = int(duration_days / dt_days)
    lats = np.zeros(n_steps + 1)
    lons = np.zeros(n_steps + 1)
    dates = []

    lats[0], lons[0] = start_lat, start_lon
    cur_date = datetime.fromisoformat(start_date)

    for i in range(n_steps):
        date_str = cur_date.strftime('%Y-%m-%d')
        dates.append(date_str)

        u, v = get_closest_current(data, lats[i], lons[i], date_str)

        # 转换：m/s → 度/天（1° ≈ 111 km）
        DEG_PER_KM = 1.0 / 111.0
        lat_factor = np.cos(np.radians(lats[i]))
        d_lon = u * 86400 * DEG_PER_KM / lat_factor if abs(lat_factor) > 0.01 else 0
        d_lat = v * 86400 * DEG_PER_KM

        lons[i+1] = (lons[i] + d_lon * dt_days) % 360
        lats[i+1] = lats[i] + d_lat * dt_days
        cur_date += timedelta(days=dt_days)

    dates.append(cur_date.strftime('%Y-%m-%d'))
    return lats, lons, np.array(dates)


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'query'

    if mode == 'query':
        data = query_currents(
            lat_min=-45, lat_max=-15,
            lon_min=60, lon_max=120,
            time_start='2014-03-01',
            time_end='2016-12-31'
        )
        print('\n数据时间范围:')
        all_dates = sorted(data['u'].keys())
        print(f'  首日: {all_dates[0]}')
        print(f'  末日: {all_dates[-1]}')
        print(f'  天数: {len(all_dates)}')

        # 查一个点看看
        lat, lon = -34.23, 93.78  # UGIB LEP
        u, v = get_closest_current(data, lat, lon, '2014-03-10')
        print(f'\nUGIB LEP ({lat:.2f}S, {lon:.2f}E) 附近海表流:')
        print(f'  u = {u:.4f} m/s (东向)')
        print(f'  v = {v:.4f} m/s (北向)')
        speed = np.sqrt(u**2 + v**2) * 86400 / 1000  # km/天
        print(f'  合成流速 ≈ {speed:.1f} km/天')

    elif mode == 'drift':
        data = query_currents(
            lat_min=-45, lat_max=-15,
            lon_min=60, lon_max=120,
            time_start='2014-03-01',
            time_end='2016-12-31',
        )
        lats, lons, dates = drifter_simulation(
            data, -34.23, 93.78, '2014-03-08',
            duration_days=365, dt_days=5
        )
        print(f'\n一年漂流轨迹:')
        print(f'  起始: {dates[0]} ({lats[0]:.1f}S, {lons[0]:.1f}E)')
        for i in range(0, len(dates), 12):
            print(f'  第{i*5:3d}天 ({dates[i]}): ({lats[i]:.1f}S, {lons[i]:.1f}E)')
        print(f'  第{len(dates)*5-5:3d}天 ({dates[-1]}): ({lats[-1]:.1f}S, {lons[-1]:.1f}E)')
