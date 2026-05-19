"""
MH370 Pixel Hunter — Satellite Data Sources
=============================================
Multi-source satellite data acquisition for historical (2014) MH370 analysis.

API backends:
  1. NASA CMR API        — Granule discovery (no key, free, all NASA datasets)
  2. LAADS DAAC          — MODIS L1B/L2 direct download (needs Earthdata login)
  3. Worldview Snapshot  — Recent GIBS imagery (existing, bound to ~1yr window)
  4. ERDDAP              — Oceanographic / SST data
  5. GIBS WMS            — Recent data tiles (existing, bound to ~1yr window)

CMR is the primary backend for 2014 data discovery.
"""

import numpy as np
import json, os, sys, time, warnings, re
from datetime import datetime, timedelta, timezone
try:
    import urllib.request
    import urllib.parse
    import http.client
    URLLIB_OK = True
except ImportError:
    URLLIB_OK = False

warnings.filterwarnings('ignore')

# =============================================================================
# Cache
# =============================================================================

_CACHE_DIR = None

def _ensure_cache():
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR

def _cache_path(key):
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)[:120]
    return os.path.join(_ensure_cache(), f'{safe}.json')

def _cache_get(key, max_age_hours=24):
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    age = time.time() - os.path.getmtime(p)
    if age > max_age_hours * 3600:
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def _cache_set(key, data):
    p = _cache_path(key)
    try:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except:
        pass


# =============================================================================
# 1. NASA CMR API — Granule Discovery (No API Key Required)
# =============================================================================

CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
CMR_COLLECTION_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"

# Key collection shortnames for MH370 analysis
COLLECTIONS = {
    # MODIS Level 1B (reflectance)
    'MODIS_Terra_L1B': 'MOD021KM',      # 1km calibrated radiances
    'MODIS_Aqua_L1B': 'MYD021KM',
    # MODIS Cloud Product
    'MODIS_Terra_Cloud': 'MOD06_L2',
    'MODIS_Aqua_Cloud': 'MYD06_L2',
    # MODIS Surface Reflectance
    'MODIS_Terra_SR': 'MOD09GA',
    'MODIS_Aqua_SR': 'MYD09GA',
    # VIIRS
    'VIIRS_SNPP_SDR': 'VNP02IMG',
    # Landsat
    'Landsat_8_OLI': 'LANDSAT_8_OLI_TIRS_L1T',
}

def search_cmr(short_name, bbox=None, time_start=None, time_end=None,
               page_size=50, page_num=1):
    """
    Search NASA CMR for granules matching criteria.
    
    Parameters:
    -----------
    short_name : str  — e.g. 'MOD021KM', 'MOD06_L2'
    bbox : tuple      — (min_lon, min_lat, max_lon, max_lat)
    time_start : str  — ISO datetime
    time_end : str    — ISO datetime
    
    Returns:
    --------
    List of granule dicts with: title, time, bbox, download_url
    """
    if not URLLIB_OK:
        print("[CMR] urllib not available")
        return []
    
    cache_key = f'cmr_{short_name}_bbox{bbox}_t{time_start}_{time_end}'
    cached = _cache_get(cache_key, max_age_hours=12)
    if cached is not None:
        return cached
    
    params = {
        'short_name': short_name,
        'page_size': min(page_size, 2000),
        'page_num': page_num,
        'sort_key': '-start_date',
    }
    
    if bbox:
        params['bounding_box'] = f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}'
    if time_start:
        params['temporal'] = time_start
        if time_end:
            params['temporal'] += ',' + time_end
    
    url = CMR_GRANULE_URL + '?' + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MH370_Pixel_Hunter/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        granules = []
        for entry in data.get('feed', {}).get('entry', []):
            gran = {
                'title': entry.get('title', ''),
                'time': entry.get('time_end', entry.get('time_start', '')),
                'start_time': entry.get('time_start', ''),
                'end_time': entry.get('time_end', ''),
                'bbox': entry.get('bbox', None),
                'links': [l['href'] for l in entry.get('links', []) 
                         if l.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#' 
                         or '.hdf' in l.get('href', '').lower()],
                'cloud_cover': entry.get('cloud_cover', None),
                'granule_ur': entry.get('producer_granule_id', ''),
            }
            if gran['links']:
                granules.append(gran)
        
        _cache_set(cache_key, granules)
        return granules
        
    except Exception as e:
        print(f"[CMR] Search error for {short_name}: {e}")
        return []


def search_modis_in_area(bbox, date_str='2014-03-08', collection='MOD021KM'):
    """
    Quick wrapper: find MODIS granules covering bbox on a given date.
    
    Example:
    --------
    >>> granules = search_modis_in_area((92, -40, 104, -25), '2014-03-08')
    >>> len(granules)  # -> MODIS granules covering SE Indian Ocean
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    t_start = dt.strftime('%Y-%m-%dT00:00:00Z')
    t_end = (dt + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    
    return search_cmr(collection, bbox=bbox, 
                      time_start=t_start, time_end=t_end)


# =============================================================================
# 2. Worldview Snapshot API (existing, recent data only)
# =============================================================================

WORLDVIEW_URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot"
                 "?REQUEST=GetSnapshot"
                 "&CRS=EPSG:4326"
                 "&WRAP=on"
                 "&LAYERS={layer}"
                 "&FORMAT=image/png"
                 "&WIDTH={width}"
                 "&HEIGHT={height}"
                 "&BBOX={bbox}"
                 "&TIME={date_str}T00:00:00Z")

def fetch_worldview(bbox, date_str, layer='MODIS_Terra_CorrectedReflectance_TrueColor',
                    width=1600, output_dir=None):
    """
    Fetch satellite image via NASA Worldview snapshot API.
    
    NOTE: This API returns a blank image for dates > ~1 year ago.
    For 2014 data, use search_cmr() + Earthdata download instead.
    
    Returns: path to saved image or None
    """
    if not URLLIB_OK:
        return None
    
    output_dir = output_dir or _ensure_cache()
    
    # Format bbox
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    
    # Worldview snapshots default to latest if TIME is omitted
    url = WORLDVIEW_URL.format(
        layer=layer,
        width=width,
        height=int(width * (bbox[3]-bbox[1]) / (bbox[2]-bbox[0])),
        bbox=bbox_str,
        date_str=date_str
    )
    
    safe_name = f"worldview_{date_str.replace('-','')}_{layer[:20]}_{bbox[1]:.0f}_{bbox[0]:.0f}.png"
    out_path = os.path.join(output_dir, safe_name)
    
    if os.path.exists(out_path):
        return out_path
    
    print(f"[FETCH] Worldview: {layer} @ {date_str} [{bbox[0]:.1f},{bbox[1]:.1f}]")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        
        if len(data) < 1000:
            print(f"[FETCH] Warning: response too small ({len(data)} bytes) - likely blank")
            return None
        
        with open(out_path, 'wb') as f:
            f.write(data)
        return out_path
    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return None


# =============================================================================
# 3. MODIS IR Bands via Worldview (same backend, different layers)
# =============================================================================

def fetch_modis_ir(bbox, date_str, band='MODIS_Terra_Band31_Night', output_dir=None):
    """
    Fetch MODIS thermal IR band via Worldview.
    
    Common layers for contrail detection:
      - MODIS_Terra_Band31_Night  (11μm, nighttime)
      - MODIS_Terra_Band32_Night  (12μm)
      - MODIS_Terra_Brightness_Temp_Band31_Day
      - MODIS_Aqua_Band31_Night
    """
    return fetch_worldview(bbox, date_str, layer=band, output_dir=output_dir)


# =============================================================================
# 4. ERDDAP SST / Ocean Data
# =============================================================================

ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/"

def query_erddap_sst(lat, lon, date_str='2014-03-08', radius_deg=0.5):
    """
    Query MUR SST at a point via ERDDAP.
    Returns SST value in Celsius or None.
    """
    if not URLLIB_OK:
        return None
    
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    # MUR SST is daily at ~09:00 UTC (T09 for ascending node)
    date_str_t = dt.strftime('%Y-%m-%dT09:00:00Z')
    
    url = (f"{ERDDAP_URL}griddap/jplMURSST41.csv?"
           f"analysed_sst%5B({date_str_t})%5D%5B({lat})%5D%5B({lon})%5D")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MH370_Pixel_Hunter/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode('utf-8')
        
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            # Last line: "2014-03-08T12:00:00Z,-35.0,100.0,<sst>" or NaN
            parts = lines[-1].split(',')
            if len(parts) >= 4:
                sst_str = parts[-1].strip()
                if sst_str not in ('NaN', 'nan', ''):
                    sst_val = float(sst_str)
                    # MUR CSV returns Celsius; check if Kelvian
                    if 270 < sst_val < 310:  # Kelvian range
                        return sst_val - 273.15
                    elif -5 < sst_val < 45:  # Celsius range
                        return sst_val
    except Exception as e:
        pass
    
    return None


# =============================================================================
# 5. Source Availability Checker
# =============================================================================

def check_data_availability(date_str='2014-03-08', bbox=(92, -40, 104, -25)):
    """
    Probe which data sources have coverage for a given date/location.
    
    Returns dict: {source: available_status}
    """
    results = {}
    
    # Worldview (expected to fail for 2014)
    print(f"[CHECK] Testing Worldview for {date_str}...")
    try:
        path = fetch_worldview(bbox, date_str, width=800)
        if path and os.path.getsize(path) > 5000:
            results['worldview'] = 'OK'
        else:
            results['worldview'] = 'BLANK (likely no data)'
    except Exception as e:
        results['worldview'] = f'FAIL ({e})'
    
    # CMR MODIS
    print(f"[CHECK] Testing CMR MODIS for {date_str}...")
    granules = search_modis_in_area(bbox, date_str, 'MOD021KM')
    results['cmr_modis_l1b'] = f'{len(granules)} granules'
    
    granules_cloud = search_modis_in_area(bbox, date_str, 'MOD06_L2')
    results['cmr_modis_cloud'] = f'{len(granules_cloud)} granules'
    
    # ERDDAP SST
    print(f"[CHECK] Testing ERDDAP SST...")
    sst = query_erddap_sst(-35, 100, date_str)
    results['erddap_sst'] = f'{sst:.1f}C' if sst else 'FAIL'
    
    return results


# =============================================================================
# Summary
# =============================================================================

__all__ = [
    'search_cmr', 'search_modis_in_area',
    'fetch_worldview', 'fetch_modis_ir',
    'query_erddap_sst',
    'check_data_availability',
]


if __name__ == '__main__':
    print("="*60)
    print("MH370 Pixel Hunter — Data Source Check")
    print("="*60)
    
    results = check_data_availability(
        date_str='2014-03-08',
        bbox=(92, -40, 104, -25)
    )
    
    print("\nResults:")
    for src, status in results.items():
        print(f"  {src:20s}: {status}")
    print()
