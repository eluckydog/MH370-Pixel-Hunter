"""
SkyPrint Agent — Contrail & Anomaly Screening Pipeline for Satellite Imagery
=============================================================================
An open-source tool agent for human-in-the-loop satellite anomaly detection.

Pipeline stages:
  1. Image Acquisition  — NASA GIBS auto-download (given bbox + time window)
  2. Coarse Screening   — Radon transform linear feature extraction (~16 candidates/frame)
  3. Advanced Discriminators:
     a. Texture Entropy     — gradient orientation consistency along candidate line
     b. Parallel Track Pair — find paired lines with contrail-typical separation
     c. Multi-Temporal Diff — compare same region at different times
  4. Scoring & Ranking     — weighted ensemble for human review priority
  5. Review Package        — annotated PNGs + JSON report + ranked candidate list

Design philosophy:
  - NOT a fully automatic classifier (that doesn't work for contrails)
  - A screening pipeline that reduces ~10^6 km^2 per frame to ~16 candidates
  - Human analyst reviews annotated output in ~30 seconds per frame
  - Modular: each discriminator can be enabled/disabled independently

Usage:
  python skyprint_agent.py --bbox "31S,38S,92E,104E" --date 2014-03-08 \
      --window 7 --output review_package/

Author: math-science agent
Date: 2026-05-20
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Ellipse, Polygon
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, sobel
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import map_coordinates
from skimage.transform import radon, iradon
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
import json
import os
import sys
import time
import argparse
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


def _get_endpoints(cand):
    """Get (x1, y1, x2, y2) from a candidate dict, handling both list and dict formats."""
    if 'endpoints' not in cand:
        return None
    ep = cand['endpoints']
    if isinstance(ep, (list, tuple)) and len(ep) == 4:
        return float(ep[0]), float(ep[1]), float(ep[2]), float(ep[3])
    if isinstance(ep, dict):
        return float(ep.get('x1', 0)), float(ep.get('y1', 0)), float(ep.get('x2', 0)), float(ep.get('y2', 0))
    return None

# === Import existing contrail detector for Radon screening stage ===
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from contrail_detector import (
    ContrailDetector, AtmosphericModel, ContrailPhysics,
    SensorModel, fetch_worldview_image
)

# =============================================================================
# Stage 1: Image Acquisition
# =============================================================================

class GIBSImageAcquirer:
    """
    Automated image acquisition from NASA GIBS.
    
    Handles:
    - Bbox parsing (various formats)
    - Time window iteration
    - Multi-sensor support (MODIS Terra/Aqua, VIIRS)
    - Automatic retry with backoff
    - Cache management
    """
    
    GIBS_URL = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                "?service=WMS&version=1.3.0&request=GetMap"
                "&format=image/png&transparent=false"
                "&crs=EPSG:4326"
                "&layers={layer}"
                "&styles=default"
                "&bbox={bbox}"
                "&width={width}&height={height}"
                "&time={time}")
    
    LAYERS = {
        'MODIS_Terra_TrueColor': 'MODIS_Terra_CorrectedReflectance_TrueColor',
        'MODIS_Aqua_TrueColor': 'MODIS_Aqua_CorrectedReflectance_TrueColor',
        'VIIRS_TrueColor': 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
        'MODIS_Terra_Bands721': 'MODIS_Terra_CorrectedReflectance_Bands721',
        'MODIS_Aqua_Bands721': 'MODIS_Aqua_CorrectedReflectance_Bands721',
    }
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), 'cache', 'gibs')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @staticmethod
    def parse_bbox(bbox_str):
        """
        Parse bbox string to (min_lat, max_lat, min_lon, max_lon).
        
        Accepts: "31S,38S,92E,104E" or "-31,-38,92,104" or "-31, -38, 92, 104"
        Returns: (min_lat: float, max_lat: float, min_lon: float, max_lon: float)
        """
        parts = [p.strip() for p in bbox_str.split(',')]
        if len(parts) != 4:
            raise ValueError(f"Need 4 comma-separated values, got: {bbox_str}")
        
        def parse_coord(s):
            s = s.strip().upper()
            if 'S' in s:
                return -float(s.replace('S', ''))
            elif 'N' in s:
                return float(s.replace('N', ''))
            elif 'W' in s:
                return -float(s.replace('W', ''))
            elif 'E' in s:
                return float(s.replace('E', ''))
            else:
                return float(s)
        
        lat1 = parse_coord(parts[0])
        lat2 = parse_coord(parts[1])
        lon1 = parse_coord(parts[2])
        lon2 = parse_coord(parts[3])
        
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)
        
        return min_lat, max_lat, min_lon, max_lon
    
    def acquire(self, bbox_str, date_str, layer='MODIS_Terra_TrueColor',
                width=1200, retries=3):
        """
        Download a single image from GIBS.
        
        Returns (image_array, filepath) or (None, None) on failure.
        """
        min_lat, max_lat, min_lon, max_lon = self.parse_bbox(bbox_str)
        
        # Compute height to maintain aspect ratio
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        if lon_range > 0:
            height = int(width * lat_range / lon_range)
        else:
            height = width
        height = max(height, 200)
        
        # Build bbox string for WMS (min_lon, min_lat, max_lon, max_lat)
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        
        layer_key = self.LAYERS.get(layer, layer)
        url = self.GIBS_URL.format(
            layer=layer_key, bbox=bbox, width=width, height=height, time=date_str)
        
        cache_name = f"{layer}_{date_str}_{min_lat}_{max_lat}_{min_lon}_{max_lon}_{width}.png"
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # Check cache
        if os.path.exists(cache_path):
            from matplotlib import image as mpimg
            img = mpimg.imread(cache_path)
            return img, cache_path
        
        # Download
        import urllib.request
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                
                with open(cache_path, 'wb') as f:
                    f.write(data)
                
                from matplotlib import image as mpimg
                img = mpimg.imread(cache_path)
                
                # Check if all-black (no data for this time)
                if img.ndim >= 2 and img.shape[-1] >= 3:
                    rgb_max = np.max(img[:, :, :3])
                elif img.ndim >= 2:
                    rgb_max = np.max(img)
                else:
                    rgb_max = 0
                
                if rgb_max < 0.01:
                    os.remove(cache_path)
                    return None, None
                
                return img, cache_path
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [WARN] Failed to acquire {date_str}: {e}")
                    return None, None
        
        return None, None
    
    def acquire_window(self, bbox_str, start_date, end_date, step_days=1,
                       layer='MODIS_Terra_TrueColor', width=1200):
        """
        Acquire images over a time window.
        
        Returns list of (image_array, filepath, date_str) tuples.
        """
        results = []
        
        if isinstance(start_date, str):
            dt = datetime.strptime(start_date, '%Y-%m-%d')
            edt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            dt = start_date
            edt = end_date
        
        current = dt
        n_total = (edt - dt).days // step_days + 1
        n_success = 0
        
        print(f"Acquiring {n_total} images ({dt.date()} to {edt.date()})...")
        
        for i in range(n_total):
            date_str = current.strftime('%Y-%m-%d')
            img, path = self.acquire(bbox_str, date_str, layer, width)
            
            if img is not None:
                results.append((img, path, date_str))
                n_success += 1
                print(f"  [{i+1}/{n_total}] {date_str} OK")
            else:
                print(f"  [{i+1}/{n_total}] {date_str} SKIP (no data)")
            
            current += timedelta(days=step_days)
        
        print(f"Acquired {n_success}/{n_total} images")
        return results


# =============================================================================
# Stage 2: Coarse Screening (Radon-based, reuse existing ContrailDetector)
# =============================================================================

class CoarseScreener:
    """
    Wraps ContrailDetector for coarse linear feature extraction.
    
    Returns raw Radon peak candidates before classification,
    because the advanced discriminators will handle classification.
    """
    
    def __init__(self, sensor_name='MODIS_Terra'):
        self.detector = ContrailDetector(sensor_name=sensor_name)
    
    def _prepare_image(self, image):
        """Normalize image and set detector state. Returns working image."""
        img_work = image.astype(np.float64) if image.dtype != np.float64 else image.copy()
        if img_work.max() > 1.0:
            img_work = img_work / 255.0 if img_work.max() <= 255 else img_work / 65535.0
        if img_work.shape[-1] == 4:
            img_work = img_work[:, :, :3]
        self.detector.original = img_work.copy()
        self.detector.image = img_work
        self.detector.height, self.detector.width = img_work.shape[:2]
        return img_work
    
    def screen(self, image, min_peak_prominence=0.15):
        """
        Extract raw Radon peak candidates from an image.
        
        Returns list of dicts with: rho, angle_deg, prominence, line_profile, endpoints
        """
        img_work = self._prepare_image(image)
        self.detector.preprocess()
        
        # Run Radon detection and return raw candidates
        detections = self.detector.detect_linear_features_radon(
            theta_range=(-5, 95), n_theta=180,
            threshold_sigma=2.0
        )
        
        # Classify to get endpoints etc
        classified = self.detector.classify_contrails(detections)
        
        return classified
    
    def screen_raw_radon(self, image):
        """
        Get raw Radon transform and sinogram for advanced analysis.
        """
        img_work = self._prepare_image(image)
        self.detector.preprocess()
        
        # Run Radon
        gray = self.detector.gray
        theta_deg = np.linspace(-5, 95, 180)
        theta_rad = np.deg2rad(theta_deg)
        sinogram = radon(gray, theta=theta_deg, circle=True)
        
        # Find peaks
        from scipy.ndimage import maximum_filter
        prominence_thresh = 0.15 * np.max(sinogram)
        
        local_max = maximum_filter(sinogram, size=(5, 5))
        peaks_mask = (sinogram == local_max) & (sinogram > prominence_thresh)
        peak_coords = np.argwhere(peaks_mask)
        
        peaks = []
        for rho_idx, theta_idx in peak_coords:
            peaks.append({
                'rho_idx': int(rho_idx),
                'theta_idx': int(theta_idx),
                'rho': sinogram.shape[0] // 2 - rho_idx,
                'angle_deg': theta_deg[theta_idx],
                'prominence': float(sinogram[rho_idx, theta_idx]),
                'sinogram': sinogram,
                'theta_deg_arr': theta_deg,
            })
        
        # Sort by prominence
        peaks.sort(key=lambda p: p['prominence'], reverse=True)
        return peaks[:30], sinogram, theta_deg


# =============================================================================
# Stage 3a: Texture Entropy Discriminator
# =============================================================================

class TextureEntropyDiscriminator:
    """
    Analyzes gradient orientation consistency along candidate line.
    
    Physical basis:
    - Contrails are smooth, straight ice-crystal lines
    - Cloud edges have chaotic, multi-directional texture
    - Gradient vectors along a contrail should point CONSISTENTLY
      perpendicular to the line (bright line against dark background)
    - Cloud boundaries have randomly oriented gradients
    
    Metric: orientation entropy of gradient vectors in a band along the line
    Lower entropy = more contrail-like (consistent edge orientation)
    """
    
    def __init__(self, gradient_sigma=1.5, band_half_width=15):
        self.gradient_sigma = gradient_sigma
        self.band_half_width = band_half_width
    
    def compute_gradient_field(self, image):
        """Compute gradient orientation field from grayscale image."""
        if image.ndim == 3:
            gray = np.mean(image[:, :, :3], axis=2)
        else:
            gray = image
        
        # Smooth to reduce noise
        smoothed = gaussian_filter(gray.astype(float), sigma=self.gradient_sigma)
        
        # Sobel gradients
        Gx = sobel(smoothed, axis=1)
        Gy = sobel(smoothed, axis=0)
        
        # Magnitude and orientation
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx)  # [-pi, pi]
        
        return magnitude, orientation, Gx, Gy
    
    def sample_along_line(self, x1, y1, x2, y2, orientation_field, magnitude_field,
                          n_samples=30, n_orthogonal=10):
        """
        Sample gradient orientations in a band along the line.
        
        For each sample point along the line, sample n_orthogonal points
        perpendicular to the line (forming a band).
        
        Returns orientations sampled in the band.
        """
        H, W = orientation_field.shape
        
        # Line direction vector
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx**2 + dy**2)
        
        if line_length < 1:
            return [], []
        
        # Unit direction
        ux, uy = dx / line_length, dy / line_length
        
        # Unit perpendicular (orthogonal to line, pointing "left")
        px, py = -uy, ux
        
        orientations = []
        magnitudes = []
        
        for i in range(n_samples):
            # Point along the line
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            cx = x1 + t * dx
            cy = y1 + t * dy
            
            # Sample orthogonal band
            for j in range(n_orthogonal):
                offset = (j - n_orthogonal/2) * (2 * self.band_half_width / n_orthogonal)
                sx = int(cx + px * offset)
                sy = int(cy + py * offset)
                
                if 0 <= sx < W and 0 <= sy < H:
                    orientations.append(orientation_field[sy, sx])
                    magnitudes.append(magnitude_field[sy, sx])
        
        return orientations, magnitudes
    
    def orientation_entropy(self, orientations, magnitudes, n_bins=36):
        """
        Compute Shannon entropy of gradient orientation distribution.
        
        Lower entropy = more consistent edge direction = more contrail-like.
        
        Returns entropy value (0 = perfectly aligned, ~3.5 = random).
        """
        if len(orientations) < 10:
            return 3.5, 0.0
        
        orientations = np.array(orientations)
        magnitudes = np.array(magnitudes)
        
        # Weight by magnitude (strong gradients matter more)
        weights = magnitudes / (np.sum(magnitudes) + 1e-10)
        
        # Normalize orientations to [0, pi) for edge direction (undirected)
        orientations_mod = np.mod(orientations, np.pi)
        
        # Histogram weighted by gradient magnitude
        hist, _ = np.histogram(orientations_mod, bins=n_bins, range=(0, np.pi),
                               weights=weights)
        
        # Normalize
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Max entropy for uniform distribution = log(n_bins)
        max_entropy = np.log(n_bins)
        
        # Normalized entropy [0, 1], 0 = perfectly ordered
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Kurtosis of gradient magnitudes along the line (should be peaked for contrails)
        if len(magnitudes) > 3:
            mag_mean = np.mean(magnitudes)
            mag_std = np.std(magnitudes)
            if mag_std > 1e-6:
                kurtosis = np.mean(((magnitudes - mag_mean) / mag_std)**4) - 3
            else:
                kurtosis = 0
        else:
            kurtosis = 0
        
        return norm_entropy, kurtosis
    
    def score(self, image, candidate):
        """
        Score a candidate line for contrail-likeness based on texture entropy.
        
        Returns (score_0_to_1, entropy_value, kurtosis_value)
        """
        magnitude, orientation, _, _ = self.compute_gradient_field(image)
        
        # Get endpoints
        ep = _get_endpoints(candidate)
        if ep is None and 'x1' in candidate:
            ep = (candidate['x1'], candidate['y1'], candidate['x2'], candidate['y2'])
        if ep is None:
            return 0.2, 3.5, 0.0
        x1, y1, x2, y2 = ep
        
        orientations, mags = self.sample_along_line(
            x1, y1, x2, y2, orientation, magnitude)
        
        entropy, kurtosis = self.orientation_entropy(orientations, mags)
        
        # Convert to score: low entropy = high score
        score = np.clip(1.0 - entropy, 0.0, 1.0)
        
        return score, entropy, kurtosis


# =============================================================================
# Stage 3b: Parallel Track Pair Discriminator
# =============================================================================

class ParallelTrackDiscriminator:
    """
    Find paired linear features with contrail-typical geometry.
    
    Physical basis:
    - Aircraft contrails often appear in pairs (2 engines → 2 parallel trails)
    - Even from 4-engine aircraft, the paired structure is visible
    - Contrails spread: initial separation ~30-50m (engine spacing)
    - After diffusion aging: apparent width ~1-5km in satellite imagery
    - Paired structure: two near-parallel lines separated by 1-5km
    
    Cloud boundaries:
    - May have multiple edges, but not at consistent 1-5km parallel separation
    - Random orientation and spacing
    
    Metric: find pairs of Radon candidates with:
    - |angle1 - angle2| < 5 degrees (near-parallel)
    - 1km < separation < 5km (contrail-typical)
    - Similar prominence (companion line should also be strong)
    """
    
    def __init__(self, angle_tolerance_deg=5.0, min_sep_km=1.0, max_sep_km=8.0):
        self.angle_tolerance = angle_tolerance_deg
        self.min_sep = min_sep_km
        self.max_sep = max_sep_km
    
    def line_from_candidate(self, cand):
        """Extract (x1, y1, x2, y2, angle_deg) from candidate dict."""
        ep = _get_endpoints(cand)
        if ep is None and 'x1' in cand:
            ep = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
        if ep is None:
            return None
        x1, y1, x2, y2 = ep
        
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(-dy, dx))  # y flipped in image coords
        angle = angle % 180
        
        return x1, y1, x2, y2, angle
    
    def line_separation_distance(self, line1, line2, image_width_px, image_coverage_km):
        """
        Compute perpendicular separation between two nearly-parallel lines.
        
        Approximates distance in km using image scale.
        """
        x1a, y1a, x2a, y2a, ang_a = line1
        x1b, y1b, x2b, y2b, ang_b = line2
        
        # Use midpoint of first line
        mid_x = (x1a + x2a) / 2
        mid_y = (y1a + y2a) / 2
        
        # Direction of second line
        dx_b = x2b - x1b
        dy_b = y2b - y1b
        len_b = np.sqrt(dx_b**2 + dy_b**2)
        
        if len_b < 1:
            return 999
        
        # Unit normal to second line
        nx = -dy_b / len_b
        ny = dx_b / len_b
        
        # Distance from midpoint of line1 to line2
        dist_px = abs(nx * (mid_x - x1b) + ny * (mid_y - y1b))
        
        # Convert to km
        km_per_px = image_coverage_km / image_width_px
        dist_km = dist_px * km_per_px
        
        return dist_km
    
    def find_pairs(self, candidates, image_width_px=None, image_coverage_km=None):
        """
        Find parallel pairs among candidates.
        
        Returns list of (idx1, idx2, angle_diff, separation_km, pair_score)
        """
        if image_width_px is None:
            image_width_px = 1200
        if image_coverage_km is None:
            image_coverage_km = 500  # Default ~500km coverage
        
        lines = []
        valid_indices = []
        
        for i, cand in enumerate(candidates):
            line = self.line_from_candidate(cand)
            if line is not None:
                lines.append(line)
                valid_indices.append(i)
        
        pairs = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                ang_diff = abs(lines[i][4] - lines[j][4])
                if ang_diff > 90:
                    ang_diff = 180 - ang_diff
                
                if ang_diff > self.angle_tolerance:
                    continue
                
                sep = self.line_separation_distance(
                    lines[i], lines[j], image_width_px, image_coverage_km)
                
                if sep < self.min_sep or sep > self.max_sep:
                    continue
                
                # Pair score: closer to ideal separation (2-3km) is better
                ideal_sep = 2.5
                sep_score = 1.0 - abs(sep - ideal_sep) / max(self.min_sep, self.max_sep - ideal_sep)
                sep_score = max(0, min(1, sep_score))
                
                # Angle score
                ang_score = 1.0 - ang_diff / self.angle_tolerance
                ang_score = max(0, min(1, ang_score))
                
                pair_score = 0.5 * sep_score + 0.5 * ang_score
                
                pairs.append((
                    valid_indices[i], valid_indices[j],
                    ang_diff, sep, pair_score
                ))
        
        pairs.sort(key=lambda p: p[4], reverse=True)
        return pairs
    
    def score_candidates(self, candidates, pairs, image_width_px=None,
                         image_coverage_km=None):
        """
        Boost scores for candidates that form parallel pairs.
        
        Returns updated candidate list with pair_score field.
        """
        # Mark which candidates are in pairs
        paired_indices = set()
        pair_scores = defaultdict(float)
        
        for i1, i2, ang_diff, sep, pair_score in pairs:
            paired_indices.add(i1)
            paired_indices.add(i2)
            pair_scores[i1] = max(pair_scores[i1], pair_score)
            pair_scores[i2] = max(pair_scores[i2], pair_score)
        
        # Update candidates
        for i, cand in enumerate(candidates):
            if i in paired_indices:
                cand['in_pair'] = True
                cand['pair_score'] = pair_scores[i]
            else:
                cand['in_pair'] = False
                cand['pair_score'] = 0.0
        
        return candidates, pairs


# =============================================================================
# Stage 3c: Multi-Temporal Difference Discriminator
# =============================================================================

class MultiTemporalDiscriminator:
    """
    Compare same region at different times to isolate transient features.
    
    Physical basis:
    - Contrails: transient (hours), appear in one image, gone in another
    - Cloud edges: persistent or slow-moving
    - Coastlines / terrain: stationary across all times
    - Ocean: relatively stable brightness
    
    Pipeline:
    1. For each candidate in image_t1, extract pixel values along the line
    2. Extract values at the same spatial location from image_t2
    3. Compute difference: if it's a contrail, difference should be large
       (contrail in t1, none in t2 → large |delta|)
    4. If multiple time points available, compute temporal variance
    """
    
    def __init__(self, min_temporal_change=0.05):
        self.min_change = min_temporal_change
    
    def register_images(self, img1, img2):
        """
        Simple registration: assume same WMS tile (same bbox, same size)
        Returns (img1, img2_registered)
        """
        if img1.shape != img2.shape:
            # Resample img2 to match img1
            from scipy.ndimage import zoom
            factors = (img1.shape[0] / img2.shape[0],
                       img1.shape[1] / img2.shape[1])
            if img2.ndim == 3:
                factors = factors + (1,)
            img2 = zoom(img2, factors, order=1)
        
        return img1, img2
    
    def temporal_contrast_along_line(self, img1, img2, x1, y1, x2, y2,
                                     n_samples=50, band_width=10):
        """
        Compute temporal difference along a candidate line.
        
        Returns mean absolute difference.
        """
        H, W = img1.shape[:2]
        
        if img1.ndim == 3:
            gray1 = np.mean(img1[:, :, :3], axis=2)
            gray2 = np.mean(img2[:, :, :3], axis=2) if img2 is not None else None
        else:
            gray1 = img1
            gray2 = img2
        
        dx = x2 - x1
        dy = y2 - y1
        line_len = np.sqrt(dx**2 + dy**2)
        
        if line_len < 1:
            return 0.0, 0.0
        
        # Perpendicular direction
        ux, uy = dx / line_len, dy / line_len
        px, py = -uy, ux
        
        diffs = []
        vals1 = []
        
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            cx = x1 + t * dx
            cy = y1 + t * dy
            
            # Average in band
            band_vals1 = []
            band_vals2 = []
            
            for j in range(band_width):
                offset = j - band_width // 2
                sx = int(cx + px * offset)
                sy = int(cy + py * offset)
                
                if 0 <= sx < W and 0 <= sy < H:
                    band_vals1.append(gray1[sy, sx])
                    if gray2 is not None:
                        band_vals2.append(gray2[sy, sx])
            
            if band_vals1:
                mean1 = np.mean(band_vals1)
                vals1.append(mean1)
                
                if band_vals2:
                    mean2 = np.mean(band_vals2)
                    diffs.append(abs(mean1 - mean2))
        
        if not diffs:
            return 0.0, 0.0
        
        mean_diff = np.mean(diffs)
        # Normalize by background variation
        std1 = np.std(vals1) if len(vals1) > 1 else 1e-6
        
        # Normalized temporal contrast
        if std1 > 1e-6:
            norm_diff = mean_diff / (std1 + 1e-6)
        else:
            norm_diff = 0.0
        
        return mean_diff, norm_diff
    
    def score_with_reference(self, candidates, img_t1, img_t2=None,
                             img_sequence=None):
        """
        Score candidates based on temporal change.
        
        If img_t2 provided: compute pairwise temporal difference
        If img_sequence provided: compute temporal variance across all times
        
        High temporal change → transient feature → more likely contrail
        """
        for cand in candidates:
            # Get endpoints
            ep = _get_endpoints(cand)
            if ep is None and 'x1' in cand:
                ep = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
            if ep is None:
                cand['temporal_change'] = 0.0
                cand['temporal_score'] = 0.2
                continue
            x1, y1, x2, y2 = ep
            
            if img_t2 is not None:
                diff, norm_diff = self.temporal_contrast_along_line(
                    img_t1, img_t2, x1, y1, x2, y2)
                cand['temporal_change'] = diff
                # Sigmoid scoring
                cand['temporal_score'] = 1.0 / (1.0 + np.exp(-5 * (diff - self.min_change)))
            
            elif img_sequence is not None:
                # Multi-temporal: compute variance over all times
                all_vals = []
                for img in img_sequence:
                    dx = x2 - x1
                    dy = y2 - y1
                    line_len = np.sqrt(dx**2 + dy**2)
                    if line_len < 1:
                        continue
                    
                    ux, uy = dx / line_len, dy / line_len
                    px, py = -uy, ux
                    band_sum = 0
                    count = 0
                    mid_t = 0.5
                    cx = x1 + mid_t * dx
                    cy = y1 + mid_t * dy
                    for j in range(10):
                        offset = j - 5
                        sx = int(cx + px * offset)
                        sy = int(cy + py * offset)
                        H_img, W_img = img.shape[:2]
                        if 0 <= sx < W_img and 0 <= sy < H_img:
                            if img.ndim == 3:
                                band_sum += np.mean(img[sy, sx, :3])
                            else:
                                band_sum += img[sy, sx]
                            count += 1
                    
                    if count > 0:
                        all_vals.append(band_sum / count)
                
                if len(all_vals) > 2:
                    temporal_std = np.std(all_vals)
                    temporal_mean = np.mean(all_vals)
                    if temporal_mean > 1e-6:
                        cv = temporal_std / temporal_mean
                        cand['temporal_change'] = float(cv)
                        cand['temporal_score'] = np.clip(cv / 0.1, 0, 1)
                    else:
                        cand['temporal_change'] = 0.0
                        cand['temporal_score'] = 0.2
                else:
                    cand['temporal_change'] = 0.0
                    cand['temporal_score'] = 0.2
            else:
                cand['temporal_change'] = 0.0
                cand['temporal_score'] = 0.2
        
        return candidates


# =============================================================================
# Stage 3d: Cloud Edge vs Contrail Discriminator
# =============================================================================

class CloudDiscriminator:
    """
    Distinguish contrails from cloud edges using physical features.

    Physical basis:
    - Thick clouds are very bright (VIS albedo >0.6) — contrails less so (0.2-0.5)
    - Contrails on dark ocean: high local contrast (bright line vs dark bg)
    - Cloud edges on bright cloud: low local contrast (similar brightness on both sides)
    - Contrails have sharper cross-track profile (narrow with steep edges)
    - Aged contrails become diffuse but still narrower than cloud boundaries

    Metrics:
    1. Mean brightness along candidate line (high = cloud)
    2. Local contrast ratio vs surrounding background (high = contrail on ocean)
    3. Cross-track edge sharpness (contrast gradient steepness)

    Combined output: cloud_score in [0,1], 0 = contrail-like, 1 = cloud-like
    """

    def __init__(self, brightness_threshold=0.55, min_contrast=0.08):
        self.brightness_threshold = brightness_threshold
        self.min_contrast = min_contrast

    def sample_line_profile(self, image, x1, y1, x2, y2, band_width=10):
        """
        Sample image values along the line and its immediate surroundings.
        Returns (line_values, bg_values, brightness_values)
        """
        H, W = image.shape[:2]

        if image.ndim == 3:
            gray = np.mean(image[:, :, :3], axis=2)
        else:
            gray = image

        dx = x2 - x1
        dy = y2 - y1
        line_len = np.sqrt(dx**2 + dy**2)

        if line_len < 1:
            return [], [], []

        ux, uy = dx / line_len, dy / line_len
        px, py = -uy, ux  # perpendicular

        n_samples = max(20, int(line_len / 3))

        line_vals = []
        bg_vals = []
        brightness_vals = []

        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            cx = x1 + t * dx
            cy = y1 + t * dy

            # Line center (1px wide)
            sx, sy = int(cx), int(cy)
            if 0 <= sx < W and 0 <= sy < H:
                line_vals.append(gray[sy, sx])
                brightness_vals.append(gray[sy, sx])

            # Background: sample 3-5px on each side
            bg_samps = []
            for side in [-1, 1]:
                for offset in range(3, band_width):
                    bsx = int(cx + px * offset * side)
                    bsy = int(cy + py * offset * side)
                    if 0 <= bsx < W and 0 <= bsy < H:
                        bg_samps.append(gray[bsy, bsx])
            if bg_samps:
                bg_vals.append(np.median(bg_samps))

        return line_vals, bg_vals, brightness_vals

    def score(self, image, candidate):
        """
        Score a candidate: 0=contrail-like, 1=cloud-like.
        Returns (cloud_score, brightness_score, contrast_score)
        """
        ep = _get_endpoints(candidate)
        if ep is None:
            return 0.5, 0.5, 0.5
        x1, y1, x2, y2 = ep

        line_vals, bg_vals, brightness_vals = self.sample_line_profile(
            image, x1, y1, x2, y2)

        if len(line_vals) < 5:
            return 0.5, 0.5, 0.5

        # 1. Mean brightness score: higher = more cloud-like
        mean_brightness = np.mean(line_vals)
        brightness_score = np.clip(mean_brightness / self.brightness_threshold, 0, 1)

        # 2. Local contrast: line vs background
        if bg_vals:
            mean_line = np.mean(line_vals)
            mean_bg = np.median(bg_vals)
            abs_contrast = abs(mean_line - mean_bg)
            # Normalized: high contrast on dark bg = contrail
            contrast_score = 1.0 - np.clip(abs_contrast / 0.3, 0, 1)
            # If line is DARKER than bg, it's cloud shadow/edge — also cloud-like
            if mean_line < mean_bg:
                contrast_score = max(contrast_score, 0.6)
        else:
            contrast_score = 0.5

        # 3. Combined cloud score (weighted average)
        cloud_score = 0.4 * brightness_score + 0.6 * contrast_score

        # Invert for contrail score (0=contrail, 1=cloud)
        return cloud_score, brightness_score, contrast_score


# =============================================================================
# Stage 4: Ensemble Scoring & Ranking
# =============================================================================

class ContrailScorer:
    """
    Weighted ensemble of discriminators for final contrail probability.
    
    Weights (configurable):
    - geometry:      0.25  (FWHM, length, straightness from base detector)
    - texture:       0.15  (orientation entropy from TextureEntropyDiscriminator)
    - parallel_pair: 0.15  (bonus for forming a paired track)
    - temporal:      0.10  (transience vs persistence)
    - context:       0.10  (ocean fraction, swath gap, land avoidance)
    - cloud:         0.25  (brightness + contrast — cloud vs contrail)
    """
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'geometry': 0.25,
            'texture': 0.15,
            'parallel_pair': 0.15,
            'temporal': 0.10,
            'context': 0.10,
            'cloud': 0.25,
        }
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def score(self, candidate):
        """
        Compute ensemble score from candidate dict fields.
        
        candidate must have fields from various discriminators:
        - geometry score from base detector (or compute from FWHM etc)
        - texture_entropy if TextureEntropyDiscriminator was run
        - pair_score if ParallelTrackDiscriminator was run
        - temporal_score if MultiTemporalDiscriminator was run
        - ocean_fraction, swath_gap from base detector
        """
        scores = {}
        
        # Geometry score (from base detector features)
        fwhm = candidate.get('fwhm_km', 5.0) or 5.0
        width_score = max(0, 1.0 - abs(fwhm - 2.0) / 3.0)  # Ideal: 1-3km
        
        length = candidate.get('length_km', 50)
        length_score = min(1.0, length / 100.0)  # Longer = better (up to 100km)
        
        profile_mean = candidate.get('profile_mean', 0.01)
        is_dark = profile_mean <= 0.01
        bright_score = 0.0 if is_dark else 0.8  # Contrails are bright in VIS
        
        geometry = 0.4 * width_score + 0.3 * length_score + 0.3 * bright_score
        scores['geometry'] = geometry
        
        # Texture score
        scores['texture'] = candidate.get('texture_score', 0.3)
        
        # Parallel pair score
        scores['parallel_pair'] = candidate.get('pair_score', 0.0)
        
        # Temporal score
        scores['temporal'] = candidate.get('temporal_score', 0.2)
        
        # Context score
        ocean_frac = candidate.get('ocean_fraction', 0.5)
        is_gap = candidate.get('in_swath_gap', False)
        context = ocean_frac * (0.0 if is_gap else 1.0)
        scores['context'] = context
        
        # Cloud score (inverted: low cloud score = good = contrail-like)
        cloud_raw = candidate.get('cloud_score', 0.3)
        scores['cloud'] = 1.0 - cloud_raw  # Invert: 0.3 cloud → 0.7 contrail score
        
        # Weighted ensemble
        total_score = sum(
            self.weights.get(k, 0) * v for k, v in scores.items()
        )
        
        return total_score, scores


# =============================================================================
# Stage 5: Review Package Output
# =============================================================================

class ReviewPackageGenerator:
    """
    Generate annotated review package for human analysis.
    
    Outputs per frame:
    - annotated PNG with candidates numbered and color-coded by confidence
    - JSON summary with ranked candidate list and all features
    - Optional: individual candidate zoom-in panels
    """
    
    def __init__(self, output_dir='review_package'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, image, candidates, image_meta, paired_pairs=None,
                 sinogram=None, theta_deg=None, output_prefix='frame'):
        """
        Generate multi-panel review visualization.
        
        Panels:
        (a) Original image with numbered candidate overlays
        (b) Top-8 candidate close-ups
        (c) Confidence breakdown bar chart
        (d) Radon sinogram with detected peaks
        """
        H, W = image.shape[:2]
        n_candidates = len(candidates)
        
        # Sort by ensemble score descending
        sorted_cands = sorted(
            candidates, key=lambda c: c.get('ensemble_score', 0), reverse=True)
        
        fig = plt.figure(figsize=(20, 14))
        
        # Panel (a): Main image with candidates
        ax_a = plt.subplot(2, 3, (1, 2))
        if image.shape[-1] >= 3:
            ax_a.imshow(np.clip(image[:, :, :3], 0, 1))
        else:
            ax_a.imshow(image, cmap='gray')
        
        # Draw candidates with color coding
        cmap = plt.cm.RdYlGn
        for i, cand in enumerate(sorted_cands):
            score = cand.get('ensemble_score', 0.3)
            color = cmap(score)
            
            ep = _get_endpoints(cand)
            if ep is None and 'x1' in cand:
                ep = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
            if ep is None:
                continue
            x1, y1, x2, y2 = ep
            
            # Line width by confidence
            lw = 1.5 + 3.0 * score
            ax_a.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.7)
            
            # Number label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax_a.annotate(f"{i+1}", (mid_x, mid_y),
                         fontsize=8, fontweight='bold',
                         color='white' if score > 0.5 else 'black',
                         bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.8),
                         ha='center', va='center')
        
        meta_title = image_meta.get('date', '') + ' ' + image_meta.get('layer', '')
        ax_a.set_title(f"Candidate Lines (n={n_candidates}) | {meta_title}", fontsize=12)
        ax_a.set_xlim(0, W)
        ax_a.set_ylim(H, 0)
        ax_a.axis('off')
        
        # Panel (b): Top-N close-ups
        ax_b = plt.subplot(2, 3, 3)
        top_n = min(8, n_candidates)
        
        if top_n > 0:
            # Show confidence bars for top N
            top_scores = [c.get('ensemble_score', 0) for c in sorted_cands[:top_n]]
            colors_b = [cmap(s) for s in top_scores]
            
            bars = ax_b.barh(range(top_n), top_scores, color=colors_b, edgecolor='gray')
            ax_b.set_yticks(range(top_n))
            ax_b.set_yticklabels([f"#{i+1}" for i in range(top_n)])
            ax_b.set_xlim(0, 1)
            ax_b.set_xlabel('Confidence Score')
            ax_b.set_title(f'Top {n_candidates} Candidates', fontsize=12)
            ax_b.invert_yaxis()
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, top_scores)):
                ax_b.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{score:.2f}', va='center', fontsize=9)
        else:
            ax_b.text(0.5, 0.5, 'No candidates found', ha='center', va='center',
                     transform=ax_b.transAxes, fontsize=14)
            ax_b.axis('off')
        
        # Panel (c): Feature breakdown
        ax_c = plt.subplot(2, 3, 4)
        if n_candidates > 0:
            # Aggregate feature breakdown
            features = ['geometry', 'texture', 'parallel_pair', 'temporal', 'context']
            avg_scores = []
            for feat in features:
                vals = []
                for c in candidates:
                    if 'feature_scores' in c and feat in c['feature_scores']:
                        vals.append(c['feature_scores'][feat])
                avg_scores.append(np.mean(vals) if vals else 0)
            
            colors_c = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            ax_c.bar(features, avg_scores, color=colors_c, edgecolor='gray')
            ax_c.set_ylim(0, 1)
            ax_c.set_ylabel('Avg Score')
            ax_c.set_title('Feature Breakdown (All Candidates)', fontsize=12)
            ax_c.set_xticklabels(features, rotation=30, ha='right')
        else:
            ax_c.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax_c.transAxes)
            ax_c.axis('off')
        
        # Panel (d): Radon sinogram
        ax_d = plt.subplot(2, 3, 5)
        if sinogram is not None and theta_deg is not None:
            ax_d.imshow(sinogram, aspect='auto',
                       extent=[theta_deg[0], theta_deg[-1], 0, sinogram.shape[0]],
                       cmap='hot')
            
            # Mark detected peaks
            for cand in sorted_cands[:15]:
                if 'angle_deg' in cand and 'rho_idx' in cand:
                    ax_d.plot(cand['angle_deg'], cand['rho_idx'], 'go',
                            markersize=4, alpha=0.6)
            
            ax_d.set_xlabel('Angle (deg)')
            ax_d.set_ylabel('Rho (pixels)')
            ax_d.set_title('Radon Sinogram with Detected Peaks', fontsize=12)
        else:
            ax_d.text(0.5, 0.5, 'Sinogram N/A', ha='center', va='center',
                     transform=ax_d.transAxes)
            ax_d.axis('off')
        
        # Panel (e): Pair info
        ax_e = plt.subplot(2, 3, 6)
        if paired_pairs and len(paired_pairs) > 0:
            pair_text = "Detected Parallel Pairs:\n\n"
            for idx, (i1, i2, ang_diff, sep, score) in enumerate(paired_pairs[:8]):
                pair_text += (f"Pair {idx+1}: Cand #{i1+1} + #{i2+1}\n"
                            f"  Angle diff: {ang_diff:.1f}deg | "
                            f"Sep: {sep:.1f}km | "
                            f"Score: {score:.2f}\n")
            ax_e.text(0.05, 0.95, pair_text, transform=ax_e.transAxes,
                     fontsize=9, fontfamily='monospace', va='top')
            ax_e.set_title('Parallel Track Pairs', fontsize=12)
        else:
            ax_e.text(0.5, 0.5, 'No parallel pairs found', ha='center', va='center',
                     transform=ax_e.transAxes, fontsize=14)
        ax_e.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{output_prefix}_review.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also generate individual close-up panels
        self._generate_closeups(image, sorted_cands, output_prefix)
        
        return output_path
    
    def _generate_closeups(self, image, sorted_cands, output_prefix,
                           closeup_size=80):
        """Generate close-up panels for top candidates."""
        H, W = image.shape[:2]
        top_n = min(8, len(sorted_cands))
        
        if top_n == 0:
            return
        
        n_cols = min(4, top_n)
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = np.atleast_1d(axes)
        if axes.ndim > 1:
            axes = axes.flatten()
        
        for i in range(n_rows * n_cols):
            ax = axes[i]
            if i < top_n:
                cand = sorted_cands[i]
                
                ep = _get_endpoints(cand)
                if ep is None and 'x1' in cand:
                    ep = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
                if ep is None:
                    ax.axis('off')
                    continue
                x1, y1, x2, y2 = ep
                
                # Extract close-up region around line midpoint
                mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                x0 = max(0, mx - closeup_size)
                x1c = min(W, mx + closeup_size)
                y0 = max(0, my - closeup_size)
                y1c = min(H, my + closeup_size)
                
                if image.shape[-1] >= 3:
                    crop = image[y0:y1c, x0:x1c, :3]
                else:
                    crop = image[y0:y1c, x0:x1c]
                
                ax.imshow(np.clip(crop, 0, 1))
                
                # Draw the candidate line in the crop coord system
                lx1 = x1 - x0
                ly1 = y1 - y0
                lx2 = x2 - x0
                ly2 = y2 - y0
                ax.plot([lx1, lx2], [ly1, ly2], 'r-', linewidth=2, alpha=0.8)
                
                score = cand.get('ensemble_score', 0)
                fwhm = cand.get('fwhm_km', 0) or 0
                entropy = cand.get('texture_entropy', 3.5)
                
                title = (f"#{i+1} Score:{score:.2f} | FWHM:{fwhm:.1f}km")
                if cand.get('in_pair'):
                    title += " | PAIRED"
                ax.set_title(title, fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        closeup_path = os.path.join(self.output_dir,
                                    f"{output_prefix}_closeups.png")
        plt.savefig(closeup_path, dpi=120, facecolor='white')
        plt.close()
    
    def export_json(self, candidates, image_meta, paired_pairs, output_prefix):
        """Export full review data as JSON."""
        report = {
            'metadata': {
                'generated': datetime.utcnow().isoformat(),
                'image_info': image_meta,
                'n_candidates': len(candidates),
                'n_pairs': len(paired_pairs) if paired_pairs else 0,
            },
            'candidates': [],
            'pairs': [],
        }
        
        # Candidates (sorted by score)
        sorted_cands = sorted(
            candidates, key=lambda c: c.get('ensemble_score', 0), reverse=True)
        
        for i, cand in enumerate(sorted_cands):
            report['candidates'].append({
                'rank': i + 1,
                'ensemble_score': float(cand.get('ensemble_score', 0)),
                'feature_scores': {k: float(v) for k, v in cand.get('feature_scores', {}).items()},
                'geometry': {
                    'fwhm_km': float(cand.get('fwhm_km', 0)),
                    'line_length': float(cand.get('line_length', 0)),
                    'angle_deg': float(cand.get('angle_deg', 0)),
                    'is_dark_line': bool(cand.get('is_dark_line', False)),
                },
                'texture': {
                    'entropy': float(cand.get('texture_entropy', 3.5)),
                    'kurtosis': float(cand.get('gradient_kurtosis', 0)),
                },
                'temporal': {
                    'change': float(cand.get('temporal_change', 0)),
                },
                'context': {
                    'ocean_fraction': float(cand.get('ocean_fraction', 0)),
                    'in_swath_gap': bool(cand.get('in_swath_gap', False)),
                    'in_pair': bool(cand.get('in_pair', False)),
                },
            })
        
        if paired_pairs:
            for i1, i2, ang_diff, sep, score in paired_pairs:
                report['pairs'].append({
                    'candidate1': i1 + 1,
                    'candidate2': i2 + 1,
                    'angle_diff_deg': float(ang_diff),
                    'separation_km': float(sep),
                    'pair_score': float(score),
                })
        
        json_path = os.path.join(self.output_dir, f"{output_prefix}_review.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return json_path, report


# =============================================================================
# SkyPrint Agent — Main Pipeline Orchestrator
# =============================================================================

class SkyPrintAgent:
    """
    Main orchestrator for the SkyPrint screening pipeline.
    
    Usage:
        agent = SkyPrintAgent()
        agent.run(bbox="31S,38S,92E,104E", start_date="2014-03-08",
                  end_date="2014-03-15", output_dir="review_mh370/")
    """
    
    def __init__(self, output_dir='review_package', cache_dir=None,
                 discriminators=None):
        """
        Initialize with optional discriminator configuration.
        
        discriminators: dict of 'texture'/'parallel'/'temporal' = bool
        """
        self.output_dir = output_dir
        self.disc_config = discriminators or {
            'texture': True,
            'parallel': True,
            'temporal': True,
        }
        
        # Initialize components
        self.acquirer = GIBSImageAcquirer(cache_dir=cache_dir)
        self.screener = CoarseScreener(sensor_name='MODIS_Terra')
        self.texture_disc = TextureEntropyDiscriminator()
        self.parallel_disc = ParallelTrackDiscriminator()
        self.temporal_disc = MultiTemporalDiscriminator()
        self.cloud_disc = CloudDiscriminator()
        self.scorer = ContrailScorer()
        self.packager = ReviewPackageGenerator(output_dir=output_dir)
    
    def run_single(self, image, image_meta, ref_image=None):
        """
        Full pipeline on a single image.
        
        Returns (output_png, json_path, report)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_meta.get('date', 'unknown')}"
              f" | {image_meta.get('layer', 'unknown')}")
        print(f"{'='*60}")
        
        # Stage 2: Coarse screening (Radon)
        print("  [2/5] Radon coarse screening...")
        candidates = self.screener.screen(image)
        
        if not candidates:
            print("  No candidates found after Radon screening")
            return None, None, {'candidates': [], 'pairs': []}
        
        print(f"  Found {len(candidates)} Radon candidates")
        
        # Also get raw sinogram for visualization
        peaks_raw, sinogram, theta_deg = self.screener.screen_raw_radon(image)
        
        # Estimate image coverage
        meta_lat_range = image_meta.get('lat_range', 10)  # degrees
        meta_lon_range = image_meta.get('lon_range', 10)
        avg_range = (meta_lat_range + meta_lon_range) / 2
        image_coverage_km = avg_range * 111.0  # degrees to km
        image_coverage_km = max(image_coverage_km, 100)
        
        # Stage 3a: Texture entropy
        if self.disc_config.get('texture', True):
            print("  [3a/5] Texture entropy analysis...")
            for i, cand in enumerate(candidates):
                score, entropy, kurtosis = self.texture_disc.score(image, cand)
                cand['texture_score'] = score
                cand['texture_entropy'] = entropy
                cand['gradient_kurtosis'] = kurtosis
        
        # Stage 3b: Parallel track pairing
        paired_pairs = []
        if self.disc_config.get('parallel', True):
            print("  [3b/5] Parallel track pairing...")
            image_w = image.shape[1]
            paired_pairs = self.parallel_disc.find_pairs(
                candidates, image_w, image_coverage_km)
            
            # Mark candidates with pair scores
            _, paired_pairs = self.parallel_disc.score_candidates(
                candidates, paired_pairs, image_w, image_coverage_km)
            
            print(f"  Found {len(paired_pairs)} parallel pairs")
        
        # Stage 3c: Multi-temporal
        if self.disc_config.get('temporal', True) and ref_image is not None:
            print("  [3c/5] Multi-temporal difference...")
            candidates = self.temporal_disc.score_with_reference(
                candidates, image, img_t2=ref_image)

        # Stage 3d: Cloud detection
        if self.disc_config.get('cloud', True):
            print("  [3d/5] Cloud edge discrimination...")
            for i, cand in enumerate(candidates):
                cloud_score, bright_s, contrast_s = self.cloud_disc.score(image, cand)
                cand['cloud_score'] = cloud_score
                cand['cloud_brightness'] = bright_s
                cand['cloud_contrast'] = contrast_s

        # Stage 4: Ensemble scoring
        print("  [4/5] Ensemble scoring...")
        for cand in candidates:
            ensemble, feature_scores = self.scorer.score(cand)
            cand['ensemble_score'] = ensemble
            cand['feature_scores'] = feature_scores
        
        # Stage 5: Review package
        print("  [5/5] Generating review package...")
        png_path = self.packager.generate(
            image, candidates, image_meta,
            paired_pairs=paired_pairs,
            sinogram=sinogram, theta_deg=theta_deg,
            output_prefix=f"frame_{image_meta.get('date', 'unknown').replace('-', '')}"
        )
        
        json_path, report = self.packager.export_json(
            candidates, image_meta, paired_pairs,
            output_prefix=f"frame_{image_meta.get('date', 'unknown').replace('-', '')}"
        )
        
        # Summary stats
        scores = [c.get('ensemble_score', 0) for c in candidates]
        n_paired = sum(1 for c in candidates if c.get('in_pair'))
        
        print(f"\n  Summary:")
        print(f"    Candidates: {len(candidates)}")
        print(f"    Mean score: {np.mean(scores):.3f}")
        print(f"    Top score:  {np.max(scores):.3f}")
        print(f"    In pairs:   {n_paired}")
        print(f"    Output PNG: {png_path}")
        print(f"    Output JSON: {json_path}")
        
        return png_path, json_path, report
    
    def run_window(self, bbox, start_date, end_date, step_days=1,
                   layer='MODIS_Terra_TrueColor', width=1200):
        """
        Full pipeline over a time window.
        
        1. Acquire images
        2. For each image, run detection
        3. If temporal enabled, use consecutive pairs for differencing
        4. Generate per-frame review package
        5. Generate summary report
        
        Returns dict with summary statistics.
        """
        print(f"\n{'='*70}")
        print(f"SkyPrint Agent — Screening Pipeline")
        print(f"{'='*70}")
        print(f"Bbox: {bbox}")
        print(f"Window: {start_date} to {end_date}")
        print(f"Layer: {layer}")
        print()
        
        # Stage 1: Image acquisition
        print(f"[1/5] Image acquisition...")
        images = self.acquirer.acquire_window(
            bbox, start_date, end_date, step_days, layer, width)
        
        if not images:
            print("No images acquired. Aborting.")
            return {'error': 'no_images', 'n_frames': 0}
        
        print(f"Processing {len(images)} frames...\n")
        
        # Parse bbox for metadata
        min_lat, max_lat, min_lon, max_lon = self.acquirer.parse_bbox(bbox)
        lat_range = abs(max_lat - min_lat)
        lon_range = abs(max_lon - min_lon)
        
        all_results = []
        cumulative_candidates = []
        
        for idx, (img, path, date_str) in enumerate(images):
            meta = {
                'date': date_str,
                'layer': layer,
                'bbox': bbox,
                'lat_range': lat_range,
                'lon_range': lon_range,
                'width_px': img.shape[1],
                'height_px': img.shape[0],
                'source': os.path.basename(path),
            }
            
            # Use next frame as reference for temporal differencing
            ref_img = None
            if self.disc_config.get('temporal', True) and idx + 1 < len(images):
                ref_img = images[idx + 1][0]
            
            png_path, json_path, report = self.run_single(img, meta, ref_img)
            
            if report:
                report['frame_index'] = idx
                report['date'] = date_str
                all_results.append(report)
                cumulative_candidates.extend(report.get('candidates', []))
        
        # Generate summary report
        self._generate_summary(all_results, cumulative_candidates,
                               start_date, end_date, bbox)
        
        return {
            'n_frames': len(images),
            'total_candidates': len(cumulative_candidates),
            'results': all_results,
        }
    
    def _generate_summary(self, all_results, all_candidates,
                          start_date, end_date, bbox):
        """Generate overall summary report."""
        
        summary_path = os.path.join(self.output_dir, 'SUMMARY_REPORT.md')
        
        n_frames = len(all_results)
        n_candidates = len(all_candidates)
        
        scores = [c.get('ensemble_score', 0) for c in all_candidates]
        n_high = sum(1 for s in scores if s > 0.6)
        n_medium = sum(1 for s in scores if 0.4 < s <= 0.6)
        n_low = sum(1 for s in scores if s <= 0.4)
        
        # Top candidates
        sorted_cands = sorted(all_candidates,
                             key=lambda c: c.get('ensemble_score', 0),
                             reverse=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""# SkyPrint Screening Summary

**Generated:** {datetime.utcnow().isoformat()}
**Search Window:** {start_date} to {end_date}
**Bbox:** {bbox}
**Frames Analyzed:** {n_frames}
**Total Candidates:** {n_candidates}

## Confidence Distribution

| Tier | Score Range | Count | % |
|------|------------|-------|---|
| HIGH | > 0.60 | {n_high} | {100*n_high/max(1,n_candidates):.1f}% |
| MEDIUM | 0.40 - 0.60 | {n_medium} | {100*n_medium/max(1,n_candidates):.1f}% |
| LOW | < 0.40 | {n_low} | {100*n_low/max(1,n_candidates):.1f}% |

## Top 10 Candidates (Global)

""")
            for i, cand in enumerate(sorted_cands[:10]):
                f.write(f"**#{i+1}** | Score: {cand.get('ensemble_score',0):.3f}"
                       f" | FWHM: {cand.get('fwhm_km',0) or 0:.1f}km"
                       f" | Length: {cand.get('length_km',0):.1f}km")
                if cand.get('in_pair'):
                    f.write(" | PAIRED")
                f.write("\n")
            
            f.write(f"""
## Human Review Instructions

1. Open each `frame_*_review.png` in the output directory
2. Visually inspect numbered candidates
3. Focus on **HIGH** tier candidates first (red/orange lines)
4. Use close-up panels for detail
5. Cross-reference with known flight paths
6. Mark confirmed/ambiguous/rejected in `frame_*_review.json`

## Pipeline Configuration

- Discriminators: texture_entropy, parallel_track, multi_temporal
- Screening: Radon transform linear feature extraction
- Scoring: weighted ensemble (geometry 0.30, texture 0.25, parallel 0.20, temporal 0.15, context 0.10)
- Human review expected time: ~{n_frames*0.5:.0f} minutes ({n_frames} frames @ 30s each)
""")
        
        print(f"\nSummary report: {summary_path}")
        return summary_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SkyPrint Agent — Satellite Contrail Screening Pipeline')
    
    parser.add_argument('--bbox', type=str, required=True,
                       help='Bounding box, e.g. "31S,38S,92E,104E"')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date YYYY-MM-DD (default: same as start)')
    parser.add_argument('--window', type=int, default=1,
                       help='Days to search forward from start-date')
    parser.add_argument('--layer', type=str, default='MODIS_Terra_TrueColor',
                       help='GIBS layer name')
    parser.add_argument('--output', type=str, default='review_package',
                       help='Output directory')
    parser.add_argument('--width', type=int, default=1200,
                       help='Image width in pixels')
    parser.add_argument('--no-texture', action='store_true',
                       help='Disable texture entropy discriminator')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel track discriminator')
    parser.add_argument('--no-temporal', action='store_true',
                       help='Disable multi-temporal discriminator')
    parser.add_argument('--cache', type=str, default=None,
                       help='GIBS cache directory')
    
    args = parser.parse_args()
    
    # Compute end date
    if args.end_date is None and args.window > 1:
        end_dt = datetime.strptime(args.start_date, '%Y-%m-%d') + \
                 timedelta(days=args.window - 1)
        args.end_date = end_dt.strftime('%Y-%m-%d')
    elif args.end_date is None:
        args.end_date = args.start_date
    
    discriminators = {
        'texture': not args.no_texture,
        'parallel': not args.no_parallel,
        'temporal': not args.no_temporal,
    }
    
    # Create and run agent
    agent = SkyPrintAgent(
        output_dir=args.output,
        cache_dir=args.cache,
        discriminators=discriminators,
    )
    
    result = agent.run_window(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        layer=args.layer,
        width=args.width,
    )
    
    print(f"\n{'='*70}")
    print(f"SkyPrint pipeline complete.")
    print(f"Frames: {result.get('n_frames', 0)}")
    print(f"Total candidates: {result.get('total_candidates', 0)}")
    print(f"Output: {args.output}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
