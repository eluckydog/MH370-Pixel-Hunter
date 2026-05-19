"""
Contrail (Condensation Trail) Detection Module for Satellite Imagery
=====================================================================
Detects aircraft contrails in MODIS/Terra/Aqua, VIIRS, and MTSAT-2 imagery.

Physical basis:
1. Contrails form at T < -40C with ice-supersaturated atmosphere
2. In VIS: bright linear features against darker background (cloud top or ocean)
3. In IR: cold linear features (ice emissivity ~0.8 vs cloud ~0.9-1.0)
4. At night: only IR/DNB viable; DNB needs moonlight for contrast

Detection pipeline:
  - Stage 1: Image acquisition (NASA GIBS WMS/TMS)
  - Stage 2: Preprocessing (atmospheric correction proxy, enhancement)
  - Stage 3: Linear feature extraction (morphological + Hough/Radon)
  - Stage 4: Contrail-specific classification (shape + thermal + context)
  - Stage 5: Flight path association (great-circle arc fitting)

Author: math-science agent
Date: 2026-05-20
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing
from scipy.ndimage import label as ndimage_label
from skimage.transform import radon, iradon
from skimage.feature import canny
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import regionprops, find_contours
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Physical Constants & Models
# =============================================================================

class AtmosphericModel:
    """Standard atmosphere + contrail formation conditions."""
    
    # ISA standard atmosphere layers (altitude km -> temperature K)
    ISA_TABLE = {
        0: 288.15,   # Sea level
        2: 275.15,
        4: 262.15,
        6: 249.15,
        8: 236.15,
        10: 223.15,
        11: 216.65,   # Tropopause start
        12: 216.65,
        14: 216.65,
        16: 216.65,
        20: 216.65,   # Lower stratosphere
    }
    
    @staticmethod
    def temperature_at_altitude(alt_km):
        """Interpolate temperature at given altitude (km)."""
        altitudes = sorted(AtmosphericModel.ISA_TABLE.keys())
        temps = [AtmosphericModel.ISA_TABLE[a] for a in altitudes]
        
        if alt_km <= altitudes[0]:
            return temps[0]
        if alt_km >= altitudes[-1]:
            return temps[-1]
        
        for i in range(len(altitudes) - 1):
            if altitudes[i] <= alt_km <= altitudes[i+1]:
                t_frac = (alt_km - altitudes[i]) / (altitudes[i+1] - altitudes[i])
                return temps[i] + t_frac * (temps[i+1] - temps[i])
        return temps[-1]
    
    @staticmethod
    def pressure_at_altitude(alt_km):
        """Approximate pressure at altitude using barometric formula."""
        T0 = 288.15  # K
        P0 = 101325  # Pa
        g = 9.81     # m/s^2
        M = 0.029    # kg/mol (air)
        R = 8.314    # J/(mol*K)
        L = 0.0065   # K/m lapse rate
        
        h = alt_km * 1000
        if h < 11000:
            T = T0 - L * h
            return P0 * (T / T0) ** (g * M / (R * L))
        else:
            T11 = T0 - L * 11000
            P11 = P0 * (T11 / T0) ** (g * M / (R * L))
            h_strato = h - 11000
            return P11 * np.exp(-g * M * h_strato / (R * T11))
    
    @staticmethod
    def contrail_formation_probability(alt_km, lat_deg, month=3):
        """
        Estimate probability of contrail formation.
        Based on Schmidt-Appleman criterion simplified.
        
        Returns float 0-1 representing likelihood.
        """
        T = AtmosphericModel.temperature_at_altitude(alt_km)
        
        # Must be below -40C for persistent contrail
        if T > 233.15:
            base_prob = 0.0
        elif T < 220.0:
            base_prob = 0.95  # Very cold, almost certain
        else:
            base_prob = (233.15 - T) / (233.15 - 220.0)
        
        # Latitude factor: higher latitude = more likely (colder tropopause)
        abs_lat = abs(lat_deg)
        lat_factor = 0.7 + 0.3 * min(abs_lat / 45.0, 1.0)
        
        # Seasonal factor: winter more likely than summer
        if month in [12, 1, 2]:
            season_factor = 1.1  # Winter NH / Summer SH
        elif month in [6, 7, 8]:
            season_factor = 0.85
        else:
            season_factor = 1.0
        
        prob = base_prob * lat_factor * season_factor
        return np.clip(prob, 0.0, 1.0)


class ContrailPhysics:
    """Physical model of contrail appearance in satellite imagery."""
    
    # Typical B777 engine parameters
    B777_PARAMS = {
        'engines': 2,
        'fuel_flow_kg_s': 2.5,       # Cruise fuel flow per engine
        'exhaust_temp_K': 550,         # Engine exit temperature
        'water_emission_g_kg': 1.25,   # g H2O per kg fuel burned
        'soot_particles_per_kg': 1e15,  # Soot CCN concentration
        'cruise_speed_m_s': 250,       # Typical cruise speed
        'wingspan_m': 64.8,            # B777-200ER wingspan
    }
    
    @staticmethod
    def initial_contrail_width(fuel_flow_kg_s=2.5, ambient_T_K=217, 
                               wind_speed_m_s=30, age_seconds=300):
        """
        Estimate contrail width after diffusion.
        
        Initial width from engines: ~ wingspan (engine spacing)
        Diffusion grows as sqrt(t) due to turbulent mixing.
        """
        w0 = 50.0  # Initial half-width ~ engine spacing / 2 (meters)
        K_turb = 200.0  # Turbulent diffusivity (m^2/s) for young contrails
        
        # Diffusive spreading
        w_diff = w0 + 2.0 * np.sqrt(K_turb * age_seconds)
        
        # Wind shear spreading
        shear_rate = 0.01  # Approximate vertical wind shear (1/s)
        w_shear = shear_rate * wind_speed_m_s * age_seconds * 0.1
        
        return max(w_diff, w_shear)
    
    @staticmethod
    def contrail_optical_depth(age_seconds=600, initial_od=3.0):
        """
        Optical depth evolution.
        Young contrails: OD >> 1 (opaque white)
        Aging: OD decreases as ice crystals grow and sediment out.
        """
        # Exponential decay with time constant ~ 30 minutes for sublimation
        tau_decay = 1800.0  # seconds
        od = initial_od * np.exp(-age_seconds / tau_decay)
        return od
    
    @staticmethod
    def radiance_contrast_vis(od, sun_zenith_deg, sensor_zenith_deg,
                              rel_azimuth_deg, surface_albedo=0.05):
        """
        Visible band radiance contrast between contrail and background.
        
        Returns delta_L/L_background (fractional contrast).
        """
        # Ice crystal single-scattering albedo
        omega_ice = 0.95
        
        # Asymmetry parameter for ice crystals
        g_ice = 0.85
        
        # Simplified contrast model
        mu_sun = np.cos(np.radians(sun_zenith_deg))
        mu_sensor = np.cos(np.radians(sensor_zenith_deg))
        
        if mu_sun <= 0 or mu_sensor <= 0:
            return 0.0  # Night time, no visible signal
        
        # Scattering phase function (Henyey-Greenstein approx)
        cos_phase = (-mu_sun * mu_sensor + 
                     np.sqrt((1-mu_sun**2)*(1-mu_sensor**2)) * 
                     np.cos(np.radians(rel_azimuth_deg)))
        phase = (1 - g_ice**2) / (1 + g_ice**2 - 2*g_ice*cos_phase)**1.5
        
        # Single scattering approximation
        L_contrail = omega_ice * od * phase * mu_sun / (4 * np.pi * (mu_sun + mu_sensor))
        L_background = surface_albedo * mu_sun / np.pi
        
        if L_background < 1e-6:
            return 0.0
        
        contrast = (L_contrail - L_background) / L_background
        return contrast
    
    @staticmethod
    def brightness_temperature_contrir(T_contrail_K, T_background_K):
        """IR brightness temperature difference."""
        return T_contrail_K - T_background_K


# =============================================================================
# Satellite Sensor Models
# =============================================================================

class SensorModel:
    """Sensor characteristics for common satellite instruments."""
    
    SENSORS = {
        'MODIS_Terra': {
            'name': 'MODIS on Terra',
            'swath_km': 2330,
            'resolution_vis': 250,      # meters (bands 1-2)
            'resolution_ir': 1000,       # meters (TIR bands)
            'equator_crossing': 1030,    # LT descending
            'revisit_days': 1,
            'bands': {
                'vis_red': {'center_um': 0.65, 'res_m': 250},
                'vis_nir': {'center_um': 0.86, 'res_m': 250},
                'swir': {'center_um': 1.64, 'res_m': 500},
                'tir_11': {'center_um': 11.03, 'res_m': 1000},
                'tir_12': {'center_um': 12.02, 'res_m': 1000},
                'tir_85': {'center_um': 8.55, 'res_m': 1000},
            },
            'bits': 12,
        },
        'VIIRS': {
            'name': 'VIIRS on Suomi NPP',
            'swath_km': 3040,
            'resolution_vis': 375,       # I-band
            'resolution_ir': 750,        # I-band IR
            'dnb_resolution': 750,       # Day-Night Band
            'equator_crossing': 1330,    # LT ascending
            'revisit_days': 1,
            'bands': {
                'i1': {'center_um': 0.64, 'res_m': 375},
                'i2': {'center_um': 0.86, 'res_m': 375},
                'i4': {'center_um': 3.74, 'res_m': 375},
                'i5': {'center_um': 11.45, 'res_m': 375},
                'dnb': {'center_um': 0.70, 'res_m': 750, 'type': 'dnb'},
            },
            'bits': 14,
        },
        'MTSAT2': {
            'name': 'MTSAT-2 (Himawari-7)',
            'position_lon': 145.0,
            'resolution_vis': 1400,      # At nadir; degrades off-nadir
            'resolution_ir': 4000,       # At nadir
            'cadence_min': 30,
            'bands': {
                'vis': {'center_um': 0.65, 'res_m': 1400},
                'ir1': {'center_um': 10.8, 'res_m': 4000},
                'ir2': {'center_um': 12.0, 'res_m': 4000},
                'ir3': {'center_um': 6.8, 'res_m': 4000},
                'ir4': {'center_um': 3.9, 'res_m': 4000},
            },
            'bits': 10,
        },
    }
    
    @staticmethod
    def gsd_at_location(sensor_name, target_lat, target_lon):
        """
        Calculate Ground Sample Distance at target location.
        Accounts for off-nadir viewing angle for GEO satellites.
        """
        sensor = SensorModel.SENSORS[sensor_name]
        
        if sensor_name == 'MTSAT2':
            # GEO satellite: GSD degrades with zenith angle
            sat_lon = sensor['position_lon']
            
            # Calculate satellite zenith angle at target
            from math import radians, sin, cos, acos, degrees
            
            phi_r = radians(target_lat)
            lambda_r = radians(target_lon)
            lambda_s = radians(sat_lon)
            
            # Geocentric angle between satellite and target
            cos_gamma = (sin(phi_r) * sin(0) + 
                        cos(phi_r) * cos(0) * cos(lambda_r - lambda_s))
            gamma = acos(np.clip(cos_gamma, -1, 1))
            
            # Zenith angle at target
            Re = 6378.137  # Earth radius (km)
            Rs = 42164.0   # GEO radius (km)
            
            cos_eta = sin(gamma) / (Rs / Re * sin(np.pi - gamma))
            eta = np.arccos(np.clip(abs(cos_eta), 0, 1))
            
            # GSD scales as 1/cos(eta)
            nadir_res = sensor['resolution_vis']
            gsd = nadir_res / max(np.cos(eta), 0.1)
            return gsd
        
        else:
            # LEO satellite: minimal variation across swath
            return sensor['resolution_vis']


# =============================================================================
# Orbit & Overpass Calculator
# =============================================================================

class OverpassCalculator:
    """Calculate satellite overpass times for a given location."""
    
    @staticmethod
    def modis_overpass_times(target_lat, target_lon, date_str='2014-03-08'):
        """
        Estimate MODIS Terra/Aqua overpass windows for a target.
        
        MODIS Terra: ~10:30 LT descending (night side ascending)
        MODIS Aqua: ~13:30 LT ascending (night side descending)
        
        Returns dict with overpass UTC times and geometry.
        """
        # Simplified: use equatorial crossing time + longitude offset
        # Real calculation would need TLE propagation
        
        # Time zone offset from UTC to local solar time
        lon_offset_hours = target_lon / 15.0
        
        # Terra descending node (daytime) at target
        terra_day_lt = 10.30 + (target_lon % 15) / 15.0 * 0.5  # ~30min window variation
        terra_day_utc = terra_day_lt - lon_offset_hours
        terra_day_utc = terra_day_utc % 24
        
        # Terra ascending node (nighttime)
        terra_night_lt = 22.30 + (target_lon % 15) / 15.0 * 0.5
        terra_night_utc = terra_night_lt - lon_offset_hours
        terra_night_utc = terra_night_utc % 24
        
        # Aqua
        aqua_day_lt = 13.30 + (target_lon % 15) / 15.0 * 0.5
        aqua_day_utc = aqua_day_lt - lon_offset_hours
        aqua_day_utc = aqua_day_utc % 24
        
        aqua_night_lt = 01.30 + (target_lon % 15) / 15.0 * 0.5
        aqua_night_utc = aqua_night_lt - lon_offset_hours
        aqua_night_utc = aqua_night_utc % 24
        
        return {
            'terra_day_utc': terra_day_utc,
            'terra_night_utc': terra_night_utc,
            'aqua_day_utc': aqua_day_utc,
            'aqua_night_utc': aqua_night_utc,
            'target_lat': target_lat,
            'target_lon': target_lon,
            'date': date_str,
        }
    
    @staticmethod
    def viirs_overpass_times(target_lat, target_lon, date_str='2014-03-08'):
        """VIIRS on Suomi NPP: ~13:30 LT ascending."""
        lon_offset_hours = target_lon / 15.0
        
        viirs_day_lt = 13.30 + (target_lon % 15) / 15.0 * 0.3
        viirs_day_utc = (viirs_day_lt - lon_offset_hours) % 24
        viirs_night_utc = (viirs_day_utc + 12) % 24
        
        return {
            'day_utc': viirs_day_utc,
            'night_utc': viirs_night_utc,
        }


# =============================================================================
# Image Acquisition
# =============================================================================

def fetch_worldview_image(bbox, date_str, layer='MODIS_Terra_CorrectedReflectance_TrueColor',
                          width=1200, height=800, output_dir=None):
    """
    Fetch satellite image from NASA Worldview/GIBS WMS.
    
    Parameters:
        bbox: [west, south, east, north] in degrees
        date_str: YYYY-MM-DD format
        layer: GIBS layer name
        output_dir: directory to save image
    
    Returns: path to saved image file
    """
    import urllib.request
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    bbox_str = ','.join(map(str, bbox))
    url = (f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?"
           f"REQUEST=GetSnapshot&TIME={date_str}"
           f"&BBOX={bbox_str}&CRS=EPSG:4326"
           f"&LAYERS={layer}&FORMAT=image/png"
           f"&WIDTH={width}&HEIGHT={height}")
    
    safe_name = f"worldview_{date_str}_{layer[:20]}_{bbox[1]:.0f}_{bbox[0]:.0f}.png"
    filepath = os.path.join(output_dir, safe_name)
    
    try:
        print(f"[FETCH] {url[:80]}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        
        with open(filepath, 'wb') as f:
            f.write(data)
        
        size_kb = len(data) / 1024
        print(f"[OK] Saved {filepath} ({size_kb:.0f} KB)")
        return filepath
    except Exception as e:
        print(f"[FAIL] {e}")
        return None


def fetch_modis_ir_image(bbox, date_str, output_dir=None):
    """Fetch MODIS thermal infrared (band 31/32) false color."""
    return fetch_worldview_image(
        bbox, date_str,
        layer='MODIS_Terra_Bands721',
        width=1200, height=800,
        output_dir=output_dir
    )


# =============================================================================
# Detection Pipeline
# =============================================================================

class ContrailDetector:
    """
    Main detection class. Pipeline:
    1. Load image
    2. Enhance linear features
    3. Extract candidates via Radon/Hough
    4. Classify by contrail physics
    5. Output detections with confidence scores
    """
    
    def __init__(self, sensor_name='MODIS_Terra', target_lat=None, target_lon=None):
        self.sensor = sensor_name
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.gsd = None
        if target_lat is not None and target_lon is not None:
            self.gsd = SensorModel.gsd_at_location(sensor_name, target_lat, target_lon)
        
        self.results = []
        self.debug_images = {}
    
    def load_image(self, filepath):
        """Load image from file."""
        from PIL import Image
        
        img = plt.imread(filepath)
        self.original = img.copy()
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Already normalized 0-1
            self.image = img
        else:
            # uint8 or uint16
            self.image = img.astype(np.float64) / np.iinfo(img.dtype).max
        
        # Handle RGBA -> RGB
        if self.image.shape[-1] == 4:
            self.image = self.image[:, :, :3]
        
        self.filepath = filepath
        self.height, self.width = self.image.shape[:2]
        print(f"[LOAD] {self.shape_desc()}")
        return self
    
    def shape_desc(self):
        return f"{self.width}x{self.height}px, dtype={self.image.dtype}, range=[{self.image.min():.3f}, {self.image.max():.3f}]"
    
    def preprocess(self):
        """
        Preprocessing for contrail detection.
        - Convert to grayscale luminance
        - Adaptive histogram equalization (CLAHE-like)
        - High-pass filter to enhance linear features
        """
        # RGB to luminance
        if self.image.shape[-1] == 3:
            gray = 0.2126 * self.image[:,:,0] + 0.7152 * self.image[:,:,1] + 0.0722 * self.image[:,:,2]
        else:
            gray = self.image.copy()
        
        self.gray = gray
        
        # Adaptive local contrast enhancement
        # Use difference-of-Gaussians as edge-enhancement proxy
        sigma_low = 8.0
        sigma_high = 1.0
        
        blur_low = gaussian_filter(gray, sigma=sigma_low)
        blur_high = gaussian_filter(gray, sigma=sigma_high)
        
        # High-pass: enhances fine linear structures
        self.enhanced = blur_high - blur_low
        self.enhanced = (self.enhanced - self.enhanced.mean()) / (self.enhanced.std() + 1e-8)
        
        # Also compute a version that preserves sign (bright/dark lines)
        self.signed_enhanced = blur_high - blur_low
        
        print(f"[PREPROCESS] Enhanced range: [{self.enhanced.min():.2f}, {self.enhanced.max():.2f}]")
        return self
    
    def detect_linear_features_radon(self, theta_range=(-5, 95), n_theta=180,
                                      threshold_sigma=2.0):
        """
        Use Radon transform to detect linear features.
        
        Contrails appear as straight or gently curved lines in satellite imagery.
        The Radon transform accumulates intensity along all possible lines.
        """
        print("[RADON] Computing transform...")
        
        # Pad image to avoid edge artifacts
        pad_size = min(self.width, self.height) // 8
        padded = np.pad(self.enhanced, pad_size, mode='reflect')
        
        theta = np.linspace(theta_range[0], theta_range[1], n_theta)
        sinogram = radon(padded, theta=theta, circle=True)
        
        # Find peaks in sinogram space
        mean_val = sinogram.mean()
        std_val = sinogram.std()
        threshold = mean_val + threshold_sigma * std_val
        
        peaks_mask = sinogram > threshold
        
        # Label connected regions in peak mask
        labeled, num_features = ndimage_label(peaks_mask)
        
        # Extract peak properties
        regions = regionprops(labeled, intensity_image=sinogram)
        
        # Sort by intensity (strongest first)
        regions.sort(key=lambda r: r.max_intensity, reverse=True)
        
        # Convert back to image coordinates
        detections = []
        for i, region in enumerate(regions[:50]):  # Top 50 candidates
            centroid = region.centroid
            rho_idx, theta_idx = centroid
            
            rho = (rho_idx - pad_size)  # Adjust for padding
            angle = theta[int(theta_idx) % len(theta)]
            
            # Compute line endpoints in original image
            angle_rad = np.radians(angle)
            
            # Line equation: x*cos(a) + y*sin(a) = rho
            endpoints = self._line_endpoints(rho, angle_rad, margin=0.05)
            
            length = np.sqrt((endpoints[2]-endpoints[0])**2 + 
                           (endpoints[3]-endpoints[1])**2)
            
            # Length must be physically plausible for a contrail
            # Minimum: ~5km at this resolution
            # Maximum: ~500km (very long persistent contrail)
            min_length_px = max(10, 5000 / (self.gsd or 250))
            max_length_px = min(max(self.width, self.height) * 0.8, 
                               500000 / (self.gsd or 250))
            
            if length < min_length_px or length > max_length_px:
                continue
            
            score = (region.max_intensity - mean_val) / (std_val + 1e-8)
            
            detections.append({
                'id': i,
                'rho': rho,
                'angle_deg': angle,
                'angle_rad': angle_rad,
                'endpoints': endpoints,
                'length_px': length,
                'length_km': length * (self.gsd or 250) / 1000,
                'score': score,
                'intensity': region.max_intensity,
                'centroid_pixel': (rho_idx, theta_idx),
            })
        
        self.radon_detections = detections
        self.sinogram = sinogram
        self.theta = theta
        
        print(f"[RADON] Found {len(detections)} linear feature candidates")
        return detections
    
    def _line_endpoints(self, rho, angle_rad, margin=0.05):
        """Compute line endpoints within image bounds."""
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        H, W = self.height, self.width
        
        points = []
        for x in [margin*W, (1-margin)*W]:
            if abs(sin_a) > 1e-6:
                y = (rho - x * cos_a) / sin_a
                if 0 <= y <= H:
                    points.append((x, y))
        
        for y in [margin*H, (1-margin)*H]:
            if abs(cos_a) > 1e-6:
                x = (rho - y * sin_a) / cos_a
                if 0 <= x <= W:
                    points.append((x, y))
        
        if len(points) >= 2:
            # Pick two farthest apart
            dists = [(p[0]-q[0])**2 + (p[1]-q[1])**2 
                    for i, p in enumerate(points) for q in points[i+1:]]
            idx = dists.index(max(dists))
            # Reconstruct which pair
            pairs = [(points[i], points[j]) 
                    for i in range(len(points)) for j in range(i+1, len(points))]
            p1, p2 = pairs[idx]
            return [p1[0], p1[1], p2[0], p2[1]]
        
        return [0, 0, W, H]
    
    def detect_edges_canny(self, sigma=2.0, low_thresh=0.1, high_thresh=0.3):
        """Edge detection as complementary method to Radon."""
        print("[CANNY] Detecting edges...")
        
        edges = canny(self.enhanced, sigma=sigma, 
                      low_threshold=low_thresh, high_threshold=high_thresh)
        
        # Skeletonize to get thin edges
        skel = skeletonize(edges)
        
        # Remove small isolated clusters
        cleaned = remove_small_objects(skel, min_size=50)
        
        self.edges = edges
        self.edge_skeleton = cleaned
        print(f"[CANNY] Edge pixels: {edges.sum()}, skeleton pixels: {cleaned.sum()}")
        return cleaned
    
    def _create_ocean_mask(self, blue_threshold=None):
        """
        Create ocean/water mask from RGB image.
        
        Strategy: Ocean pixels have high blue reflectance and low NDVI.
        Land has lower blue/red ratio, clouds are bright in all channels.
        Swath gaps have near-zero values in all channels.
        """
        if self.image.shape[-1] < 3:
            return np.ones((self.height, self.width), dtype=bool)
        
        R = self.image[:,:,0]
        G = self.image[:,:,1]
        B = self.image[:,:,2]
        
        # Ocean: blue-dominated, moderate-low overall brightness
        # (B-R)/(B+R) for water detection
        ndwi = (G - B) / (np.maximum(G + B, 1e-8))
        
        # Brightness: ocean is relatively dark
        brightness = (R + G + B) / 3.0
        
        # Swath gap: near-zero in all channels
        is_gap = brightness < 0.02
        
        # Ocean: low brightness, blue-biased
        is_ocean = (brightness < 0.4) & (B > R * 0.9) & (~is_gap)
        
        # Morphological cleanup
        is_ocean = binary_closing(is_ocean, structure=np.ones((5,5)), iterations=2)
        is_ocean = binary_erosion(is_ocean, structure=np.ones((3,3)), iterations=1)
        
        # Remove small non-ocean holes
        is_ocean = remove_small_objects(is_ocean, min_size=200)
        
        self.ocean_mask = is_ocean
        self.swath_gap_mask = is_gap
        
        ocean_pct = is_ocean.sum() / is_ocean.size * 100
        gap_pct = is_gap.sum() / is_gap.size * 100
        print(f"[OCEAN] {ocean_pct:.1f}% ocean, {gap_pct:.1f}% swath gap")
        return is_ocean
    
    def _line_ocean_fraction(self, x1, y1, x2, y2, n_samples=30):
        """Fraction of a line segment that falls over ocean pixels."""
        if not hasattr(self, 'ocean_mask'):
            return 1.0
        xs = np.linspace(x1, x2, n_samples).astype(int)
        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.clip(xs, 0, self.width - 1)
        ys = np.clip(ys, 0, self.height - 1)
        return self.ocean_mask[ys, xs].mean()
    
    def _line_gap_fraction(self, x1, y1, x2, y2, n_samples=30):
        """Fraction of a line that falls in swath gap."""
        if not hasattr(self, 'swath_gap_mask'):
            return 0.0
        xs = np.linspace(x1, x2, n_samples).astype(int)
        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.clip(xs, 0, self.width - 1)
        ys = np.clip(ys, 0, self.height - 1)
        return self.swath_gap_mask[ys, xs].mean()
    
    def _cross_section_fwhm(self, x1, y1, x2, y2, n_cross=5, width=15):
        """
        Measure FWHM of perpendicular cross-sections along the line.
        Contrails are thin (<2-3km), cloud boundaries are broad (>10km).
        
        Returns: (median_fwhm_px, fwhm_consistency)
        """
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 10:
            return None, None
        
        # Unit direction vector
        ux, uy = dx/length, dy/length
        # Perpendicular direction
        px, py = -uy, ux
        
        fwhms = []
        for t in np.linspace(0.1, 0.9, n_cross):
            cx = int(x1 + t * dx)
            cy = int(y1 + t * dy)
            
            # Sample perpendicular profile
            profile = []
            for w in range(-width, width+1):
                sx = int(cx + w * px)
                sy = int(cy + w * py)
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    profile.append(self.gray[sy, sx])
            
            if len(profile) < 5:
                continue
            
            profile = np.array(profile)
            profile = profile - np.min(profile)
            if profile.max() < 1e-6:
                continue
            profile = profile / profile.max()
            
            # Find half-max crossings
            half = 0.5
            above = profile > half
            transitions = np.diff(above.astype(int))
            
            if np.sum(transitions == 1) > 0 and np.sum(transitions == -1) > 0:
                left = np.where(transitions == 1)[0][0]
                right = np.where(transitions == -1)[0][-1]
                fwhms.append(right - left)
        
        if not fwhms:
            return None, 0.0
        
        median_fwhm = np.median(fwhms)
        consistency = 1.0 / (1.0 + np.std(fwhms))
        return median_fwhm, consistency
    
    def _local_background_uniformity(self, x1, y1, x2, y2, band_width=20):
        """
        Measure background uniformity near the line.
        Contrail: over uniform ocean → high uniformity
        Cloud edge: over variable cloud field → low uniformity
        Coastline: mixed land/ocean → very low uniformity
        """
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 10:
            return 1.0
        
        # Perpendicular unit vector
        ux, uy = dx/length, dy/length
        px, py = -uy, ux
        
        backgrounds = []
        for t in np.linspace(0.15, 0.85, 8):
            cx = int(x1 + t * dx)
            cy = int(y1 + t * dy)
            
            # Sample both sides of the line
            for side_sign in [-1, 1]:
                for w in range(3, band_width):
                    sx = int(cx + side_sign * w * px)
                    sy = int(cy + side_sign * w * py)
                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        backgrounds.append(self.gray[sy, sx])
        
        if len(backgrounds) < 20:
            return 0.5
        
        backgrounds = np.array(backgrounds)
        # Uniformity = 1 - normalized std (higher std = less uniform = more likely cloud/coast)
        uniformity = 1.0 / (1.0 + backgrounds.std() / max(backgrounds.mean(), 0.01))
        return float(uniformity)
    
    def classify_contrails(self, detections, min_score=3.0):
        """
        Classify linear features as contrail candidates based on physical criteria.
        
        Contrail signatures (v3 - hard gates + reweighted scoring):
        HARD GATES (fail = immediate reject):
          Gate 1: Not in swath gap (>30% in gap)
          Gate 2: Over ocean (>50% of line)
          Gate 3: Length >= 20km (noise floor)
          Gate 4: FWHM <= 5km (cloud edges are wider)
        WEIGHTED SCORING (total max ~0.65):
          1. Bright in VIS - weight 0.10
          2. Over ocean (soft) - weight 0.15
          3. Narrow FWHM (1-3km ideal) - weight 0.20
          4. Uniform background - weight 0.10
          5. Plausible length - weight 0.10
          6. Brightness consistency - weight 0.10
          7. Brightness sign - weight 0.05
        """
        # Create ocean mask for spatial filtering
        self._create_ocean_mask()
        
        classified = []
        reject_reasons = {
            'swath_gap': 0, 'over_land': 0, 'too_short': 0,
            'too_wide_hard': 0, 'too_wide_soft': 0,
            'non_uniform_bg': 0, 'dark_line': 0, 'low_score': 0
        }
        
        for det in detections:
            score = det['score']
            if score < min_score:
                reject_reasons['low_score'] += 1
                continue
            
            length_km = det.get('length_km', 0)
            x1, y1, x2, y2 = det['endpoints']
            
            # --- Pass/Fail gates (hard filters) ---
            
            # Gate 1: Must not be in swath gap (>30% of line in gap → reject)
            gap_frac = self._line_gap_fraction(x1, y1, x2, y2)
            if gap_frac > 0.3:
                reject_reasons['swath_gap'] += 1
                continue
            
            # Gate 2: Must be primarily over ocean (>50% of line over ocean)
            ocean_frac = self._line_ocean_fraction(x1, y1, x2, y2)
            if ocean_frac < 0.5:
                reject_reasons['over_land'] += 1
                continue
            
            # Gate 3 (v3): Must be long enough (contrails are >20km, noise is shorter)
            if length_km < 20:
                reject_reasons['too_short'] += 1
                continue
            
            # --- Weighted scoring ---
            
            # Sample line profile
            profile = self._sample_line_profile(x1, y1, x2, y2, n_samples=50)
            if len(profile) < 10:
                continue
            
            profile_mean = np.mean(profile)
            profile_std = np.std(profile)
            
            # Component 1: Score (0.10)
            conf_score = min(score / 10.0, 1.0) * 0.10
            
            # Component 2: Ocean coverage (0.15) - reduced from 0.25
            conf_ocean = ocean_frac * 0.15
            
            # Component 3: Cross-section width (0.20) - key discriminator
            fwhm, fwhm_cons = self._cross_section_fwhm(x1, y1, x2, y2, n_cross=5, width=20)
            if fwhm is not None:
                gsd = self.gsd or 250
                fwhm_km = fwhm * gsd / 1000
                # Contrail: 0.5-3km FWHM → high score
                # Cloud edge: 5-20km FWHM → low score
                if 1.0 < fwhm_km < 4.0:
                    conf_width = 0.20 * fwhm_cons
                elif 0.5 < fwhm_km < 6.0:
                    conf_width = 0.10 * fwhm_cons
                else:
                    conf_width = 0.0
                    if fwhm_km > 4.0:
                        reject_reasons['too_wide_soft'] += 1
            else:
                conf_width = 0.05
                fwhm_km = None
            
            # Gate 4 (v3): Hard reject FWHM > 5km (cloud edge, not contrail)
            if fwhm_km is not None and fwhm_km > 5.0:
                reject_reasons['too_wide_hard'] += 1
                continue
            
            # Component 4: Background uniformity (0.10)
            bg_uniformity = self._local_background_uniformity(x1, y1, x2, y2)
            conf_bg = bg_uniformity * 0.10
            if bg_uniformity < 0.3:
                reject_reasons['non_uniform_bg'] += 1
            
            # Component 5: Length plausibility (0.10)
            if 20 < length_km < 500:
                conf_len = 0.10
            elif 10 < length_km < 800:
                conf_len = 0.07
            elif 5 < length_km < 1200:
                conf_len = 0.04
            else:
                conf_len = 0.0
            
            # Component 6: Brightness consistency (0.10)
            consistency_val = abs(profile_mean) / (profile_std + 1e-8)
            conf_consistency = min(consistency_val / 5.0, 1.0) * 0.10
            
            # Component 7: Brightness sign (0.05) - reduced from 0.10
            if profile_mean > 0.01:
                conf_sign = 0.05
            elif profile_mean > -0.01:
                conf_sign = 0.02  # Neutral
            else:
                conf_sign = 0.0
                reject_reasons['dark_line'] += 1
            
            confidence = (conf_score + conf_ocean + conf_width + conf_bg + 
                         conf_len + conf_consistency + conf_sign)
            
            # Aspect ratio
            aspect_ratio = length_km / max((fwhm_km or 1.0), 0.1)
            
            det['confidence'] = confidence
            det['profile'] = profile.tolist()
            det['profile_mean'] = float(profile_mean)
            det['profile_std'] = float(profile_std)
            det['aspect_ratio'] = aspect_ratio
            det['fwhm_km'] = float(fwhm_km) if fwhm_km is not None else None
            det['ocean_fraction'] = float(ocean_frac)
            det['bg_uniformity'] = float(bg_uniformity)
            det['in_swath_gap'] = bool(gap_frac > 0.3)
            det['gap_frac'] = float(gap_frac)
            det['is_candidate'] = confidence > 0.35
            
            classified.append(det)
        
        classified.sort(key=lambda d: d['confidence'], reverse=True)
        self.classified = classified
        
        n_candidates = sum(1 for c in classified if c['is_candidate'])
        print(f"[CLASSIFY] {n_candidates}/{len(classified)} candidates (above 0.35)")
        print(f"[REJECT] Reasons: {reject_reasons}")
        return classified
    
    def _sample_line_profile(self, x1, y1, x2, y2, n_samples=50):
        """Sample pixel values along a line segment."""
        xs = np.linspace(x1, x2, n_samples)
        ys = np.linspace(y1, y2, n_samples)
        
        # Clip to image bounds
        xs = np.clip(xs, 0, self.width - 1).astype(int)
        ys = np.clip(ys, 0, self.height - 1).astype(int)
        
        values = self.enhanced[ys, xs]
        return values
    
    def compute_detection_metrics(self, classified):
        """Compute aggregate detection metrics."""
        if not classified:
            return {}
        
        candidates = [c for c in classified if c['is_candidate']]
        
        metrics = {
            'total_candidates': len(candidates),
            'total_linear_features': len(classified),
            'mean_confidence': np.mean([c['confidence'] for c in candidates]) if candidates else 0,
            'max_confidence': max([c['confidence'] for c in candidates]) if candidates else 0,
            'mean_length_km': np.mean([c['length_km'] for c in candidates]) if candidates else 0,
            'mean_angle_deg': np.mean([c['angle_deg'] for c in candidates]) if candidates else 0,
            'mean_score': np.mean([c['score'] for c in candidates]) if candidates else 0,
        }
        
        self.metrics = metrics
        return metrics
    
    def visualize_results(self, output_path=None, title="Contrail Detection"):
        """Generate comprehensive visualization of detection results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Panel (a): Original image
        ax = axes[0, 0]
        ax.imshow(self.original)
        ax.set_title('(a) Original Image')
        ax.axis('off')
        
        # Panel (b): Enhanced (high-pass filtered)
        ax = axes[0, 1]
        vmax = max(abs(self.enhanced.min()), abs(self.enhanced.max()))
        im = ax.imshow(self.enhanced, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title('(b) Enhanced (High-Pass)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Panel (c): Sinogram (Radon transform)
        ax = axes[0, 2]
        extent = list(self.theta[:]) + [0, self.sinogram.shape[0]]
        im = ax.imshow(self.sinogram, aspect='auto', extent=extent,
                       cmap='hot', interpolation='nearest')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('rho (pixels)')
        ax.set_title('(c) Radon Transform (Sinogram)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Panel (d): Detection overlay on original
        ax = axes[1, 0]
        ax.imshow(self.original)
        
        if hasattr(self, 'classified') and self.classified:
            colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(self.classified)+1)[1:])
            for i, det in enumerate(self.classified[:20]):  # Top 20
                x1, y1, x2, y2 = det['endpoints']
                alpha = min(det['confidence'], 1.0)
                lw = 1 + 3 * det['confidence']
                ax.plot([x1, x2], [y1, y2], color=colors[i], 
                       linewidth=lw, alpha=alpha,
                       label=f'#{det["id"]} C={det["confidence"]:.2f}' if i < 5 else '')
        
        ax.set_title(f'(d) Top Detections ({len(self.classified)} found)')
        ax.axis('off')
        
        # Panel (e): Edge detection
        ax = axes[1, 1]
        if hasattr(self, 'edge_skeleton'):
            ax.imshow(self.gray, cmap='gray')
            ax.contour(self.edge_skeleton, levels=[0.5], colors='red', linewidths=0.5)
            ax.set_title('(e) Edge Skeleton Overlay')
        else:
            ax.text(0.5, 0.5, 'No edge detection run', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('(e) Edge Detection')
        ax.axis('off')
        
        # Panel (f): Metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        
        if hasattr(self, 'metrics') and self.metrics:
            text_lines = [
                "=== Detection Metrics ===",
                "",
                f"Sensor: {self.sensor}",
                f"GSD: {self.gsd:.0f}m" if self.gsd else "GSD: unknown",
                f"Image: {self.shape_desc()}",
                "",
                f"Linear features: {self.metrics.get('total_linear_features', 0)}",
                f"Candidates (>0.3): {self.metrics.get('total_candidates', 0)}",
                f"Mean confidence: {self.metrics.get('mean_confidence', 0):.3f}",
                f"Max confidence: {self.metrics.get('max_confidence', 0):.3f}",
                f"Mean length: {self.metrics.get('mean_length_km', 0):.1f} km",
                f"Mean angle: {self.metrics.get('mean_angle_deg', 0):.1f} deg",
            ]
            
            if self.classified:
                text_lines.extend([
                    "",
                    "--- Top 5 Candidates ---",
                ])
                for det in self.classified[:5]:
                    text_lines.append(
                        f"#{det['id']}: C={det['confidence']:.3f} "
                        f"L={det['length_km']:.0f}km "
                        f"A={det['angle_deg']:.1f}deg "
                        f"S={det['score']:.1f}"
                    )
            
            ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_title('(f) Summary')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.filepath or '.'), 
                                       'output', 'contrail_detection.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[SAVE] Detection visualization: {output_path}")
        return output_path


# =============================================================================
# MH370-Specific Analysis
# =============================================================================

class MH370ContrailAnalysis:
    """
    Complete analysis pipeline for MH370 contrail detection.
    
    MH370 flight timeline (UTC):
    - 16:41 Takeoff from KUL
    - 16:52 Last ACARS
    - 17:02 Last secondary radar contact
    - 17:21 Last military radar (Pulau Perak area)
    - 17:22 Turn south into Indian Ocean
    - 18:25 Partial handshake (possible turn?)
    - 18:40 SATCOM logon request
    - 19:41 Unanswered ground-to-air call
    - 00:19 UTC (Mar 8) Final handshake (7th Arc)
    
    Key question: Did any satellite capture the aircraft's contrail during
    its flight over the Indian Ocean?
    """
    
    FLIGHT_PATH_WAYPOINTS = [
        ('KUL', 2.75, 101.71, '16:41'),     # Kuala Lumpur takeoff
        ('IGARI', 6.93, 103.59, '17:02'),   # Last ATC contact
        ('Penang', 5.30, 100.20, '17:30'),  # Penang area
        ('MEKAR', 6.80, 97.60, '18:00'),    # Andaman Sea
        ('Arc7_est', -35.5, 95.5, '00:19'), # Estimated 7th Arc endpoint
    ]
    
    def __init__(self):
        self.detection_results = {}
        self.overpass_windows = {}
    
    def analyze_all_overpasses(self, target_lat=-35.5, target_lon=95.5):
        """
        Analyze all satellite overpass opportunities during MH370 flight.
        """
        results = {}
        
        # MODIS Terra
        terra = OverpassCalculator.modis_overpass_times(target_lat, target_lon, '2014-03-08')
        results['MODIS_Terra'] = {
            'overpass': terra,
            'sensor_params': SensorModel.SENSORS['MODIS_Terra'],
            'gsd': SensorModel.gsd_at_location('MODIS_Terra', target_lat, target_lon),
            'feasibility': self._assess_feasibility(terra, 'VIS'),
        }
        
        # MODIS Aqua
        aqua = OverpassCalculator.modis_overpass_times(target_lat, target_lon, '2014-03-08')
        results['MODIS_Aqua'] = {
            'overpass': aqua,
            'sensor_params': SensorModel.SENSORS['MODIS_Terra'],
            'gsd': SensorModel.gsd_at_location('MODIS_Terra', target_lat, target_lon),
            'feasibility': self._assess_feasibility(aqua, 'VIS'),
        }
        
        # VIIRS
        viirs = OverpassCalculator.viirs_overpass_times(target_lat, target_lon, '2014-03-08')
        results['VIIRS'] = {
            'overpass': viirs,
            'sensor_params': SensorModel.SENSORS['VIIRS'],
            'gsd': SensorModel.gsd_at_location('VIIRS', target_lat, target_lon),
            'feasibility': self._assess_feasibility(viirs, 'DNB'),
        }
        
        # MTSAT-2
        results['MTSAT2'] = {
            'overpass': {'continuous_coverage': True, 'cadence_min': 30},
            'sensor_params': SensorModel.SENSORS['MTSAT2'],
            'gsd': SensorModel.gsd_at_location('MTSAT2', target_lat, target_lon),
            'feasibility': self._assess_geo_feasibility(target_lat, target_lon),
        }
        
        self.overpass_windows = results
        return results
    
    def _assess_feasibility(self, overpass, mode='VIS'):
        """Assess whether an overpass could detect a contrail."""
        day_utc = overpass.get('terra_day_utc') or overpass.get('day_utc', 0)
        night_utc = overpass.get('terra_night_utc') or overpass.get('night_utc', 0)
        
        if mode == 'VIS':
            # Need daylight
            feasible = 6 < day_utc < 18
            reason = 'Daylight' if feasible else 'Nighttime (no VIS)'
        elif mode == 'DNB':
            # Needs moonlight
            feasible = True  # Always possible with moonlight
            reason = 'Moonlight-dependent'
        elif mode == 'IR':
            feasible = True  # IR works day and night
            reason = 'Thermal IR always available'
        else:
            feasible = False
            reason = 'Unknown mode'
        
        return {'feasible': feasible, 'reason': reason, 'mode': mode}
    
    def _assess_geo_feasibility(self, lat, lon):
        """Assess GEO satellite feasibility."""
        gsd = SensorModel.gsd_at_location('MTSAT2', lat, lon)
        
        return {
            'feasible': gsd < 3000,  # Under 3km might work
            'reason': f'GSD={gsd:.0f}m at target',
            'gsd_m': gsd,
        }
    
    def generate_report(self, output_path=None):
        """Generate comprehensive analysis report."""
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), 'mh370_contrail_report.txt')
        
        lines = [
            "=" * 72,
            "MH370 CONTRAIL DETECTION ANALYSIS REPORT",
            "=" * 72,
            f"Generated: 2026-05-20",
            f"Target: Estimated crash site (~35.5S, 95.5E)",
            "",
            "-" * 72,
            "1. FLIGHT TIMELINE & SATELLITE COVERAGE",
            "-" * 72,
            "",
            "MH370 flew primarily during NIGHTTIME hours (UTC 16:41 - 00:19).",
            "This severely limits visible-light contrail detection.",
            "",
        ]
        
        for name, info in self.overpass_windows.items():
            lines.extend([
                f"",
                f"--- {name} ---",
                f"GSD at target: {info['gsd']:.0f}m",
                f"Feasibility: {info['feasibility']['feasible']} - {info['feasibility']['reason']}",
            ])
            
            ov = info.get('overpass', {})
            if isinstance(ov, dict):
                for k, v in ov.items():
                    if k != 'continuous_coverage':
                        lines.append(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        
        lines.extend([
            "",
            "-" * 72,
            "2. PHYSICAL ASSESSMENT",
            "-" * 72,
            "",
            f"Contrail formation probability at FL350 (10.7km): ",
            f"{AtmosphericModel.contrail_formation_probability(10.7, -35.5, 3)*100:.0f}%",
            "",
            f"Contrail width after 10 minutes: ",
            f"{ContrailPhysics.initial_contrail_width(age_seconds=600)/1000:.1f} km",
            "",
            f"Contrail optical depth after 10 minutes: ",
            f"{ContrailPhysics.contrail_optical_depth(age_seconds=600):.2f}",
            "",
            "-" * 72,
            "3. CONCLUSION",
            "-" * 72,
            "",
            "Primary obstacle: NIGHTTIME flight.",
            "Visible-light sensors (MODIS VIS, MTSAT-2 VIS) cannot see contrails",
            "in darkness. Thermal IR has insufficient resolution (1-4km) to resolve",
            "individual contrail lines against cloud backgrounds.",
            "",
            "The ONLY potentially viable path is VIIRS DNB (Day-Night Band) with",
            "moonlight illumination, but this requires:",
            "  1. Moon above horizon during flight window",
            "  2. Sufficient moon phase (>30%)",
            "  3. Clear or partially clear sky below contrail altitude",
            "  4. No competing cloud features creating similar linear patterns",
            "",
            "Estimated overall probability of successful contrail detection:",
            "< 1% (dominated by nighttime + resolution constraints)",
            "",
            "=" * 72,
        ])
        
        report = '\n'.join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[REPORT] {output_path}")
        return report


# =============================================================================
# Validation: Test on Recent Image with Known Air Traffic
# =============================================================================

def run_validation_test():
    """
    Run detection on a recent MODIS image of a busy air corridor.
    
    Target: North Atlantic ( busiest oceanic airspace in the world )
    Date: Most recent available from GIBS
    Region: 40N-55N, 10W-30W (North Atlantic Tracks / NAT)
    """
    print("=" * 60)
    print("CONTRAIL DETECTION VALIDATION TEST")
    print("=" * 60)
    
    # North Atlantic - one of the busiest air corridors globally
    # NAT tracks carry 1000+ flights per day
    nat_bbox = [-30, 40, -5, 55]  # W, S, E, N
    test_date = '2026-05-18'       # Recent date for validation
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Fetch true-color image
    print("\n--- Step 1: Fetch MODIS True Color ---")
    vis_path = fetch_worldview_image(
        nat_bbox, test_date,
        layer='MODIS_Terra_CorrectedReflectance_TrueColor',
        width=1600, height=1000,
        output_dir=output_dir
    )
    
    if vis_path is None:
        print("[ERROR] Could not fetch validation image")
        return None
    
    # Also fetch IR for comparison
    print("\n--- Step 2: Fetch MODIS IR (Bands 7-2-1) ---")
    ir_path = fetch_modis_ir_image(nat_bbox, test_date, output_dir=output_dir)
    
    # Step 3: Run detector on VIS image
    print("\n--- Step 3: Run Contrail Detector (VIS) ---")
    detector = ContrailDetector(
        sensor_name='MODIS_Terra',
        target_lat=47.5,  # Center of NAT region
        target_lon=-17.5
    )
    
    detector.load_image(vis_path)
    detector.preprocess()
    
    # Radon-based linear feature detection
    detections = detector.detect_linear_features_radon(
        theta_range=(-5, 95),
        n_theta=180,
        threshold_sigma=2.0
    )
    
    # Edge detection complement
    detector.detect_edges_canny(sigma=2.0, low_thresh=0.15, high_thresh=0.35)
    
    # Classify
    classified = detector.classify_contrails(detections, min_score=3.0)
    
    # Metrics
    metrics = detector.compute_detection_metrics(classified)
    
    # Visualize
    vis_output = os.path.join(output_dir, 'output', 'validation_nat_vis.png')
    detector.visualize_results(vis_output, 
                               title=f'Contrail Validation: NAT {test_date}')
    
    # Step 4: Run detector on IR image if available
    if ir_path and os.path.exists(ir_path):
        print("\n--- Step 4: Run Contrail Detector (IR) ---")
        detector_ir = ContrailDetector(
            sensor_name='MODIS_Terra',
            target_lat=47.5,
            target_lon=-17.5
        )
        detector_ir.load_image(ir_path)
        detector_ir.preprocess()
        detections_ir = detector_ir.detect_linear_features_radon(threshold_sigma=2.0)
        detector_ir.detect_edges_canny(sigma=2.0)
        classified_ir = detector_ir.classify_contrails(detections_ir, min_score=3.0)
        metrics_ir = detector_ir.compute_detection_metrics(classified_ir)
        
        ir_output = os.path.join(output_dir, 'output', 'validation_nat_ir.png')
        detector_ir.visualize_results(ir_output,
                                       title=f'Contrail Validation (IR): NAT {test_date}')
    
    # Step 5: Generate summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nRegion: North Atlantic ({test_date})")
    print(f"Image: {detector.shape_desc()}")
    print(f"GSD: {detector.gsd:.0f}m")
    print(f"\nVIS Detection:")
    print(f"  Total linear features: {metrics.get('total_linear_features', 0)}")
    print(f"  Contrail candidates: {metrics.get('total_candidates', 0)}")
    print(f"  Mean confidence: {metrics.get('mean_confidence', 0):.3f}")
    print(f"  Max confidence: {metrics.get('max_confidence', 0):.3f}")
    
    if classified:
        print(f"\n  Top 5 candidates:")
        for det in classified[:5]:
            print(f"    #{det['id']}: C={det['confidence']:.3f} "
                  f"L={det['length_km']:.0f}km A={det['angle_deg']:.1f}deg "
                  f"profile_mean={det['profile_mean']:+.3f}")
    
    # Save results JSON
    results = {
        'validation_date': test_date,
        'region': 'North Atlantic (NAT)',
        'bbox': nat_bbox,
        'vis_metrics': metrics,
        'top_candidates': [
            {k: v for k, v in det.items() if k != 'profile'}
            for det in classified[:10]
        ] if classified else [],
    }
    
    results_path = os.path.join(output_dir, 'output', 'validation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return {
        'detector': detector,
        'classified': classified,
        'metrics': metrics,
        'vis_output': vis_output,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Contrail Detection Module')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation test on recent NAT imagery')
    parser.add_argument('--mh370', action='store_true',
                       help='Run MH370-specific analysis')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to custom image to analyze')
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    
    args = parser.parse_args()
    
    if args.validate:
        run_validation_test()
    elif args.mh370:
        analysis = MH370ContrailAnalysis()
        analysis.analyze_all_overpasses()
        report = analysis.generate_report()
        print(report)
    elif args.image:
        detector = ContrailDetector(
            sensor_name='MODIS_Terra',
            target_lat=args.lat or 0,
            target_lon=args.lon or 0
        )
        detector.load_image(args.image)
        detector.preprocess()
        detections = detector.detect_linear_features_radon()
        detector.detect_edges_canny()
        classified = detector.classify_contrails(detections)
        detector.compute_detection_metrics(classified)
        detector.visualize_results()
    else:
        # Default: run validation
        run_validation_test()
