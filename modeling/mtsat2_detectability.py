#!/usr/bin/env python3
"""MTSAT-2 (Himawari-7) Visible-Light Debris Detectability Simulation for MH370.

Simulates whether the MTSAT-2 geostationary weather satellite at 145E could have 
detected MH370 debris field on the ocean surface on March 8, 2014.

Key physics:
- GEO viewing geometry: satellite at 145E/0N, target at 96E/-35S
- Solar geometry: March 8 near equinox, sun declination ~ -4 deg
- Ocean reflectance: low (~3-5%) due to Fresnel reflection
- Debris reflectance: metal (high), fuel slick (low), foam (moderate)
- Pixel size degradation with off-nadir angle
- Atmospheric path radiance at 58 deg zenith angle
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# CONSTANTS: MTSAT-2 / Himawari-7 Parameters
# =====================================================================

# Satellite position
GEO_SAT_LON = 145.0   # degrees East
GEO_SAT_LAT = 0.0     # equatorial

# Search area center (7th arc peak probability)
TARGET_LON = 95.5
TARGET_LAT = -35.5

# MTSAT-2 JAMI instrument parameters
VIS_WAVELENGTH = 0.65  # um, central wavelength of VIS band (0.55-0.80)
VIS_BANDWIDTH = 0.25   # um
IFOV_NADIR = 28e-6     # rad, instantaneous FOV at nadir (28 urad)
ALTITUDE_GEO = 35786   # km

# Derived pixel size
PIXEL_NADIR_KM = ALTITUDE_GEO * IFOV_NADIR  # ~1.0 km at sub-satellite point

# Detector specs
DETECTOR_ROWS = 1024   # VIS detector array size
DETECTOR_COLS = 1024
QUANTIZATION_BITS = 10 # 10-bit ADC
DYNAMIC_RANGE = 1024   # DN levels (0-1023)

# Noise parameters
READ_NOISE_DN = 2      # read noise in DN
SHOT_NOISE_FACTOR = 0.03  # shot noise as fraction of signal
FIXED_PATTERN_NOISE = 0.01  # 1% FPN
STRAY_LIGHT_FRACTION = 0.05  # stray light fraction


# =====================================================================
# GEOMETRY ENGINE
# =====================================================================

def great_circle_distance(lat1, lon1, lat2, lon2):
    """Great circle distance in km using Haversine formula."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r)*np.cos(lat2_r)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def geosat_zenith_angle(target_lat, target_lon, sat_lon=145):
    """Zenith angle of geosat from target point (degrees)."""
    # Geosat is at (sat_lon, 0), target at (target_lon, target_lat)
    # Use law of cosines on unit sphere
    lat_t = np.radians(target_lat)
    lon_diff = np.radians(target_lon - sat_lon)
    
    cos_z = np.cos(lat_t) * np.cos(lon_diff)
    cos_z = np.clip(cos_z, -1, 1)
    return np.degrees(np.arccos(cos_z))


def pixel_size_at_target(target_lat, target_lon, sat_lon=145):
    """Pixel ground sample distance (km) at target location."""
    z = geosat_zenith_angle(target_lat, target_lon, sat_lon)
    z_rad = np.radians(z)
    # GSD increases as 1/cos(z) approximately
    return PIXEL_NADIR_KM / np.cos(z_rad)


def solar_geometry(target_lat, target_lon, utc_hour, date_day_of_year=67):
    """
    Compute solar angles at target.
    
    Returns:
        solar_elev: elevation angle above horizon (deg)
        solar_azim: azimuth from north (deg)
        cos_solar_zenith: cosine of solar zenith angle
        day_fraction: 0-1, fraction of daylight hours
    """
    # Solar declination (approximate for March 8, DOY=67)
    # Using simplified equation of time
    dec = 23.44 * np.sin(np.radians(360 * (284 + date_day_of_year) / 365))
    dec_rad = np.radians(dec)
    
    # Hour angle
    lst = utc_hour + target_lon / 15  # local solar time
    ha = np.radians((lst - 12) * 15)  # hour angle
    
    # Latitude
    lat_rad = np.radians(target_lat)
    
    # Solar elevation
    sin_elev = (np.sin(dec_rad) * np.sin(lat_rad) + 
                np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha))
    sin_elev = np.clip(sin_elev, -1, 1)
    elev = np.degrees(np.arcsin(sin_elev))
    
    # Solar azimuth (simplified)
    cos_azim = ((np.sin(dec_rad) - np.sin(lat_rad) * sin_elev) /
                (np.cos(lat_rad) * np.cos(np.radians(elev)) + 1e-10))
    cos_azim = np.clip(cos_azim, -1, 1)
    azim = np.degrees(np.arccos(cos_azim))
    if ha > 0:
        azim = 360 - azim
    
    return {
        'elevation': max(elev, -90),
        'azimuth': azim,
        'cos_zenith': sin_elev,
        'zenith': 90 - max(elev, -90),
        'is_day': elev > 0,
    }


def atmospheric_path_radiance(zenith_angle_deg, wavelength_um=0.65,
                              visibility_km=23, aerosol_optical_depth=0.05):
    """
    Compute atmospheric path radiance and transmittance.
    
    Uses simplified MODTRAN-like parameterization for maritime atmosphere.
    
    Returns:
        tau: total transmittance (0-1)
        L_path: path radiance (W/m^2/sr/um)
        L_total: total radiance reaching sensor
    """
    z = np.radians(min(zenith_angle_deg, 85))
    
    # Air mass factor (Kasten-Young approximation)
    amf = 1 / (np.cos(z) + 0.50572 * (96.07995 - np.degrees(z))**(-1.6364))
    if zenith_angle_deg > 88:
        amf = 40  # cap for extreme angles
    
    # Rayleigh optical depth at sea level
    tau_rayleigh = 0.008569 * wavelength_um**(-4) * (1 + 0.0113*wavelength_um**(-2) + 0.00013*wavelength_um**(-4))
    
    # Aerosol optical depth (maritime clean)
    tau_aerosol = aerosol_optical_depth
    
    # Total optical depth
    tau_total = (tau_rayleigh + tau_aerosol) * amf
    
    # Transmittance
    tau = np.exp(-tau_total)
    
    # Path radiance (single scattering approximation)
    # Simplified: path radiance proportional to scattered sunlight
    E_sun = 1361  # W/m^2 solar constant
    omega_0 = 0.9  # single scattering albedo (aerosols)
    phase_func = 0.75 * (1 + np.cos(z)**2)  # Rayleigh phase function
    
    L_path = (E_sun * omega_0 * (1 - np.exp(-tau_total)) * phase_func / 
               (4 * np.pi * np.cos(z) + 1e-10))
    L_path *= 1e-6  # scale to reasonable units
    
    return {
        'transmittance': min(tau, 1.0),
        'path_radiance': L_path,
        'optical_depth': tau_total,
        'air_mass': amf,
    }


# =====================================================================
# SURFACE REFLECTANCE MODEL
# =====================================================================

def ocean_reflectance(solar_zenith_deg, view_zenith_deg, relative_azimuth_deg,
                      wind_speed_ms=5.0, wavelength_um=0.65):
    """
    Ocean surface BRDF using Cox-Munk wave slope model.
    
    Ocean appears dark in visible light except for sun glint region.
    
    Returns:
        rho_ocean: bidirectional reflectance factor (0-1)
    """
    theta_i = np.radians(solar_zenith_deg)
    theta_v = np.radians(view_zenith_deg)
    phi = np.radians(relative_azimuth_deg)
    
    # Wind-induced surface roughness (Cox-Munk)
    sigma_sq = 0.003 + 0.00512 * wind_speed_ms  # mean square slope
    
    # Facet normal distribution (Gaussian)
    # ... simplified: use empirical model
    
    # Sun glint condition: specular reflection direction
    # Specular facet has normal bisecting sun-view vector
    cos_theta_n = -(np.cos(theta_i) * np.cos(theta_v) +
                    np.sin(theta_i) * np.sin(theta_v) * np.cos(phi))
    cos_theta_n = np.clip(cos_theta_n, -1, 1)
    theta_n = np.arccos(abs(cos_theta_n))
    
    # Probability density of facets oriented for glint
    p_facet = (1 / (np.pi * sigma_sq)) * np.exp(-(np.tan(theta_n)**2) / sigma_sq)
    
    # Fresnel reflectance at the specular angle
    n_water = 1.33  # refractive index
    theta_specular = theta_n
    cos_ts = np.cos(theta_specular)
    sin_ts = np.sin(theta_specular)
    
    # Fresnel equations (unpolarized)
    n_ratio = n_water
    rs = ((n_ratio * cos_ts - np.sqrt(n_ratio**2 - sin_ts**2)) /
          (n_ratio * cos_ts + np.sqrt(n_ratio**2 - sin_ts**2) + 1e-10))**2
    rp = ((np.sqrt(n_ratio**2 - sin_ts**2) - n_ratio * cos_ts) /
          (np.sqrt(n_ratio**2 - sin_ts**2) + n_ratio * cos_ts + 1e-10))**2
    rho_fresnel = (rs + rp) / 2
    
    # Total ocean reflectance = glint + whitecap + volume scattering
    rho_glint = p_facet * rho_fresnel / (4 * np.cos(theta_v) * np.cos(theta_i) + 1e-10)
    rho_glint *= 1e4  # scale factor for numerical stability
    
    # Whitecap contribution (foam)
    wc_frac = 2.51e-4 * wind_speed_ms**3.11  # whitecap coverage fraction
    rho_whitecap = wc_frac * 0.5  # foam reflectance ~50%
    
    # Volume scattering (water body, very small in visible)
    rho_volume = 0.02  # ~2% diffuse reflectance from water column
    
    rho_total = min(rho_glint + rho_whitecap + rho_volume, 1.0)
    
    return {
        'reflectance': rho_total,
        'glint': min(rho_glint, 1.0),
        'whitecap': rho_whitecap,
        'volume': rho_volume,
        'is_glint_zone': rho_glint > 0.1,
    }


def debris_reflectance_model(debris_type='metal'):
    """
    Spectral reflectance of debris types.
    
    Returns:
        rho_debris: hemispherical reflectance (0-1)
        description: physical description
    """
    models = {
        'metal_aluminum': {
            'reflectance': 0.70,  # bare aluminum, high specularity
            'specular_fraction': 0.8,
            'description': 'Bare aluminum fuselage skin',
        },
        'metal_painted': {
            'reflectance': 0.35,  # painted aircraft aluminum
            'specular_fraction': 0.3,
            'description': 'Painted fuselage (white/grey)',
        },
        'composite': {
            'reflectance': 0.25,  # CFRP composite
            'specular_fraction': 0.1,
            'description': 'Composite structure (tail, wing)',
        },
        'fuel_slick': {
            'reflectance': 0.02,  # dark oil film
            'specular_fraction': 0.05,
            'description': 'Jet-A1 fuel slick on water',
        },
        'foam_debris': {
            'reflectance': 0.45,  # flotation foam, seat cushions
            'specular_fraction': 0.05,
            'description': 'White flotation foam, life raft',
        },
        'fabric': {
            'reflectance': 0.18,  # interior fabrics, carpet
            'specular_fraction': 0.02,
            'description': 'Interior fabric, carpet, clothing',
        },
        'luggage': {
            'reflectance': 0.22,  # colored luggage
            'specular_fraction': 0.08,
            'description': 'Passenger luggage, bags',
        },
    }
    return models.get(debris_type, models['metal_painted'])


# =====================================================================
# RADIATIVE TRANSFER & SIGNAL MODEL
# =====================================================================

def compute_pixel_signal(reflectance, atm, solar_geom, pixel_area_km2=1.0):
    """
    Compute top-of-atmosphere radiance and DN value for a pixel.
    
    Radiative transfer equation:
      L_TOA = L_path + (tau * E_sun * cos(sza) * rho / pi)
      
    Then convert to DN through sensor response.
    """
    E_sun = 1361  # W/m^2 solar irradiance
    cos_sza = solar_geom['cos_zenith']
    
    if cos_sza <= 0:
        return {'radiance': 0, 'dn': 0, 'snr': 0}
    
    # Surface-reflected component
    L_surface = (atm['transmittance']**2 * E_sun * cos_sza * reflectance / np.pi)
    
    # Total TOA radiance
    L_toa = atm['path_radiance'] + L_surface
    
    # Convert to DN (simplified linear response)
    # Assume gain such that full-scale DN=1023 corresponds to bright cloud
    L_cloud_max = (atm['transmittance']**2 * E_sun * 0.8 / np.pi)  # 80% cloud
    gain = DYNAMIC_RANGE / L_cloud_max
    
    dn = int(np.clip(L_toa * gain, 0, DYNAMIC_RANGE - 1))
    
    # SNR estimation
    signal_dn = dn
    noise_dn = np.sqrt(READ_NOISE_DN**2 + (SHOT_NOISE_FACTOR * signal_dn)**2 + 
                       (FIXED_PATTERN_NOISE * signal_dn)**2)
    snr = signal_dn / (noise_dn + 1e-10)
    
    return {
        'radiance': L_toa,
        'dn': dn,
        'snr': snr,
        'signal_dn': signal_dn,
        'noise_dn': noise_dn,
    }


def simulate_mtsat_image(debris_field_params, utc_hours=None, wind_speed=5.0,
                         image_size_px=64, sub_pixel_samples=30):
    """
    Simulate an MTSAT-2 VIS image of the search area with debris field.

    Uses per-pixel sub-sampling (Monte Carlo or regular grid within each
    debis-covered pixel) to properly model fractional pixel coverage.

    Parameters:
        debris_field_params: dict with debris components
        utc_hours: list of UTC hours to simulate
        wind_speed: m/s
        image_size_px: output image resolution (scene = image_size_px * GSD km)
        sub_pixel_samples: sub-samples per pixel within debris region
    """
    if utc_hours is None:
        utc_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    
    # Geometry
    z = geosat_zenith_angle(TARGET_LAT, TARGET_LON)
    gsd_km = pixel_size_at_target(TARGET_LAT, TARGET_LON)
    
    # Image extent (km)
    extent_km = image_size_px * gsd_km
    half_extent = extent_km / 2
    
    # Coordinate grids (output level)
    x_px = np.linspace(-half_extent + gsd_km/2, half_extent - gsd_km/2, image_size_px)  # pixel centers
    y_px = np.linspace(-half_extent + gsd_km/2, half_extent - gsd_km/2, image_size_px)
    
    results = {}
    
    for utc_h in utc_hours:
        sol = solar_geometry(TARGET_LAT, TARGET_LON, utc_h)
        
        if not sol['is_day']:
            results[utc_h] = {'image': None, 'is_night': True, 'metrics': None}
            continue
        
        # Relative azimuth: sun-to-target-to-satellite
        rel_azim = abs(sol['azimuth'] - satellite_azimuth(TARGET_LAT, TARGET_LON))
        
        atm = atmospheric_path_radiance(z)
        oc = ocean_reflectance(sol['zenith'], z, rel_azim, wind_speed)
        
        # Background ocean
        bg_rho = oc['reflectance']
        bg_signal = compute_pixel_signal(bg_rho, atm, sol)
        
        # Initialize output image with ocean background
        img = np.full((image_size_px, image_size_px), bg_signal['dn'], dtype=float)
        truth_mask = np.zeros((image_size_px, image_size_px), dtype=bool)
        frac_coverage = np.zeros((image_size_px, image_size_px))  # fraction of pixel covered by debris
        
        # Process each debris component: compute area overlap with output pixels
        total_debris_subpixels = 0
        total_hits = 0
        debris_signals = []
        
        for comp in debris_field_params.get('components', []):
            ctype = comp['type']
            cx_km = comp.get('center_x_km', 0)
            cy_km = comp.get('center_y_km', 0)
            semi_major_km = comp.get('semi_major_km', 0.5)
            semi_minor_km = comp.get('semi_minor_km', 0.3)
            rotation_deg = comp.get('rotation_deg', 0)
            
            dm = debris_reflectance_model(ctype)
            rho_d = dm['reflectance']
            
            # Find pixels that could intersect this debris component
            cos_r = np.cos(np.radians(rotation_deg))
            sin_r = np.sin(np.radians(rotation_deg))
            
            # Bounding box of the rotated ellipse in km
            a = semi_major_km
            b = semi_minor_km
            bbox_half = np.sqrt((a*cos_r)**2 + (b*sin_r)**2)  # projected half-width
            
            x_min = cx_km - bbox_half
            x_max = cx_km + bbox_half
            y_min = cy_km - bbox_half
            y_max = cy_km + bbox_half
            
            # Pixel indices overlapping the bounding box
            px_min = max(0, int(np.floor((x_min + half_extent) / gsd_km)))
            px_max = min(image_size_px - 1, int(np.ceil((x_max + half_extent) / gsd_km)))
            py_min = max(0, int(np.floor((y_min + half_extent) / gsd_km)))
            py_max = min(image_size_px - 1, int(np.ceil((y_max + half_extent) / gsd_km)))
            
            comp_hits = 0
            comp_samples = 0
            
            for pi in range(py_min, py_max + 1):
                pcy = y_px[pi]  # pixel center y
                for pj in range(px_min, px_max + 1):
                    pcx = x_px[pj]  # pixel center x
                    
                    # Sub-sample this pixel with a fine grid
                    # Sub-pixel spacing within this pixel
                    subs = sub_pixel_samples
                    dx = gsd_km / subs
                    sx = np.linspace(pcx - gsd_km/2 + dx/2, pcx + gsd_km/2 - dx/2, subs)
                    sy = np.linspace(pcy - gsd_km/2 + dx/2, pcy + gsd_km/2 - dx/2, subs)
                    sxx, syy = np.meshgrid(sx, sy)
                    
                    # Rotate sub-sample coordinates into ellipse frame
                    rx = (sxx - cx_km) * cos_r + (syy - cy_km) * sin_r
                    ry = -(sxx - cx_km) * sin_r + (syy - cy_km) * cos_r
                    
                    in_ellipse = (rx**2 / a**2 + ry**2 / b**2) <= 1
                    hit_count = int(in_ellipse.sum())
                    total_count = subs * subs
                    
                    comp_hits += hit_count
                    comp_samples += total_count
                    
                    if hit_count > 0:
                        truth_mask[pi, pj] = True
                        frac_coverage[pi, pj] += hit_count / total_count
            
            debris_hit_fraction = comp_hits / max(comp_samples, 1)
            total_hits += comp_hits
            total_debris_subpixels += comp_samples
            
            debris_signals.append({
                'type': ctype,
                'subpixel_samples': comp_samples,
                'subpixel_hits': comp_hits,
                'area_km2_effective': debris_hit_fraction * semi_major_km * semi_minor_km * np.pi * gsd_km**2,
                'signal_dn': compute_pixel_signal(rho_d, atm, sol)['signal_dn'],
                'background_dn': bg_signal['dn'],
                'contrast_at_subpixel': compute_pixel_signal(rho_d, atm, sol)['signal_dn'] - bg_signal['dn'],
                'snr': compute_pixel_signal(rho_d, atm, sol)['snr'],
                'reflectance': rho_d,
                'description': dm['description'],
            })
        
        # Apply area-weighted mixing to pixels that contain debris
        for pi in range(image_size_px):
            for pj in range(image_size_px):
                if truth_mask[pi, pj]:
                    f = min(frac_coverage[pi, pj], 1.0)
                    # Mixed reflectance: area-weighted average
                    mixed_rho = f * max(d['reflectance'] for d in debris_signals) + (1 - f) * bg_rho
                    # Use median debris reflectance for the pixel
                    debris_rhos = [d['reflectance'] for d in debris_signals]
                    avg_debris_rho = np.mean(debris_rhos)
                    mixed_rho = f * avg_debris_rho + (1 - f) * bg_rho
                    mixed_signal = compute_pixel_signal(mixed_rho, atm, sol)
                    img[pi, pj] = mixed_signal['dn']
        
        # Add sensor noise
        noise = np.random.normal(0, READ_NOISE_DN + SHOT_NOISE_FACTOR * np.abs(img) + 
                                 FIXED_PATTERN_NOISE * np.abs(img),
                                 img.shape)
        noisy_img = np.clip(img + noise, 0, DYNAMIC_RANGE - 1)
        
        # Detection analysis
        contrast = np.max(noisy_img[truth_mask]) - np.median(noisy_img[~truth_mask]) if truth_mask.any() else 0
        background_std = np.std(noisy_img[~truth_mask])
        t_stat = contrast / (background_std + 1e-10)
        
        # Simple threshold detection
        threshold = np.median(noisy_img) + 3 * background_std
        detected = noisy_img > threshold
        true_positive = (detected & truth_mask).sum()
        false_positive = (detected & ~truth_mask).sum()
        false_negative = (~detected & truth_mask).sum()
        
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        
        results[utc_h] = {
            'image': noisy_img,
            'truth_mask': truth_mask,
            'clean_image': img,
            'is_night': False,
            'solar': sol,
            'atmosphere': atm,
            'ocean': oc,
            'background_dn': bg_signal['dn'],
            'background_std': float(background_std),
            'debris_signals': debris_signals,
            'total_debris_pixels': total_debris_subpixels // (sub_pixel_samples**2),
            'contrast_dn': float(contrast),
            't_statistic': float(t_stat),
            'detection': {
                'threshold_dn': float(threshold),
                'true_positives': int(true_positive),
                'false_positives': int(false_positive),
                'false_negatives': int(false_negative),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': 2*precision*recall/(precision+recall+1e-10),
            },
            'gsd_km': gsd_km,
            'zenith_angle': z,
        }
    
    return results


def satellite_azimuth(target_lat, target_lon, sat_lon=145):
    """Azimuth from target to geosat."""
    lat_t = np.radians(target_lat)
    dlon = np.radians(sat_lon - target_lon)
    
    x = np.sin(dlon) * np.cos(lat_t)
    y = np.cos(np.radians(90)) * np.sin(lat_t) - np.sin(np.radians(90)) * np.cos(lat_t) * np.cos(dlon)
    
    azim = np.degrees(np.arctan2(x, y))
    return (azim + 360) % 360


# =====================================================================
# DEBRIS FIELD CONFIGURATION
# =====================================================================

MH370_DEBRIS_FIELD = {
    'name': 'MH370 B777-200ER Impact Debris Field (2-4 hours post-impact)',
    'date': '2014-03-08',
    'impact_velocity_kts': '~500 kts (nearly level flight)',
    'note': 'Sizes inflated ~3-5x for wave/wind dispersion over 2-4 hours',
    'components': [
        # Main impact zone - spread by wind/waves
        {
            'type': 'fuel_slick',
            'center_x_km': 0,
            'center_y_km': 0,
            'semi_major_km': 0.5,  # ~1km diameter after 2-4h dispersion
            'semi_minor_km': 0.25,
            'rotation_deg': 30,
            'fill_factor': 0.8,
            'note': 'Fuel slick, wind-elongated',
        },
        {
            'type': 'metal_painted',
            'center_x_km': 0.15,
            'center_y_km': -0.2,
            'semi_major_km': 0.3,
            'semi_minor_km': 0.15,
            'rotation_deg': 45,
            'fill_factor': 0.15,
            'note': 'Fuselage fragment cluster',
        },
        {
            'type': 'foam_debris',
            'center_x_km': -0.15,
            'center_y_km': 0.1,
            'semi_major_km': 0.25,
            'semi_minor_km': 0.12,
            'rotation_deg': -20,
            'fill_factor': 0.25,
            'note': 'Flotation items, seat cushions',
        },
        {
            'type': 'composite',
            'center_x_km': 0.3,
            'center_y_km': 0.05,
            'semi_major_km': 0.25,
            'semi_minor_km': 0.12,
            'rotation_deg': 60,
            'fill_factor': 0.10,
            'note': 'Tail/wing composite fragments',
        },
        {
            'type': 'luggage',
            'center_x_km': -0.1,
            'center_y_km': -0.3,
            'semi_major_km': 0.2,
            'semi_minor_km': 0.1,
            'rotation_deg': 10,
            'fill_factor': 0.12,
            'note': 'Floating baggage cluster',
        },
        {
            'type': 'fabric',
            'center_x_km': 0.4,
            'center_y_km': -0.15,
            'semi_major_km': 0.15,
            'semi_minor_km': 0.08,
            'rotation_deg': 80,
            'fill_factor': 0.08,
            'note': 'Interior textiles',
        },
        {
            'type': 'metal_aluminum',
            'center_x_km': -0.3,
            'center_y_km': 0.2,
            'semi_major_km': 0.15,
            'semi_minor_km': 0.08,
            'rotation_deg': -45,
            'fill_factor': 0.20,
            'note': 'Exposed aluminum skin',
        },
    ],
}


# =====================================================================
# MAIN SIMULATION & VISUALIZATION
# =====================================================================

def main():
    print('=' * 70)
    print('MTSAT-2 VIS Debris Detectability Simulation')
    print(f'Target: {TARGET_LAT}S, {TARGET_LON}E | Date: 2014-03-08')
    print(f'Satellite: MTSAT-2 @ {GEO_SAT_LON}E (Himawari-7)')
    print('=' * 70)
    
    # Print geometry
    z = geosat_zenith_angle(TARGET_LAT, TARGET_LON)
    gsd = pixel_size_at_target(TARGET_LAT, TARGET_LON)
    dist = great_circle_distance(GEO_SAT_LAT, GEO_SAT_LON, TARGET_LAT, TARGET_LON)
    
    print(f'\n--- Viewing Geometry ---')
    print(f'  Great circle distance to satellite: {dist:.0f} km')
    print(f'  Zenith angle at target: {z:.1f} deg')
    print(f'  Nadir pixel size: {PIXEL_NADIR_KM:.2f} km')
    print(f'  Target pixel size (GSD): {gsd:.2f} km ({gsd*1000:.0f} m)')
    print(f'  Resolution degradation: {gsd/PIXEL_NADIR_KM:.2f}x')
    
    # Simulate over different hours
    print(f'\n--- Running simulation ---')
    results = simulate_mtsat_image(MH370_DEBRIS_FIELD, 
                                   utc_hours=[0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 18, 21],
                                   wind_speed=5.0,
                                   image_size_px=64)
    
    # Summary table
    print(f'\n{"UTC":>4} | {"LT":>4} | {"Sun Elev":>8} | {"Ocean DN":>8} | '
          f'{"Debris DN":>8} | {"Contrast":>8} | {"SNR":>6} | '
          f'{"TP":>4} {"FP":>4} {"FN":>4} | {"F1":>5}')
    print('-' * 90)
    
    best_f1 = 0
    best_hour = None
    hourly_data = []
    
    for utc_h in sorted(results.keys()):
        r = results[utc_h]
        if r['is_night']:
            print(f'{utc_h:4.0f} | {(utc_h+95/15)%24:4.1f} | NIGHT')
            continue
        
        sol = r['solar']
        det = r['detection']
        
        # Best debris signal
        best_ds = max(r['debris_signals'], key=lambda x: x['snr']) if r['debris_signals'] else None
        
        f1 = det['f1_score']
        hourly_data.append({
            'utc': utc_h,
            'lt': (utc_h + 95/15) % 24,
            'sun_elev': sol['elevation'],
            'ocean_dn': r['background_dn'],
            'debris_dn': best_ds['signal_dn'] if best_ds else 0,
            'contrast': r['contrast_dn'],
            'snr': best_ds['snr'] if best_ds else 0,
            'tp': det['true_positives'],
            'fp': det['false_positives'],
            'fn': det['false_negatives'],
            'f1': f1,
        })
        
        marker = ' ***' if f1 > best_f1 else ''
        if f1 > best_f1:
            best_f1 = f1
            best_hour = utc_h
        
        print(f'{utc_h:4.0f} | {(utc_h+95/15)%24:4.1f} | {sol["elevation"]:8.1f} deg | '
              f'{r["background_dn"]:8.1f} | {best_ds["signal_dn"] if best_ds else 0:8.1f} | '
              f'{r["contrast_dn"]:8.1f} | {best_ds["snr"] if best_ds else 0:6.1f} | '
              f'{det["true_positives"]:4d} {det["false_positives"]:4d} {det["false_negatives"]:4d} | '
              f'{f1:.3f}{marker}')
    
    print(f'\nBest detection hour: UTC {best_hour} (F1={best_f1:.3f})')
    
    # ===== PLOTTING =====
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('MTSAT-2 (Himawari-7) VIS Band Debris Detectability — MH370\n'
                 f'Search Area: {TARGET_LAT}S, {TARGET_LON}E | '
                 f'GSD: {gsd:.1f} km/px | Zenith: {z:.1f}',
                 fontsize=14, fontweight='bold')
    
    # --- Panel (a): Hourly F1 score ---
    ax = fig.add_subplot(2, 3, 1)
    hours = [h['utc'] for h in hourly_data]
    f1s = [h['f1'] for h in hourly_data]
    sun_elevs = [h['sun_elev'] for h in hourly_data]
    
    colors = ['green' if f > 0.3 else ('orange' if f > 0.1 else 'red') for f in f1s]
    bars = ax.bar(hours, f1s, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(0.3, color='blue', ls='--', alpha=0.5, label='F1=0.3 threshold')
    ax.axhline(0.1, color='orange', ls='--', alpha=0.5, label='F1=0.1 threshold')
    ax.set_xlabel('UTC Hour (March 8)')
    ax.set_ylabel('Detection F1 Score')
    ax.set_title('(a) Detection Performance by Hour\n(Green=detectable, Orange=marginal, Red=undetectable)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours)
    
    # Secondary axis: sun elevation
    ax2 = ax.twinx()
    ax2.plot(hours, sun_elevs, 'k-', lw=1.5, marker='o', markersize=3, label='Sun elevation')
    ax2.set_ylabel('Solar Elevation (deg)', color='gray')
    ax2.tick_params(axis='y', colors='gray')
    
    # --- Panel (b): Best-hour simulated image ---
    ax = fig.add_subplot(2, 3, 2)
    if best_hour and best_hour in results:
        r_best = results[best_hour]
        im = ax.imshow(r_best['image'], cmap='gray', vmin=0, vmax=DYNAMIC_RANGE)
        ax.contour(r_best['truth_mask'], levels=[0.5], colors=['red'], linewidths=1)
        ax.set_title(f'(b) Simulated Image @ UTC {best_hour:02.0f}:00\n'
                     f'(Red outline = debris field)')
        plt.colorbar(im, ax=ax, label='DN (0-1023)', shrink=0.8)
    
    # --- Panel (c): Contrast vs Sun Angle ---
    ax = fig.add_subplot(2, 3, 3)
    contrasts = [h['contrast'] for h in hourly_data]
    scatters = []
    for i, h in enumerate(hourly_data):
        s = ax.scatter(h['sun_elev'], h['contrast'], c=f1s[i], cmap='RdYlGn',
                       s=100, vmin=0, vmax=1, edgecolors='black', linewidths=0.5)
        ax.annotate(f'{int(h["utc"])}h', (h['sun_elev'], h['contrast']),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)
    plt.colorbar(s, ax=ax, label='F1 Score')
    ax.set_xlabel('Solar Elevation (deg)')
    ax.set_ylabel('Contrast (DN)')
    ax.set_title('(c) Contrast vs Solar Geometry\n(Higher sun = better contrast)')
    ax.grid(True, alpha=0.3)
    
    # --- Panel (d): Debris component signals ---
    ax = fig.add_subplot(2, 3, 4)
    if best_hour is not None and best_hour in results:
        ds_list = results[best_hour]['debris_signals']
        names = [d['description'][:25] for d in ds_list]
        snrs = [d['snr'] for d in ds_list]
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, snrs, color='steelblue', alpha=0.7, edgecolor='navy')
        
        # Color-code by detectability
        for i, (bar, snr) in enumerate(zip(bars, snrs)):
            if snr > 3:
                bar.set_color('green')
            elif snr > 1:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_title(f'(d) Per-Component SNR @ UTC {best_hour:02.0f}:00\n'
                     f'(Green>3, Orange 1-3, Red<1)')
        ax.axvline(3, color='green', ls=':', alpha=0.5)
        ax.axvline(1, color='orange', ls=':', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # --- Panel (e): Reflectance comparison ---
    ax = fig.add_subplot(2, 3, 5)
    debris_types = ['metal_aluminum', 'metal_painted', 'composite', 'fuel_slick',
                   'foam_debris', 'fabric', 'luggage']
    rhos = [debris_reflectance_model(dt)['reflectance'] for dt in debris_types]
    labels_short = ['Aluminum', 'Painted Al', 'Composite', 'Fuel Slick',
                   'Foam', 'Fabric', 'Luggage']
    
    # Ocean baseline
    oc_base = ocean_reflectance(45, 58, 135, wind_speed_ms=5.0)['reflectance']
    
    x = np.arange(len(labels_short))
    width = 0.35
    bars1 = ax.bar(x - width/2, rhos, width, label='Debris', color='coral', edgecolor='darkred')
    bars2 = ax.bar(x + width/2, [oc_base]*len(debris_types), width, label='Ocean', color='steelblue', edgecolor='navy')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Reflectance')
    ax.set_title('(e) Debris vs Ocean Reflectance\n(Large gap = high contrast)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate contrast ratio
    for i, (rho_d, name) in enumerate(zip(rhos, labels_short)):
        ratio = rho_d / (oc_base + 1e-10)
        ax.annotate(f'{ratio:.1f}x', (i, max(rho_d, oc_base) + 0.02),
                   ha='center', fontsize=7, color='darkgreen' if ratio > 2 else 'gray')
    
    # --- Panel (f): Detection feasibility summary ---
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    
    # Compute summary statistics
    n_detectable = sum(1 for h in hourly_data if h['f1'] > 0.3)
    n_marginal = sum(1 for h in hourly_data if 0.1 < h['f1'] <= 0.3)
    n_undetectable = sum(1 for h in hourly_data if h['f1'] <= 0.1)
    max_contrast = max(h['contrast'] for h in hourly_data) if hourly_data else 0
    max_snr = max(h['snr'] for h in hourly_data) if hourly_data else 0
    
    if best_hour is not None and best_hour in results:
        best_lt = (best_hour + 95/15) % 24
        best_sun_elev = results[best_hour]['solar']['elevation']
        best_case_str = f"""
  BEST CASE (UTC {best_hour:02.0f}:00, LT {best_lt:.1f}h):
    Sun elevation:   {best_sun_elev:.1f} deg
    Max contrast:    {max_contrast:.1f} DN
    Max SNR:         {max_snr:.1f}
    F1 Score:        {best_f1:.3f}"""
    else:
        best_case_str = f"""
  BEST CASE:
    No detections achieved at any hour.
    Max contrast:    {max_contrast:.1f} DN
    Max SNR:         {max_snr:.1f}"""
    
    summary_text = f"""
  MTSAT-2 VIS DETECTABILITY SUMMARY
  {'='*42}

  GEOMETRY:
    Satellite:       MTSAT-2 (Himawari-7) @ 145E
    Target:          {TARGET_LAT}S, {TARGET_LON}E
    Zenith angle:    {z:.1f} deg
    Ground sampling: {gsd:.2f} km ({gsd*1000:.0f} m)/pixel
    Degradation:     {gsd/PIXEL_NADIR_KM:.2f}x from nadir
{best_case_str}

  HOURLY BREAKDOWN:
    Detectable (F1>0.3):  {n_detectable}/{len(hourly_data)} hours
    Marginal (0.1-0.3):   {n_marginal}/{len(hourly_data)} hours  
    Undetectable (<=0.1): {n_undetectable}/{len(hourly_data)} hours

  KEY FINDINGS:

  1. RESOLUTION LIMIT:
     GSD = {gsd*1000:.0f}m >> debris feature size (10-150m)
     Sensor pixel integrates over ~{gsd*1000**2:.0f} m2 area
     Debris signal diluted by pixel area averaging

  2. CONTRAST MECHANISM:
     Foam/bright debris: HIGH contrast vs dark ocean
     Fuel slick: LOW contrast (both dark)
     Metal: MODERATE (depends on sun angle/specularity)

  3. SOLAR GEOMETRY IS CRITICAL:
     Best detection near local noon (sun overhead)
     Low sun angle → longer path → more atmosphere → lower SNR

  4. ATMOSPHERIC PATH:
     {z:.0f} deg zenith → air mass ~{atmospheric_path_radiance(z)["air_mass"]:.1f}
     Path radiance reduces effective contrast

  5. VERDICT:
     {'POSSIBLE but marginal' if best_f1 > 0.1 else 'NOT DETECTABLE with single pixel analysis'}
     Sub-pixel anomaly detection OR multi-temporal stacking required
"""
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=8,
           fontfamily='monospace', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    out_path = os.path.join(OUT, 'mtsat2_detectability.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.close()
    
    # === SECOND FIGURE: Multi-angle visualization ===
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle('MTSAT-2 Simulated Images — Diurnal Cycle\n'
                  f'March 8, 2014 | {TARGET_LAT}S, {TARGET_LON}E | GSD={gsd:.1f}km',
                  fontsize=13, fontweight='bold')
    
    daytime_results = [(utc_h, r) for utc_h, r in sorted(results.items()) 
                       if not r['is_night']]
    
    for idx, (utc_h, r) in enumerate(daytime_results[:8]):
        ax = axes2.flat[idx]
        im = ax.imshow(r['image'], cmap='gray', vmin=0, vmax=DYNAMIC_RANGE)
        ax.contour(r['truth_mask'], levels=[0.5], colors=['red'], linewidths=1.5)
        lt = (utc_h + 95/15) % 24
        f1 = r['detection']['f1_score']
        ax.set_title(f'UTC {utc_h:02d}:00 (LT {lt:.1f}h)\n'
                     f'Sun={r["solar"]["elevation"]:.0f}, F1={f1:.2f}', fontsize=9)
        ax.set_xlabel('Pixels east-west')
        ax.set_ylabel('Pixels north-south')
    
    # Hide unused subplots
    for idx in range(len(daytime_results), 8):
        axes2.flat[idx].axis('off')
    
    plt.tight_layout()
    out_path2 = os.path.join(OUT, 'mtsat2_diurnal_images.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path2}')
    plt.close()
    
    return out_path, out_path2, results


if __name__ == '__main__':
    paths = main()
