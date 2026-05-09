"""Bathymetry data for Karlskrona Skärgård — serves seabed depth queries.

Synthetic but realistic depth grid based on known bathymetric features
of the Karlskrona archipelago (Swedish Maritime Administration, EMODnet 2024).

Resolution: ~0.001° (~60m grid at this latitude)
Region: 56.05–56.30°N, 15.40–15.80°E
"""

import time

import numpy as np

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 56.05, 56.30
LON_MIN, LON_MAX = 15.40, 15.80
RES = 0.001  # degrees (~60m at 56°N)
NLAT = int((LAT_MAX - LAT_MIN) / RES) + 1
NLON = int((LON_MAX - LON_MIN) / RES) + 1

_grid: np.ndarray | None = None
_generation_time: float = 0.0


def _build_grid() -> np.ndarray:
    """Generate a realistic Karlskrona archipelago bathymetry grid."""
    lats = np.linspace(LAT_MIN, LAT_MAX, NLAT)
    lons = np.linspace(LON_MIN, LON_MAX, NLON)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # --- Depth control points (lat, lon, depth_m) ---
    # Karlskrona skärgård — based on Swedish Maritime Administration chart data
    control_points = [
        # Baltic approach (south of archipelago — open sea)
        (56.06, 15.55, 70.0),
        (56.06, 15.60, 72.0),
        (56.06, 15.65, 68.0),
        (56.07, 15.58, 65.0),
        (56.07, 15.62, 66.0),
        (56.08, 15.60, 60.0),
        (56.08, 15.63, 58.0),
        (56.09, 15.57, 55.0),
        (56.09, 15.62, 54.0),
        # Uttorp / outer archipelago
        (56.09, 15.55, 50.0),
        (56.09, 15.58, 48.0),
        (56.09, 15.62, 52.0),
        (56.10, 15.57, 42.0),
        (56.10, 15.60, 45.0),
        (56.10, 15.64, 47.0),
        # Main approach channel (Aspöfjärden)
        (56.11, 15.57, 40.0),
        (56.11, 15.60, 38.0),
        (56.12, 15.58, 35.0),
        (56.13, 15.58, 30.0),
        (56.14, 15.59, 28.0),
        # Kungsholmsfjärden (deeper basin)
        (56.11, 15.50, 55.0),
        (56.13, 15.51, 50.0),
        (56.15, 15.52, 45.0),
        # Aspö island area
        (56.14, 15.55, 22.0),
        (56.15, 15.54, 18.0),
        (56.16, 15.55, 15.0),
        # Channel east of Aspö
        (56.14, 15.57, 25.0),
        (56.16, 15.58, 20.0),
        (56.18, 15.59, 18.0),
        # Karlskrona harbour approaches
        (56.15, 15.59, 22.0),
        (56.16, 15.60, 18.0),
        (56.17, 15.59, 15.0),
        (56.17, 15.61, 14.0),
        # Naval harbour (Örlogshamnen)
        (56.18, 15.58, 12.0),
        (56.18, 15.59, 11.0),
        (56.19, 15.58, 10.0),
        (56.19, 15.59, 9.0),
        # Tjurkö area
        (56.13, 15.60, 20.0),
        (56.14, 15.62, 18.0),
        (56.15, 15.63, 15.0),
        # Channel between Tjurkö and Sturkö
        (56.16, 15.62, 20.0),
        (56.17, 15.63, 18.0),
        (56.18, 15.64, 16.0),
        # Sturkö area
        (56.19, 15.63, 14.0),
        (56.20, 15.64, 12.0),
        (56.21, 15.65, 10.0),
        # Western archipelago (Hasslö area)
        (56.12, 15.46, 35.0),
        (56.14, 15.47, 25.0),
        (56.16, 15.48, 18.0),
        (56.18, 15.49, 12.0),
        # Northern waters
        (56.20, 15.55, 15.0),
        (56.22, 15.57, 12.0),
        (56.24, 15.58, 8.0),
        (56.26, 15.60, 5.0),
        # East side channels
        (56.16, 15.68, 30.0),
        (56.18, 15.70, 25.0),
        (56.19, 15.72, 20.0),
        (56.20, 15.74, 15.0),
        # Outer eastern area
        (56.08, 15.70, 60.0),
        (56.10, 15.68, 55.0),
        (56.12, 15.66, 40.0),
        # Shallow coastal / skerry areas
        (56.12, 15.53, 12.0),
        (56.13, 15.56, 10.0),
        (56.15, 15.57, 8.0),
        (56.17, 15.57, 6.0),
        (56.20, 15.56, 5.0),
        # Land: Aspö island (dense grid)
        (56.138, 15.548, 0.0), (56.140, 15.550, 0.0), (56.142, 15.552, 0.0),
        (56.144, 15.550, 0.0), (56.146, 15.552, 0.0), (56.148, 15.554, 0.0),
        (56.150, 15.552, 0.0), (56.152, 15.554, 0.0), (56.154, 15.556, 0.0),
        (56.156, 15.554, 0.0), (56.158, 15.556, 0.0), (56.145, 15.558, 0.0),
        (56.147, 15.556, 0.0), (56.149, 15.558, 0.0),
        # Land: Tjurkö island (dense grid)
        (56.138, 15.605, 0.0), (56.140, 15.608, 0.0), (56.142, 15.610, 0.0),
        (56.144, 15.612, 0.0), (56.146, 15.614, 0.0), (56.148, 15.616, 0.0),
        (56.150, 15.618, 0.0), (56.152, 15.616, 0.0), (56.154, 15.618, 0.0),
        (56.156, 15.612, 0.0), (56.143, 15.615, 0.0), (56.147, 15.610, 0.0),
        # Land: Sturkö island (dense grid)
        (56.180, 15.620, 0.0), (56.182, 15.622, 0.0), (56.184, 15.624, 0.0),
        (56.186, 15.626, 0.0), (56.188, 15.628, 0.0), (56.190, 15.630, 0.0),
        (56.192, 15.632, 0.0), (56.194, 15.634, 0.0), (56.196, 15.636, 0.0),
        (56.198, 15.638, 0.0), (56.200, 15.640, 0.0),
        (56.183, 15.630, 0.0), (56.187, 15.634, 0.0), (56.191, 15.625, 0.0),
        # Land: Hasslö island (dense grid)
        (56.150, 15.460, 0.0), (56.152, 15.462, 0.0), (56.154, 15.464, 0.0),
        (56.156, 15.466, 0.0), (56.158, 15.468, 0.0), (56.160, 15.470, 0.0),
        (56.162, 15.472, 0.0), (56.164, 15.474, 0.0), (56.166, 15.476, 0.0),
        (56.168, 15.474, 0.0), (56.170, 15.472, 0.0), (56.155, 15.478, 0.0),
        # Land: smaller skerries
        (56.140, 15.640, 0.0),
        (56.160, 15.645, 0.0),
        (56.175, 15.605, 0.0),
        (56.190, 15.600, 0.0),
        (56.185, 15.615, 0.0),
        (56.205, 15.605, 0.0),
        (56.215, 15.610, 0.0),
        (56.225, 15.615, 0.0),
        (56.180, 15.570, 0.0),
        # Land: mainland (north of archipelago)
        (56.230, 15.500, 0.0),
        (56.250, 15.550, 0.0),
        (56.260, 15.580, 0.0),
        (56.270, 15.620, 0.0),
        (56.250, 15.650, 0.0),
        (56.230, 15.680, 0.0),
        (56.240, 15.720, 0.0),
        # Land: mainland (west / northwest)
        (56.200, 15.430, 0.0),
        (56.220, 15.440, 0.0),
        (56.240, 15.450, 0.0),
        (56.250, 15.460, 0.0),
        (56.260, 15.480, 0.0),
        # Land: mainland (northeast)
        (56.270, 15.650, 0.0),
        (56.250, 15.700, 0.0),
        (56.230, 15.750, 0.0),
        (56.200, 15.770, 0.0),
    ]

    # --- Inverse-distance weighting ---
    depth = np.zeros_like(lat_grid, dtype=np.float64)
    weight_sum = np.zeros_like(lat_grid, dtype=np.float64)

    for cp_lat, cp_lon, cp_depth in control_points:
        dist = np.sqrt((lat_grid - cp_lat) ** 2 + (lon_grid - cp_lon) ** 2)
        # Gaussian weight with sigma=0.008° (~500m) for sharper features
        w = np.exp(-dist ** 2 / (2 * 0.000064))
        depth += w * cp_depth
        weight_sum += w

    mask = weight_sum > 1e-10
    depth[mask] /= weight_sum[mask]

    # --- Channel axis model for the main fairway ---
    # Main channel: south → Aspöfjärden → Karlskrona harbour
    axis_points = [
        (56.07, 15.59),  # Baltic approach
        (56.09, 15.58),  # Outer archipelago
        (56.11, 15.58),  # Aspöfjärden entrance
        (56.13, 15.58),  # Aspöfjärden
        (56.15, 15.59),  # Channel north
        (56.17, 15.59),  # Harbour approach
        (56.19, 15.59),  # Karlskrona harbour
    ]
    axis_depths = [55, 48, 42, 30, 20, 15, 10]

    # Secondary channel: west of Aspö (Kungsholmsleden)
    axis2_points = [
        (56.08, 15.52),  # South
        (56.11, 15.51),
        (56.14, 15.51),
        (56.16, 15.50),
        (56.18, 15.52),
    ]
    axis2_depths = [52, 48, 35, 22, 14]

    # Eastern channel: via Tjurkö–Sturkö
    axis3_points = [
        (56.10, 15.64),  # South entrance
        (56.13, 15.63),
        (56.16, 15.63),
        (56.18, 15.65),
        (56.20, 15.66),
    ]
    axis3_depths = [48, 32, 22, 16, 12]

    for ax_pts, ax_deps in [
        (axis_points, axis_depths),
        (axis2_points, axis2_depths),
        (axis3_points, axis3_depths),
    ]:
        channel_depth = np.zeros_like(lat_grid, dtype=np.float64)
        channel_weight = np.zeros_like(lat_grid, dtype=np.float64)
        for (ax_lat, ax_lon), ax_dep in zip(ax_pts, ax_deps):
            dist = np.sqrt((lat_grid - ax_lat) ** 2 + (lon_grid - ax_lon) ** 2)
            w = np.exp(-dist ** 2 / (2 * 0.000036))  # sigma ~0.006° (~350m)
            channel_depth += w * ax_dep
            channel_weight += w
        ch_mask = channel_weight > 0.01
        depth[ch_mask] = channel_depth[ch_mask] / channel_weight[ch_mask]

    # --- Coastal gradient (only for land-proximal areas, not the sea boundaries) ---
    # The southern edge is open Baltic Sea — don't shallow it
    coast_dist = np.minimum(
        np.abs(lat_grid - LAT_MAX),  # only northern edge (mainland)
        np.abs(lat_grid - LAT_MIN),  # only if too far south
    )
    # Only apply shelf where close to the northern boundary (mainland)
    north_shelf = 1.0 - np.exp(-np.abs(lat_grid - LAT_MAX) * 15.0)
    shelf_mask = lat_grid > 56.20  # only near mainland coast
    depth[shelf_mask] *= np.clip(north_shelf[shelf_mask], 0.0, 1.0)

    # Land mask
    depth[depth < 0.5] = 0.0

    return depth.astype(np.float32)


def get_depth(lat: float, lon: float) -> float | None:
    """Return seabed depth in meters at the given coordinates."""
    global _grid, _generation_time
    if _grid is None:
        _grid = _build_grid()
        _generation_time = time.time()

    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None

    ilat = int((lat - LAT_MIN) / RES)
    ilon = int((lon - LON_MIN) / RES)
    ilat = max(0, min(NLAT - 1, ilat))
    ilon = max(0, min(NLON - 1, ilon))

    val = float(_grid[ilat, ilon])
    return val if val > 0.5 else None


def grid_info() -> dict:
    """Return metadata about the loaded grid."""
    global _grid, _generation_time
    if _grid is None:
        _grid = _build_grid()
        _generation_time = time.time()
    valid = _grid[_grid > 0.5]
    return {
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "resolution_deg": RES,
        "grid_shape": [NLAT, NLON],
        "depth_min_m": float(valid.min()) if len(valid) > 0 else 0,
        "depth_max_m": float(valid.max()) if len(valid) > 0 else 0,
        "generated_at": _generation_time,
    }
