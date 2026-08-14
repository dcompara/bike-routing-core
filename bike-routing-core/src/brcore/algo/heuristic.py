from __future__ import annotations
import math
import numpy as np

from brcore.algo.coords import EARTH_RADIUS_M

# Coordinates are micro-degrees:
#   lat_microdeg = lat * 1e6
#   lon_microdeg = lon * 1e6
DEG_TO_RAD_PER_MICRODEG = math.pi / 180.0 / 1_000_000.0


def haversine_m_microdeg(lon1_u: int, lat1_u: int, lon2_u: int, lat2_u: int) -> float:
    """
    Haversine great-circle distance in meters.
    https://en.wikipedia.org/wiki/Great-circle_distance
    Inputs are int micro-degrees (lon, lat).
    Mirrors the C++ 'a = 0.5 - cos(dlat)/2 + cos(lat1)*cos(lat2)*(1-cos(dlon))/2' form.

    """
    lon1 = lon1_u * DEG_TO_RAD_PER_MICRODEG
    lat1 = lat1_u * DEG_TO_RAD_PER_MICRODEG
    lon2 = lon2_u * DEG_TO_RAD_PER_MICRODEG
    lat2 = lat2_u * DEG_TO_RAD_PER_MICRODEG

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = 0.5 - math.cos(dlat) * 0.5 + math.cos(lat1) * math.cos(lat2) * (1.0 - math.cos(dlon)) * 0.5
    # Great-circle distance = 2 R asin(sqrt(a))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def haversine_dm_microdeg(lon1_u: int, lat1_u: int, lon2_u: int, lat2_u: int) -> int:
    """
    Same distance but returned as integer decimeters (0.1 m).
    """
    return int(math.floor(10.0 * haversine_m_microdeg(lon1_u, lat1_u, lon2_u, lat2_u)))

def h_target_haversine_m(nodes: np.ndarray, u: int, t: int) -> float:
    """
    nodes[v] format: [lat_microdeg, lon_microdeg, elevation, ...]
    """
    lat_u = int(nodes[u, 0])
    lon_u = int(nodes[u, 1])
    lat_t = int(nodes[t, 0])
    lon_t = int(nodes[t, 1])
    return haversine_m_microdeg(lon_u, lat_u, lon_t, lat_t)

def h_target_haversine_dm(nodes: np.ndarray, u: int, t: int) -> int:
    """
    Same but returns decimeters.
    """
    lat_u = int(nodes[u, 0])
    lon_u = int(nodes[u, 1])
    lat_t = int(nodes[t, 0])
    lon_t = int(nodes[t, 1])
    return haversine_dm_microdeg(lon_u, lat_u, lon_t, lat_t)

def build_local_xy_m(nodes_latlon_microdeg: np.ndarray, ref_lat_microdeg: int | None = None,
                     ref_lon_microdeg: int | None = None) -> np.ndarray:
    """
    nodes_latlon_microdeg[v] format: [lat_microdeg, lon_microdeg, ...]
    Returns xy_m[v] = (x_m, y_m) using equirectangular projection around (ref_lat, ref_lon).
    """
    lat = nodes_latlon_microdeg[:, 0].astype(np.float64) * 1e-6
    lon = nodes_latlon_microdeg[:, 1].astype(np.float64) * 1e-6

    if ref_lat_microdeg is None:
        ref_lat = float(np.mean(lat))
    else:
        ref_lat = ref_lat_microdeg * 1e-6

    if ref_lon_microdeg is None:
        ref_lon = float(np.mean(lon))
    else:
        ref_lon = ref_lon_microdeg * 1e-6

    # degrees -> radians
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    ref_lat_r = np.deg2rad(ref_lat)
    ref_lon_r = np.deg2rad(ref_lon)

    x = EARTH_RADIUS_M * np.cos(ref_lat_r) * (lon_r - ref_lon_r)
    y = EARTH_RADIUS_M * (lat_r - ref_lat_r)

    xy = np.empty((nodes_latlon_microdeg.shape[0], 2), dtype=np.float64)
    xy[:, 0] = x
    xy[:, 1] = y
    return xy

def h_euclidean_xy_m(xy_m: np.ndarray, u: int, t: int) -> float:
    dx = xy_m[u, 0] - xy_m[t, 0]
    dy = xy_m[u, 1] - xy_m[t, 1]
    return float((dx*dx + dy*dy) ** 0.5)
