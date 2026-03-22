from __future__ import annotations
import numpy as np
import math

# Earth's radius in meters, used for converting geographic coordinates to Cartesian coordinates.
R_EARTH_M = 6_371_000


def build_local_xy_int(nodes_latlon_microdeg: np.ndarray) -> np.ndarray:
    """
    Converts geographic coordinates (latitude/longitude in microdegrees) to local Cartesian coordinates (x, y in meters).

    Args:
        nodes_latlon_microdeg: A NumPy array where each row is [longitude_microdeg, latitude_microdeg, elevation].

    Returns:
        A NumPy array of shape (n, 2) with integer x and y coordinates in meters.
    """
    # Extract latitude and longitude from the input array and convert microdegrees to degrees.
    lat_deg = nodes_latlon_microdeg[:, 1].astype(np.float64) * 1e-6
    lon_deg = nodes_latlon_microdeg[:, 0].astype(np.float64) * 1e-6

    # Compute the mean latitude and longitude to define the origin of the local coordinate system.
    lat0 = float(lat_deg.mean())
    lon0 = float(lon_deg.mean())

    # Convert all latitudes and longitudes from degrees to radians for trigonometric calculations.
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)

    # Compute local Cartesian coordinates (x, y) using spherical Earth approximation.
    x = R_EARTH_M * np.cos(lat0_r) * (lon_r - lon0_r)
    y = R_EARTH_M * (lat_r - lat0_r)

    # Create an empty array to store the results, round and cast to 32-bit integers.
    xy = np.empty((nodes_latlon_microdeg.shape[0], 2), dtype=np.int32)
    xy[:, 0] = np.rint(x).astype(np.int32)
    xy[:, 1] = np.rint(y).astype(np.int32)
    return xy
