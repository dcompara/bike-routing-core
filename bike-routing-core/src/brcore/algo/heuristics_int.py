from __future__ import annotations
import math
import numpy as np

def h_euclidean_int(xy: np.ndarray, u: int, t: int) -> int:
    """
    Integer Euclidean heuristic in meters.
    xy[v] = [x_m, y_m] int32
    """
    dx = int(xy[u, 0] - xy[t, 0])
    dy = int(xy[u, 1] - xy[t, 1])
    return int(math.isqrt(dx*dx + dy*dy))
