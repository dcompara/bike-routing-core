# params.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- Feature ordering convention (must match your path accumulator x_vec) ----
# Example: [length_m, elevation_m, popularity, narrowness_or_width_cost]
FEATURES = ("length", "elevation", "popularity", "narrowness")
N_FEATURES = len(FEATURES)

# ---- Hard constraints (lower/upper) ----
LOWER = np.array([30000.0,  400.0, 150, 5], dtype=np.float64)
UPPER = np.array([35000.0, 500.0, 255, 15], dtype=np.float64)

# ---- Weights for Chebyshev (dimensionless priorities) ----
W = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float64)

# ---- Search-space-reduction parameters ----
CORRIDOR_SLACK_M = 1500
MAX_HOPS_FROM_BOUNDARY = 1

# ---- Distance-biased exponential delay params (for later activation logic) ----
DIST_FEATURE = "length"
DELAY_ALPHA = 0.00025  # scale for distance bias (units depend on your length units)
DELAY_BETA = 1.5       # exponential rate, mean delay = 1/beta


@dataclass(frozen=True)
class ChebyshevNorm:
    """
    Normalized weighted Chebyshev:
        score(x) = max_j w_j * |(x_j - mid_j)/half_range_j|
    where mid=(L+U)/2 and half_range=(U-L)/2

    If half_range_j == 0, we protect against division by 0.
    """
    mid: np.ndarray
    inv_half: np.ndarray
    w: np.ndarray

    @staticmethod
    def from_bounds(lower: np.ndarray, upper: np.ndarray, w: np.ndarray) -> "ChebyshevNorm":
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)

        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)

        # Guard against degenerate ranges
        inv_half = np.empty_like(half)
        eps = 1e-12
        inv_half[half > eps] = 1.0 / half[half > eps]
        inv_half[half <= eps] = 0.0  # if no range, treat deviation as 0 here; you can also make it huge

        return ChebyshevNorm(mid=mid, inv_half=inv_half, w=w)

    def score(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        z = np.abs((x - self.mid) * self.inv_half)
        return float(np.max(self.w * z))


# Instantiate a reusable Chebyshev heuristic object
CHEB = ChebyshevNorm.from_bounds(LOWER, UPPER, W)



def distance_biased_exp_delay(x_vec: np.ndarray, rng: np.random.Generator | None = None) -> float:
    """
    Activation delay:
        T = alpha * x_dist + Exp(beta)
    where x_dist is x_vec[dist_idx].

    Keep here for later use when you implement staged activation.
    """
    if rng is None:
        rng = np.random.default_rng()

    dist_idx = FEATURES.index(DIST_FEATURE)
    dist = float(np.asarray(x_vec, dtype=np.float64)[dist_idx])
    jitter = float(rng.exponential(scale=1.0 / DELAY_BETA))
    return DELAY_ALPHA * dist + jitter
