from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class CompactDiGraph:
    """
    Compact directed graph in  Compressed Sparse Row (CSR)-like form.
    Nodes must be indexed 0..N-1.
    """

    offsets: np.ndarray  # (n_nodes+1,)
    to: np.ndarray  # (n_edges,)
    w: np.ndarray  # (n_edges, n_obj)
    road_id: np.ndarray  # (n_edges,) or zeros
    n_nodes: int
    n_edges: int
    n_obj: int

    def neighbors(self, u: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the neighbors, weights, and road IDs for all edges originating from node u.

        Args:
            u: The source node index.

        Returns:
            A tuple containing:
                - The array of destination nodes for edges from u.
                - The array of weights/objects for edges from u.
                - The array of road IDs for edges from u.
        """
        a = self.offsets[u]
        b = self.offsets[u + 1]
        return self.to[a:b], self.w[a:b], self.road_id[a:b]
