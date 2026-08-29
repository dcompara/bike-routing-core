from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from brcore.algo.coords import build_local_xy_int
from brcore.algo.search_space_reduction import search_space_reduction
from brcore.graph.compact import CompactDiGraph
from brcore.io.load_plot_xy import load_xy_graph
from brcore.io.loaders import load_boundary_nodes, load_partition, load_seeds


GRAPH_PATH = "data/graph_Paris_south_4_objectives.xy"
SEEDS_PATH = "data/seeds.txt"
PARTITION_PATH = "data/paris_voronoi_nodes.txt"
BOUNDARY_NODES_PATH = "data/paris_voronoi_boundary_nodes.txt"

DEFAULT_OUTPUT_JSON = "tmp_scalar_feasibility_experiment.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_feasibility_experiment.csv"

PARIS_BURES_SOURCE = 127
PARIS_BURES_TARGET = 4433
PARIS_BURES_LMIN = 30000.0
PARIS_BURES_LMAX = 35000.0
PARIS_BURES_HMIN = 400.0
PARIS_BURES_HMAX = 500.0
PARIS_BURES_PMIN = 150.0
PARIS_BURES_WMAX = 15.0
CORRIDOR_SLACK_M = 1500
MAX_HOPS_FROM_BOUNDARY = 1
EPS = 1e-9


EXPECTED_RETAINED_SHORTEST = {
    "length": 25592.0,
    "elevation": 327.2,
    "avg_pop": 172.45,
    "avg_width": 20.97,
}
EXPECTED_RETAINED_REFERENCE = {
    "length": 31818.0,
    "elevation": 405.4,
    "avg_pop": 199.87,
    "avg_width": 14.16,
}
EXPECTED_FULL_REFERENCE = {
    "length": 30157.0,
    "elevation": 358.9,
    "avg_pop": 212.90,
    "avg_width": 14.25,
}
REGRESSION_TOLERANCES = {
    "length": 5.0,
    "elevation": 1.0,
    "avg_pop": 0.5,
    "avg_width": 0.2,
}


@dataclass(frozen=True)
class QueryBox:
    name: str
    source: int
    target: int
    Lmin: float
    Lmax: float
    Hmin: float
    Hmax: float
    Pmin: float
    Wmax: float
    corridor_slack_m: int = CORRIDOR_SLACK_M
    max_hops_from_boundary: int = MAX_HOPS_FROM_BOUNDARY

    def is_feasible(self, metrics: "RouteMetrics") -> bool:
        return (
            self.Lmin <= metrics.length <= self.Lmax
            and self.Hmin <= metrics.elevation <= self.Hmax
            and metrics.popularity_length >= self.Pmin * metrics.length
            and metrics.width_length <= self.Wmax * metrics.length
        )

    def violations(self, metrics: "RouteMetrics") -> dict[str, float]:
        return {
            "length_low": max(0.0, self.Lmin - metrics.length),
            "length_high": max(0.0, metrics.length - self.Lmax),
            "elevation_low": max(0.0, self.Hmin - metrics.elevation),
            "elevation_high": max(0.0, metrics.elevation - self.Hmax),
            "popularity_low": max(0.0, self.Pmin - metrics.avg_popularity),
            "width_high": max(0.0, metrics.avg_width - self.Wmax),
        }

    def violation_scales(self) -> dict[str, float]:
        return {
            "length": max(abs(self.Lmax - self.Lmin), 1.0),
            "elevation": max(abs(self.Hmax - self.Hmin), 1.0),
            "popularity": max(abs(self.Pmin), 1.0),
            "width": max(abs(self.Wmax), 1.0),
        }

    def normalized_violation_score(self, metrics: "RouteMetrics") -> float:
        violations = self.violations(metrics)
        scales = self.violation_scales()
        return max(
            violations["length_low"] / scales["length"],
            violations["length_high"] / scales["length"],
            violations["elevation_low"] / scales["elevation"],
            violations["elevation_high"] / scales["elevation"],
            violations["popularity_low"] / scales["popularity"],
            violations["width_high"] / scales["width"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "Lmin": self.Lmin,
            "Lmax": self.Lmax,
            "Hmin": self.Hmin,
            "Hmax": self.Hmax,
            "Pmin": self.Pmin,
            "Wmax": self.Wmax,
            "corridor_slack_m": self.corridor_slack_m,
            "max_hops_from_boundary": self.max_hops_from_boundary,
        }


@dataclass(frozen=True)
class RouteMetrics:
    length: float
    elevation: float
    popularity_length: float
    width_length: float
    road_changes: int

    @property
    def avg_popularity(self) -> float:
        return self.popularity_length / self.length if self.length > 0.0 else 0.0

    @property
    def avg_width(self) -> float:
        return self.width_length / self.length if self.length > 0.0 else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "length": self.length,
            "elevation": self.elevation,
            "popularity_length": self.popularity_length,
            "width_length": self.width_length,
            "avg_pop": self.avg_popularity,
            "avg_width": self.avg_width,
            "road_changes": self.road_changes,
        }


@dataclass(frozen=True)
class ScalarizationSpec:
    name: str
    family: str
    parameters: dict[str, float | str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class MetricConstants:
    rho_H: float
    p_star: float
    max_slope: float
    zero_length_positive_gain_edges: int
    usable_edges: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rho_H": self.rho_H,
            "p_star": self.p_star,
            "max_slope": self.max_slope,
            "zero_length_positive_gain_edges": self.zero_length_positive_gain_edges,
            "usable_edges": self.usable_edges,
        }


@dataclass(frozen=True)
class GraphContext:
    mode: str
    edge_mask: np.ndarray
    node_count: int
    edge_count: int
    constants: MetricConstants
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "metric_constants": self.constants.as_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DijkstraStats:
    heap_pops: int
    expanded_nodes: int
    edge_scans: int
    raw_edge_rows_checked: int
    elapsed_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "heap_pops": self.heap_pops,
            "expanded_nodes": self.expanded_nodes,
            "edge_scans": self.edge_scans,
            "raw_edge_rows_checked": self.raw_edge_rows_checked,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class ScalarPathResult:
    route_found: bool
    scalar_cost: float
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    metrics: RouteMetrics | None
    stats: DijkstraStats


@dataclass(frozen=True)
class RouteValidation:
    directed_edges_ok: bool
    missing_edges: int
    metric_deltas: tuple[float, float, float, float]
    road_changes_delta: int

    @property
    def passed(self) -> bool:
        return (
            self.directed_edges_ok
            and self.missing_edges == 0
            and max(abs(delta) for delta in self.metric_deltas) <= 1e-6
            and self.road_changes_delta == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "directed_edges_ok": self.directed_edges_ok,
            "missing_edges": self.missing_edges,
            "metric_deltas": list(self.metric_deltas),
            "road_changes_delta": self.road_changes_delta,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ScalarSearchRecord:
    index: int
    spec: ScalarizationSpec
    graph_mode: str
    result: ScalarPathResult
    validation: RouteValidation
    feasible: bool
    normalized_violation_score: float | None
    violations: dict[str, float] | None
    elementary: bool
    repeated_vertex_count: int

    def as_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        metrics = self.result.metrics
        out: dict[str, Any] = {
            "index": self.index,
            "scalar_name": self.spec.name,
            "scalar_family": self.spec.family,
            "scalar_parameters": self.spec.parameters,
            "graph_mode": self.graph_mode,
            "elapsed_s": self.result.stats.elapsed_s,
            "heap_pops": self.result.stats.heap_pops,
            "expanded_nodes": self.result.stats.expanded_nodes,
            "edge_scans": self.result.stats.edge_scans,
            "raw_edge_rows_checked": self.result.stats.raw_edge_rows_checked,
            "route_found": self.result.route_found,
            "scalar_cost": _finite_or_none(self.result.scalar_cost),
            "L": None if metrics is None else metrics.length,
            "H": None if metrics is None else metrics.elevation,
            "avg_pop": None if metrics is None else metrics.avg_popularity,
            "avg_width": None if metrics is None else metrics.avg_width,
            "road_changes": None if metrics is None else metrics.road_changes,
            "path_nodes": len(self.result.path_nodes),
            "elementary": self.elementary,
            "repeated_vertex_count": self.repeated_vertex_count,
            "feasible": self.feasible,
            "normalized_violation_score": self.normalized_violation_score,
            "violations": self.violations,
            "validation": self.validation.as_dict(),
        }
        if include_paths:
            out["path_node_ids"] = list(self.result.path_nodes)
            out["csr_edge_ids"] = list(self.result.edge_ids)
        return out


@dataclass
class StaticInputs:
    G: CompactDiGraph
    nodes: np.ndarray
    xy_int: np.ndarray
    seeds: list[int]
    partition: dict[int, int]
    boundary_nodes: set[int]
    edge_sources: np.ndarray


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _safe_scale(value: float) -> float:
    return max(abs(float(value)), 1.0)


def _valid_road_id(road_id: int | None) -> int | None:
    if road_id is None or int(road_id) < 0:
        return None
    return int(road_id)


def _road_change_delta(left: int | None, right: int | None) -> int:
    left = _valid_road_id(left)
    right = _valid_road_id(right)
    if left is None or right is None:
        return 0
    return int(left != right)


def _edge_sources(G: CompactDiGraph) -> np.ndarray:
    sources = np.empty(G.n_edges, dtype=np.int32)
    for u in range(G.n_nodes):
        sources[int(G.offsets[u]) : int(G.offsets[u + 1])] = u
    return sources


def _build_reverse_edge_adjacency(
    G: CompactDiGraph,
    edge_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    reverse: list[list[tuple[int, int]]] = [[] for _ in range(G.n_nodes)]
    for u in range(G.n_nodes):
        start = int(G.offsets[u])
        end = int(G.offsets[u + 1])
        for edge_id in range(start, end):
            if not bool(edge_mask[edge_id]):
                continue
            reverse[int(G.to[edge_id])].append((u, edge_id))
    return reverse


def _physical_length_distances(
    G: CompactDiGraph,
    start: int,
    edge_mask: np.ndarray,
    *,
    reverse: bool = False,
) -> np.ndarray:
    if not (0 <= start < G.n_nodes):
        raise ValueError(f"start node {start} is outside 0..{G.n_nodes - 1}")

    reverse_adj = _build_reverse_edge_adjacency(G, edge_mask) if reverse else None
    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    dist[start] = 0.0
    frontier: list[tuple[float, int]] = [(0.0, start)]

    while frontier:
        current, node = heapq.heappop(frontier)
        if current != float(dist[node]):
            continue
        if reverse:
            assert reverse_adj is not None
            for pred, edge_id in reverse_adj[node]:
                edge_length = float(G.w[edge_id][0])
                candidate = current + edge_length
                if candidate < float(dist[pred]):
                    dist[pred] = candidate
                    heapq.heappush(frontier, (candidate, pred))
            continue

        start_idx = int(G.offsets[node])
        end_idx = int(G.offsets[node + 1])
        for edge_id in range(start_idx, end_idx):
            if not bool(edge_mask[edge_id]):
                continue
            nxt = int(G.to[edge_id])
            edge_length = float(G.w[edge_id][0])
            candidate = current + edge_length
            if candidate < float(dist[nxt]):
                dist[nxt] = candidate
                heapq.heappush(frontier, (candidate, nxt))

    return dist


def _node_set_for_mask(
    G: CompactDiGraph,
    edge_sources: np.ndarray,
    edge_mask: np.ndarray,
    *,
    extra_nodes: Iterable[int] = (),
) -> set[int]:
    nodes = {int(node) for node in extra_nodes if 0 <= int(node) < G.n_nodes}
    selected = np.flatnonzero(edge_mask)
    if len(selected) > 0:
        nodes.update(int(u) for u in edge_sources[selected])
        nodes.update(int(v) for v in G.to[selected])
    return nodes


def _metric_constants(G: CompactDiGraph, edge_mask: np.ndarray) -> MetricConstants:
    usable = np.flatnonzero(edge_mask)
    if len(usable) == 0:
        return MetricConstants(0.0, 0.0, 0.0, 0, 0)

    lengths = G.w[usable, 0].astype(np.float64)
    gains = G.w[usable, 1].astype(np.float64)
    pops = G.w[usable, 2].astype(np.float64)
    positive_lengths = lengths > EPS
    zero_positive_gain = int(np.count_nonzero((lengths <= EPS) & (gains > EPS)))
    if np.any(positive_lengths):
        slopes = gains[positive_lengths] / lengths[positive_lengths]
        max_slope = float(np.max(slopes))
    else:
        max_slope = 0.0
    rho_H = float(np.nextafter(max(max_slope, 0.0), float("inf")))
    p_star = float(np.max(pops)) if len(pops) > 0 else 0.0
    return MetricConstants(
        rho_H=rho_H,
        p_star=p_star,
        max_slope=max_slope,
        zero_length_positive_gain_edges=zero_positive_gain,
        usable_edges=int(len(usable)),
    )


def _geometric_edge_mask(G: CompactDiGraph, kept_nodes: set[int]) -> np.ndarray:
    mask = np.zeros(G.n_edges, dtype=bool)
    for u in kept_nodes:
        if not (0 <= int(u) < G.n_nodes):
            continue
        start = int(G.offsets[int(u)])
        end = int(G.offsets[int(u) + 1])
        for edge_id in range(start, end):
            if int(G.to[edge_id]) in kept_nodes:
                mask[edge_id] = True
    return mask


def _certified_length_corridor_mask(
    G: CompactDiGraph,
    edge_sources: np.ndarray,
    query: QueryBox,
    full_edge_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_s = _physical_length_distances(
        G,
        query.source,
        full_edge_mask,
        reverse=False,
    )
    d_t = _physical_length_distances(
        G,
        query.target,
        full_edge_mask,
        reverse=True,
    )
    mask = np.zeros(G.n_edges, dtype=bool)
    for edge_id in range(G.n_edges):
        u = int(edge_sources[edge_id])
        v = int(G.to[edge_id])
        candidate = float(d_s[u]) + float(G.w[edge_id][0]) + float(d_t[v])
        if math.isfinite(candidate) and candidate <= query.Lmax + 1e-6:
            mask[edge_id] = True
    return mask, d_s, d_t


def _build_graph_contexts(
    inputs: StaticInputs,
    query: QueryBox,
    requested_mode: str,
) -> tuple[list[GraphContext], dict[str, Any]]:
    G = inputs.G
    full_mask = np.ones(G.n_edges, dtype=bool)
    certified_mask, d_s, d_t = _certified_length_corridor_mask(
        G,
        inputs.edge_sources,
        query,
        full_mask,
    )
    kept_cells, kept_nodes = search_space_reduction(
        G=G,
        xy_int=inputs.xy_int,
        seeds=inputs.seeds,
        partition=inputs.partition,
        boundary_nodes=inputs.boundary_nodes,
        s=query.source,
        t=query.target,
        corridor_slack_m=query.corridor_slack_m,
        max_hops_from_boundary=query.max_hops_from_boundary,
    )
    geometric_mask = _geometric_edge_mask(G, kept_nodes)

    certified_nodes = _node_set_for_mask(
        G,
        inputs.edge_sources,
        certified_mask,
        extra_nodes=(query.source, query.target),
    )
    geometric_incident_nodes = _node_set_for_mask(
        G,
        inputs.edge_sources,
        geometric_mask,
        extra_nodes=(query.source, query.target),
    )
    comparison = {
        "full": {
            "nodes": G.n_nodes,
            "edges": G.n_edges,
        },
        "certified_length_corridor": {
            "nodes": len(certified_nodes),
            "edges": int(np.count_nonzero(certified_mask)),
            "name": "certified_length_corridor",
            "safe_edge_condition": "d_s(u)+l(u,v)+d_t(v) <= Lmax",
            "shortest_length_s_to_t": _finite_or_none(float(d_s[query.target])),
            "shortest_length_t_reverse_to_s": _finite_or_none(float(d_t[query.source])),
        },
        "geometric": {
            "kept_cells": len(kept_cells),
            "kept_nodes": len(kept_nodes),
            "incident_nodes": len(geometric_incident_nodes),
            "edges": int(np.count_nonzero(geometric_mask)),
            "corridor_slack_m": query.corridor_slack_m,
            "max_hops_from_boundary": query.max_hops_from_boundary,
        },
        "set_differences": {
            "certified_edges_not_geometric": int(
                np.count_nonzero(certified_mask & ~geometric_mask)
            ),
            "geometric_edges_not_certified": int(
                np.count_nonzero(geometric_mask & ~certified_mask)
            ),
            "certified_nodes_not_geometric": len(
                certified_nodes - geometric_incident_nodes
            ),
            "geometric_nodes_not_certified": len(
                geometric_incident_nodes - certified_nodes
            ),
        },
    }

    context_by_mode = {
        "full": GraphContext(
            mode="full",
            edge_mask=full_mask,
            node_count=G.n_nodes,
            edge_count=G.n_edges,
            constants=_metric_constants(G, full_mask),
            metadata={},
        ),
        "certified": GraphContext(
            mode="certified",
            edge_mask=certified_mask,
            node_count=len(certified_nodes),
            edge_count=int(np.count_nonzero(certified_mask)),
            constants=_metric_constants(G, certified_mask),
            metadata={
                "name": "certified_length_corridor",
                "safe_edge_condition": "d_s(u)+l(u,v)+d_t(v) <= Lmax",
                "shortest_length_s_to_t": _finite_or_none(float(d_s[query.target])),
                "shortest_length_t_reverse_to_s": _finite_or_none(
                    float(d_t[query.source])
                ),
            },
        ),
        "geometric": GraphContext(
            mode="geometric",
            edge_mask=geometric_mask,
            node_count=len(kept_nodes),
            edge_count=int(np.count_nonzero(geometric_mask)),
            constants=_metric_constants(G, geometric_mask),
            metadata={
                "kept_cells": len(kept_cells),
                "kept_nodes": len(kept_nodes),
                "incident_nodes": len(geometric_incident_nodes),
                "corridor_slack_m": query.corridor_slack_m,
                "max_hops_from_boundary": query.max_hops_from_boundary,
            },
        ),
    }
    modes = ["full", "certified", "geometric"] if requested_mode == "all" else [requested_mode]
    return [context_by_mode[mode] for mode in modes], comparison


def _reference_spec(query: QueryBox) -> ScalarizationSpec:
    return ScalarizationSpec(
        name="pop_width_reference",
        family="hinge",
        parameters={
            "alpha_L": 1.0,
            "alpha_w": 3.0,
            "alpha_p": 2.0,
            "Wref": query.Wmax,
            "Pref": query.Pmin,
        },
    )


def _fixed_portfolio(query: QueryBox) -> list[ScalarizationSpec]:
    specs = [
        ScalarizationSpec("shortest_length", "physical_length", {}),
        _reference_spec(query),
        ScalarizationSpec(
            "hinge_width_strong",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 8.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_width_very_strong",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 14.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_pop_strong",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 3.0,
                "alpha_p": 8.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_pop_very_strong",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 3.0,
                "alpha_p": 14.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_balanced_strong",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 8.0,
                "alpha_p": 8.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_low_length_pressure",
            "hinge",
            {
                "alpha_L": 0.35,
                "alpha_w": 5.0,
                "alpha_p": 5.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_medium_length_pressure",
            "hinge",
            {
                "alpha_L": 1.75,
                "alpha_w": 5.0,
                "alpha_p": 5.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "hinge_high_length_pressure",
            "hinge",
            {
                "alpha_L": 4.0,
                "alpha_w": 5.0,
                "alpha_p": 5.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "width_linear_mild",
            "width_linear",
            {"alpha_L": 1.0, "alpha_w": 0.35, "alpha_p": 0.0, "Pref": query.Pmin},
        ),
        ScalarizationSpec(
            "width_linear_strong",
            "width_linear",
            {"alpha_L": 1.0, "alpha_w": 1.25, "alpha_p": 1.5, "Pref": query.Pmin},
        ),
        ScalarizationSpec(
            "pop_complement_mild",
            "pop_complement",
            {"alpha_L": 1.0, "alpha_pop": 0.5, "alpha_w": 0.0, "Wref": query.Wmax},
        ),
        ScalarizationSpec(
            "pop_complement_strong",
            "pop_complement",
            {"alpha_L": 1.0, "alpha_pop": 2.5, "alpha_w": 2.0, "Wref": query.Wmax},
        ),
        ScalarizationSpec(
            "low_elevation_mild",
            "low_elevation",
            {"alpha_L": 1.0, "alpha_h": 1.5, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "low_elevation_strong",
            "low_elevation",
            {"alpha_L": 1.0, "alpha_h": 6.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "low_elevation_with_width_pop",
            "low_elevation",
            {
                "alpha_L": 1.0,
                "alpha_h": 2.0,
                "alpha_w": 3.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "high_elevation_weak_length",
            "high_elevation",
            {"alpha_L": 0.05, "alpha_H": 4.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "high_elevation_medium_length",
            "high_elevation",
            {"alpha_L": 0.35, "alpha_H": 4.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "high_elevation_strong_length",
            "high_elevation",
            {"alpha_L": 1.25, "alpha_H": 4.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "high_elevation_width",
            "high_elevation",
            {
                "alpha_L": 0.25,
                "alpha_H": 5.0,
                "alpha_w": 4.0,
                "alpha_p": 0.0,
                "Wref": query.Wmax,
            },
        ),
        ScalarizationSpec(
            "high_elevation_pop",
            "high_elevation",
            {
                "alpha_L": 0.25,
                "alpha_H": 5.0,
                "alpha_w": 0.0,
                "alpha_p": 4.0,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "high_elevation_width_pop",
            "high_elevation",
            {
                "alpha_L": 0.25,
                "alpha_H": 5.0,
                "alpha_w": 4.0,
                "alpha_p": 4.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "slope_inverse_beta_5",
            "nonlinear_high_elevation",
            {"form": "inverse", "beta": 5.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "slope_inverse_beta_12_width_pop",
            "nonlinear_high_elevation",
            {
                "form": "inverse",
                "beta": 12.0,
                "alpha_w": 3.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "slope_exp_beta_150_width",
            "nonlinear_high_elevation",
            {
                "form": "exp",
                "beta": 150.0,
                "alpha_w": 1.0,
                "alpha_p": 0.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "slope_exp_beta_250_width_pop_light",
            "nonlinear_high_elevation",
            {
                "form": "exp",
                "beta": 250.0,
                "alpha_w": 1.0,
                "alpha_p": 0.05,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "slope_exp_beta_8",
            "nonlinear_high_elevation",
            {"form": "exp", "beta": 8.0, "alpha_w": 0.0, "alpha_p": 0.0},
        ),
        ScalarizationSpec(
            "slope_exp_beta_400_width_light",
            "nonlinear_high_elevation",
            {
                "form": "exp",
                "beta": 400.0,
                "alpha_w": 0.25,
                "alpha_p": 0.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
        ScalarizationSpec(
            "slope_exp_beta_16_width_pop",
            "nonlinear_high_elevation",
            {
                "form": "exp",
                "beta": 16.0,
                "alpha_w": 3.0,
                "alpha_p": 3.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        ),
    ]
    return specs


def _portfolio(query: QueryBox, mode: str) -> list[ScalarizationSpec]:
    if mode == "reference":
        return [_reference_spec(query)]
    return _fixed_portfolio(query)


def _float_param(
    spec: ScalarizationSpec,
    key: str,
    default: float,
) -> float:
    value = spec.parameters.get(key, default)
    if isinstance(value, str):
        return default
    return float(value)


def _hinge_penalty(
    G: CompactDiGraph,
    edge_id: int,
    query: QueryBox,
    spec: ScalarizationSpec,
) -> float:
    row = G.w[edge_id]
    length = float(row[0])
    popularity = float(row[2])
    width = float(row[3])
    alpha_w = _float_param(spec, "alpha_w", 0.0)
    alpha_p = _float_param(spec, "alpha_p", 0.0)
    Wref = _float_param(spec, "Wref", query.Wmax)
    Pref = _float_param(spec, "Pref", query.Pmin)
    Wscale = _float_param(spec, "Wscale", _safe_scale(Wref))
    Pscale = _float_param(spec, "Pscale", _safe_scale(Pref))
    return length * (
        alpha_w * max(0.0, width - Wref) / max(Wscale, EPS)
        + alpha_p * max(0.0, Pref - popularity) / max(Pscale, EPS)
    )


def _edge_cost(
    G: CompactDiGraph,
    edge_id: int,
    query: QueryBox,
    constants: MetricConstants,
    spec: ScalarizationSpec,
) -> float:
    row = G.w[edge_id]
    length = float(row[0])
    gain = float(row[1])
    popularity = float(row[2])
    width = float(row[3])

    if spec.family == "physical_length":
        return length

    if spec.family == "hinge":
        alpha_L = _float_param(spec, "alpha_L", 1.0)
        return alpha_L * length + _hinge_penalty(G, edge_id, query, spec)

    if spec.family == "low_elevation":
        alpha_L = _float_param(spec, "alpha_L", 1.0)
        alpha_h = _float_param(spec, "alpha_h", 1.0)
        return alpha_L * length + alpha_h * gain + _hinge_penalty(G, edge_id, query, spec)

    if spec.family == "high_elevation":
        alpha_L = _float_param(spec, "alpha_L", 1.0)
        alpha_H = _float_param(spec, "alpha_H", 1.0)
        high_complement = max(0.0, constants.rho_H * length - gain)
        return (
            alpha_L * length
            + alpha_H * high_complement
            + _hinge_penalty(G, edge_id, query, spec)
        )

    if spec.family == "pop_complement":
        alpha_L = _float_param(spec, "alpha_L", 1.0)
        alpha_pop = _float_param(spec, "alpha_pop", 1.0)
        alpha_w = _float_param(spec, "alpha_w", 0.0)
        Wref = _float_param(spec, "Wref", query.Wmax)
        Wscale = _float_param(spec, "Wscale", _safe_scale(Wref))
        pop_scale = max(constants.p_star, 1.0)
        return (
            alpha_L * length
            + alpha_pop * length * max(0.0, constants.p_star - popularity) / pop_scale
            + alpha_w * length * max(0.0, width - Wref) / max(Wscale, EPS)
        )

    if spec.family == "width_linear":
        alpha_L = _float_param(spec, "alpha_L", 1.0)
        alpha_w = _float_param(spec, "alpha_w", 1.0)
        width_scale = _float_param(spec, "Wscale", _safe_scale(query.Wmax))
        return alpha_L * length + alpha_w * length * max(width, 0.0) / width_scale + _hinge_penalty(
            G,
            edge_id,
            query,
            spec,
        )

    if spec.family == "nonlinear_high_elevation":
        beta = _float_param(spec, "beta", 1.0)
        slope = gain / length if length > EPS else 0.0
        form = str(spec.parameters.get("form", "inverse"))
        if form == "exp":
            base = length * math.exp(-beta * max(0.0, slope))
        else:
            base = length / (1.0 + beta * max(0.0, slope))
        return base + _hinge_penalty(G, edge_id, query, spec)

    raise ValueError(f"unknown scalarization family {spec.family!r}")


def _metrics_from_edge_ids(
    G: CompactDiGraph,
    edge_ids: Iterable[int],
) -> RouteMetrics:
    length = 0.0
    elevation = 0.0
    popularity_length = 0.0
    width_length = 0.0
    road_changes = 0
    previous_road_id: int | None = None

    for edge_id in edge_ids:
        row = G.w[int(edge_id)]
        edge_length = float(row[0])
        length += edge_length
        elevation += float(row[1])
        popularity_length += edge_length * float(row[2])
        width_length += edge_length * float(row[3])
        road_id = int(G.road_id[int(edge_id)])
        road_changes += _road_change_delta(previous_road_id, road_id)
        previous_road_id = road_id

    return RouteMetrics(
        length=length,
        elevation=elevation,
        popularity_length=popularity_length,
        width_length=width_length,
        road_changes=road_changes,
    )


def _dijkstra_scalar_path(
    G: CompactDiGraph,
    query: QueryBox,
    context: GraphContext,
    spec: ScalarizationSpec,
) -> ScalarPathResult:
    if not (0 <= query.source < G.n_nodes and 0 <= query.target < G.n_nodes):
        raise ValueError("source or target is outside graph node id range")

    start_time = time.perf_counter()
    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    parent_node = np.full(G.n_nodes, -1, dtype=np.int32)
    parent_edge = np.full(G.n_nodes, -1, dtype=np.int32)
    best_tie: list[tuple[int, int, int, int]] = [
        (2**31 - 1, 2**31 - 1, 2**31 - 1, 2**31 - 1)
        for _ in range(G.n_nodes)
    ]
    source_tie = (0, -1, -1, -1)
    dist[query.source] = 0.0
    best_tie[query.source] = source_tie

    heap: list[tuple[float, int, int, int, int, int, int]] = [
        (0.0, *source_tie, 0, query.source)
    ]
    serial = 1
    heap_pops = 0
    expanded_nodes = 0
    edge_scans = 0
    raw_edge_rows_checked = 0

    while heap:
        item = heapq.heappop(heap)
        current_dist = float(item[0])
        current_tie = (int(item[1]), int(item[2]), int(item[3]), int(item[4]))
        node = int(item[6])
        heap_pops += 1

        if abs(current_dist - float(dist[node])) > EPS or current_tie != best_tie[node]:
            continue

        expanded_nodes += 1
        if node == query.target:
            break

        start_idx = int(G.offsets[node])
        end_idx = int(G.offsets[node + 1])
        current_last_road = current_tie[1]
        current_road_changes = current_tie[0]
        for edge_id in range(start_idx, end_idx):
            raw_edge_rows_checked += 1
            if not bool(context.edge_mask[edge_id]):
                continue
            edge_scans += 1
            step_cost = _edge_cost(G, edge_id, query, context.constants, spec)
            if not math.isfinite(step_cost):
                raise ValueError(
                    f"{spec.name} produced a non-finite cost on edge {edge_id}"
                )
            if step_cost < -EPS:
                raise ValueError(
                    f"{spec.name} produced a negative cost {step_cost} on edge {edge_id}"
                )
            if step_cost < 0.0:
                step_cost = 0.0

            nxt = int(G.to[edge_id])
            edge_road_id = int(G.road_id[edge_id])
            next_road_changes = current_road_changes + _road_change_delta(
                current_last_road,
                edge_road_id,
            )
            next_tie = (next_road_changes, edge_road_id, node, edge_id)
            candidate = current_dist + step_cost
            old = float(dist[nxt])
            if candidate < old - EPS or (
                abs(candidate - old) <= EPS and next_tie < best_tie[nxt]
            ):
                dist[nxt] = candidate
                best_tie[nxt] = next_tie
                parent_node[nxt] = node
                parent_edge[nxt] = edge_id
                heapq.heappush(heap, (candidate, *next_tie, serial, nxt))
                serial += 1

    elapsed_s = time.perf_counter() - start_time
    stats = DijkstraStats(
        heap_pops=heap_pops,
        expanded_nodes=expanded_nodes,
        edge_scans=edge_scans,
        raw_edge_rows_checked=raw_edge_rows_checked,
        elapsed_s=elapsed_s,
    )

    if not math.isfinite(float(dist[query.target])):
        return ScalarPathResult(False, float("inf"), (), (), None, stats)

    if query.source == query.target:
        metrics = RouteMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return ScalarPathResult(True, 0.0, (query.source,), (), metrics, stats)

    nodes: list[int] = []
    edge_ids: list[int] = []
    cur = query.target
    while cur != query.source:
        edge_id = int(parent_edge[cur])
        prev = int(parent_node[cur])
        if edge_id < 0 or prev < 0:
            raise RuntimeError("Dijkstra parent chain is incomplete")
        nodes.append(cur)
        edge_ids.append(edge_id)
        cur = prev
    nodes.append(query.source)
    nodes.reverse()
    edge_ids.reverse()
    metrics = _metrics_from_edge_ids(G, edge_ids)
    return ScalarPathResult(
        route_found=True,
        scalar_cost=float(dist[query.target]),
        path_nodes=tuple(nodes),
        edge_ids=tuple(edge_ids),
        metrics=metrics,
        stats=stats,
    )


def _validate_path(
    G: CompactDiGraph,
    result: ScalarPathResult,
) -> RouteValidation:
    if not result.route_found or result.metrics is None:
        return RouteValidation(False, 0, (0.0, 0.0, 0.0, 0.0), 0)

    missing = 0
    valid_edge_ids: list[int] = []
    if len(result.path_nodes) != len(result.edge_ids) + 1:
        missing += abs(len(result.path_nodes) - len(result.edge_ids) - 1)

    for u, v, edge_id in zip(result.path_nodes, result.path_nodes[1:], result.edge_ids):
        edge_id_int = int(edge_id)
        start = int(G.offsets[int(u)])
        end = int(G.offsets[int(u) + 1])
        if not (start <= edge_id_int < end) or int(G.to[edge_id_int]) != int(v):
            missing += 1
            continue
        valid_edge_ids.append(edge_id_int)

    recomputed = _metrics_from_edge_ids(G, valid_edge_ids)
    deltas = (
        recomputed.length - result.metrics.length,
        recomputed.elevation - result.metrics.elevation,
        recomputed.avg_popularity - result.metrics.avg_popularity,
        recomputed.avg_width - result.metrics.avg_width,
    )
    return RouteValidation(
        directed_edges_ok=missing == 0,
        missing_edges=missing,
        metric_deltas=deltas,
        road_changes_delta=recomputed.road_changes - result.metrics.road_changes,
    )


def _elementary_status(path_nodes: Sequence[int]) -> tuple[bool, int]:
    repeated = len(path_nodes) - len(set(path_nodes))
    return repeated == 0, repeated


def _make_search_record(
    *,
    index: int,
    spec: ScalarizationSpec,
    graph_mode: str,
    query: QueryBox,
    result: ScalarPathResult,
    validation: RouteValidation,
) -> ScalarSearchRecord:
    elementary, repeated_count = _elementary_status(result.path_nodes)
    if result.metrics is None:
        feasible = False
        score = None
        violations = None
    else:
        feasible = validation.passed and query.is_feasible(result.metrics)
        score = query.normalized_violation_score(result.metrics)
        violations = query.violations(result.metrics)
    return ScalarSearchRecord(
        index=index,
        spec=spec,
        graph_mode=graph_mode,
        result=result,
        validation=validation,
        feasible=feasible,
        normalized_violation_score=score,
        violations=violations,
        elementary=elementary,
        repeated_vertex_count=repeated_count,
    )


def _best_near_feasible(records: Sequence[ScalarSearchRecord]) -> ScalarSearchRecord | None:
    found = [
        record
        for record in records
        if record.result.metrics is not None and record.normalized_violation_score is not None
    ]
    if not found:
        return None
    return min(
        found,
        key=lambda record: (
            float(record.normalized_violation_score),
            int(record.repeated_vertex_count),
            float(record.result.metrics.length if record.result.metrics else float("inf")),
            float(record.result.metrics.elevation if record.result.metrics else float("inf")),
            record.spec.name,
        ),
    )


def _dominant_violation(query: QueryBox, metrics: RouteMetrics) -> str:
    violations = query.violations(metrics)
    scales = query.violation_scales()
    normalized = {
        "length_low": violations["length_low"] / scales["length"],
        "length_high": violations["length_high"] / scales["length"],
        "elevation_low": violations["elevation_low"] / scales["elevation"],
        "elevation_high": violations["elevation_high"] / scales["elevation"],
        "popularity_low": violations["popularity_low"] / scales["popularity"],
        "width_high": violations["width_high"] / scales["width"],
    }
    return max(normalized, key=normalized.get)


def _adaptive_spec(
    query: QueryBox,
    candidate: ScalarSearchRecord,
    round_index: int,
) -> ScalarizationSpec | None:
    if candidate.result.metrics is None:
        return None
    dominant = _dominant_violation(query, candidate.result.metrics)
    strength = float(2 ** min(round_index + 1, 6))
    if dominant == "elevation_low":
        return ScalarizationSpec(
            f"adaptive_high_elevation_r{round_index + 1}",
            "high_elevation",
            {
                "alpha_L": max(0.03, 0.25 / strength),
                "alpha_H": 3.0 * strength,
                "alpha_w": 3.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        )
    if dominant == "elevation_high":
        return ScalarizationSpec(
            f"adaptive_low_elevation_r{round_index + 1}",
            "low_elevation",
            {
                "alpha_L": 1.0,
                "alpha_h": 2.0 * strength,
                "alpha_w": 2.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        )
    if dominant == "width_high":
        return ScalarizationSpec(
            f"adaptive_width_r{round_index + 1}",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 4.0 * strength,
                "alpha_p": 3.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        )
    if dominant == "popularity_low":
        return ScalarizationSpec(
            f"adaptive_popularity_r{round_index + 1}",
            "hinge",
            {
                "alpha_L": 1.0,
                "alpha_w": 3.0,
                "alpha_p": 4.0 * strength,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        )
    if dominant == "length_high":
        return ScalarizationSpec(
            f"adaptive_length_high_r{round_index + 1}",
            "hinge",
            {
                "alpha_L": 1.0 + strength,
                "alpha_w": 3.0,
                "alpha_p": 2.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
            },
        )
    if dominant == "length_low":
        return ScalarizationSpec(
            f"adaptive_length_low_quality_detour_r{round_index + 1}",
            "nonlinear_high_elevation",
            {
                "form": "inverse" if round_index % 2 == 0 else "exp",
                "beta": 6.0 * strength,
                "alpha_w": 3.0,
                "alpha_p": 3.0,
                "Wref": query.Wmax,
                "Pref": query.Pmin,
                "note": "L<Lmin is recorded; no negative length cost is used.",
            },
        )
    return None


def _run_scalar_searches(
    inputs: StaticInputs,
    query: QueryBox,
    context: GraphContext,
    portfolio_mode: str,
    *,
    adaptive_rounds: int,
    stop_on_first: bool,
) -> list[ScalarSearchRecord]:
    specs = _portfolio(query, portfolio_mode)
    records: list[ScalarSearchRecord] = []
    seen_names = {spec.name for spec in specs}

    for spec in specs:
        result = _dijkstra_scalar_path(inputs.G, query, context, spec)
        validation = _validate_path(inputs.G, result)
        record = _make_search_record(
            index=len(records) + 1,
            spec=spec,
            graph_mode=context.mode,
            query=query,
            result=result,
            validation=validation,
        )
        records.append(record)
        if stop_on_first and record.feasible:
            return records

    if portfolio_mode != "adaptive" or any(record.feasible for record in records):
        return records

    for round_index in range(adaptive_rounds):
        candidate = _best_near_feasible(records)
        if candidate is None:
            break
        spec = _adaptive_spec(query, candidate, round_index)
        if spec is None:
            break
        if spec.name in seen_names:
            continue
        seen_names.add(spec.name)
        result = _dijkstra_scalar_path(inputs.G, query, context, spec)
        validation = _validate_path(inputs.G, result)
        record = _make_search_record(
            index=len(records) + 1,
            spec=spec,
            graph_mode=context.mode,
            query=query,
            result=result,
            validation=validation,
        )
        records.append(record)
        if record.feasible:
            break
    return records


def _summarize_records(records: Sequence[ScalarSearchRecord]) -> dict[str, Any]:
    first_feasible_index = next(
        (idx for idx, record in enumerate(records, start=1) if record.feasible),
        None,
    )
    if first_feasible_index is None:
        until_first = records
        first = None
    else:
        until_first = records[:first_feasible_index]
        first = records[first_feasible_index - 1]

    unique_routes = {
        record.result.edge_ids
        for record in records
        if record.result.route_found
    }
    totals = {
        "heap_pops": sum(record.result.stats.heap_pops for record in records),
        "expanded_nodes": sum(record.result.stats.expanded_nodes for record in records),
        "edge_scans": sum(record.result.stats.edge_scans for record in records),
        "elapsed_s": sum(record.result.stats.elapsed_s for record in records),
    }
    totals_to_first = {
        "heap_pops": sum(record.result.stats.heap_pops for record in until_first),
        "expanded_nodes": sum(record.result.stats.expanded_nodes for record in until_first),
        "edge_scans": sum(record.result.stats.edge_scans for record in until_first),
        "elapsed_s": sum(record.result.stats.elapsed_s for record in until_first),
    }
    best = _best_near_feasible(records)
    return {
        "feasible": first is not None,
        "time_to_first_feasible_route_s": (
            None if first is None else totals_to_first["elapsed_s"]
        ),
        "first_feasible_index": first_feasible_index,
        "first_feasible_scalar_name": None if first is None else first.spec.name,
        "searches_to_first_feasible": first_feasible_index,
        "total_scalar_shortest_path_searches": len(records),
        "unique_routes": len(unique_routes),
        "total_heap_pops": totals["heap_pops"],
        "total_expanded_nodes": totals["expanded_nodes"],
        "total_edge_scans": totals["edge_scans"],
        "total_elapsed_s": totals["elapsed_s"],
        "heap_pops_to_first": totals_to_first["heap_pops"],
        "expanded_nodes_to_first": totals_to_first["expanded_nodes"],
        "edge_scans_to_first": totals_to_first["edge_scans"],
        "best_near_feasible": None
        if best is None
        else {
            "scalar_name": best.spec.name,
            "normalized_violation_score": best.normalized_violation_score,
            "violations": best.violations,
            "metrics": None
            if best.result.metrics is None
            else best.result.metrics.as_dict(),
        },
        "first_feasible_route": None
        if first is None or first.result.metrics is None
        else {
            "scalar_name": first.spec.name,
            "metrics": first.result.metrics.as_dict(),
            "elementary": first.elementary,
            "repeated_vertex_count": first.repeated_vertex_count,
            "path_nodes": len(first.result.path_nodes),
        },
    }


def _csv_rows_from_run_payload(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query = run_payload["query"]
    for search in run_payload["searches"]:
        rows.append(
            {
                "query": query["name"],
                "portfolio_mode": run_payload["portfolio_mode"],
                "index": search["index"],
                "scalar_name": search["scalar_name"],
                "scalar_family": search["scalar_family"],
                "graph_mode": search["graph_mode"],
                "elapsed_s": search["elapsed_s"],
                "heap_pops": search["heap_pops"],
                "expanded_nodes": search["expanded_nodes"],
                "edge_scans": search["edge_scans"],
                "route_found": search["route_found"],
                "L": search["L"],
                "H": search["H"],
                "avg_pop": search["avg_pop"],
                "avg_width": search["avg_width"],
                "road_changes": search["road_changes"],
                "elementary": search["elementary"],
                "repeated_vertex_count": search["repeated_vertex_count"],
                "feasible": search["feasible"],
                "normalized_violation_score": search["normalized_violation_score"],
            }
        )
    return rows


def _format_record_line(record: ScalarSearchRecord) -> str:
    metrics = record.result.metrics
    if metrics is None:
        metric_text = "not_found"
    else:
        metric_text = (
            f"L={metrics.length:.1f} H={metrics.elevation:.1f} "
            f"pop={metrics.avg_popularity:.2f} width={metrics.avg_width:.2f} "
            f"roads={metrics.road_changes} elementary={record.elementary} "
            f"viol={record.normalized_violation_score:.4g}"
        )
    return (
        f"  {record.index:02d} {record.spec.name:<36} "
        f"{'FEASIBLE' if record.feasible else 'miss':<8} "
        f"{metric_text} "
        f"pops={record.result.stats.heap_pops} "
        f"scans={record.result.stats.edge_scans} "
        f"t={record.result.stats.elapsed_s:.4f}s"
    )


def _run_one_context(
    inputs: StaticInputs,
    query: QueryBox,
    context: GraphContext,
    portfolio_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(
        f"{query.name} {context.mode}: |V|={context.node_count} "
        f"|E|={context.edge_count} rho_H={context.constants.rho_H:.6g}",
        flush=True,
    )
    records = _run_scalar_searches(
        inputs,
        query,
        context,
        portfolio_mode,
        adaptive_rounds=args.adaptive_rounds,
        stop_on_first=args.stop_on_first,
    )
    for record in records:
        print(_format_record_line(record), flush=True)
    summary = _summarize_records(records)
    first_name = summary["first_feasible_scalar_name"]
    print(
        f"  summary feasible={summary['feasible']} first={first_name} "
        f"searches={summary['total_scalar_shortest_path_searches']} "
        f"unique_routes={summary['unique_routes']} "
        f"elapsed={summary['total_elapsed_s']:.4f}s",
        flush=True,
    )
    return {
        "query": query.as_dict(),
        "graph": context.as_dict(),
        "portfolio_mode": portfolio_mode,
        "road_id_mode": "vertex_state_tie_break_only",
        "summary": summary,
        "searches": [
            record.as_dict(include_paths=args.include_paths)
            for record in records
        ],
    }


def _metric_dict(metrics: RouteMetrics | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {
        "length": metrics.length,
        "elevation": metrics.elevation,
        "avg_pop": metrics.avg_popularity,
        "avg_width": metrics.avg_width,
    }


def _approx_metric_check(
    actual: RouteMetrics | None,
    expected: dict[str, float],
) -> tuple[bool, dict[str, float | None]]:
    if actual is None:
        return False, {key: None for key in expected}
    actual_dict = _metric_dict(actual)
    assert actual_dict is not None
    deltas = {
        key: actual_dict[key] - value
        for key, value in expected.items()
    }
    passed = all(
        abs(deltas[key]) <= REGRESSION_TOLERANCES[key]
        for key in deltas
    )
    return passed, deltas


def _route_edges_preserved(edge_mask: np.ndarray, result: ScalarPathResult) -> bool:
    if not result.route_found:
        return False
    return all(bool(edge_mask[int(edge_id)]) for edge_id in result.edge_ids)


def _tiny_direction_regression() -> bool:
    offsets = np.array([0, 1, 2, 3], dtype=np.int32)
    to = np.array([1, 2, 0], dtype=np.int32)
    w = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    road_id = np.array([1, 1, 1], dtype=np.int32)
    G = CompactDiGraph(
        offsets=offsets,
        to=to,
        w=w,
        road_id=road_id,
        n_nodes=3,
        n_edges=3,
        n_obj=4,
    )
    mask = np.ones(G.n_edges, dtype=bool)
    forward_from_2 = _physical_length_distances(G, 2, mask, reverse=False)
    reverse_to_2 = _physical_length_distances(G, 2, mask, reverse=True)
    return (
        abs(float(forward_from_2[0]) - 10.0) <= EPS
        and abs(float(reverse_to_2[0]) - 3.0) <= EPS
        and abs(float(reverse_to_2[1]) - 2.0) <= EPS
    )


def _run_regressions(
    inputs: StaticInputs,
    query: QueryBox,
    graph_comparison: dict[str, Any],
    contexts: Sequence[GraphContext],
) -> list[dict[str, Any]]:
    context_by_mode = {context.mode: context for context in contexts}
    missing_contexts = {
        mode
        for mode in ("full", "certified", "geometric")
        if mode not in context_by_mode
    }
    if missing_contexts:
        all_contexts, _ = _build_graph_contexts(inputs, query, "all")
        context_by_mode = {context.mode: context for context in all_contexts}

    full = context_by_mode["full"]
    certified = context_by_mode["certified"]
    geometric = context_by_mode["geometric"]

    retained_shortest = _dijkstra_scalar_path(
        inputs.G,
        query,
        geometric,
        ScalarizationSpec("shortest_length", "physical_length", {}),
    )
    retained_reference = _dijkstra_scalar_path(
        inputs.G,
        query,
        geometric,
        _reference_spec(query),
    )
    full_reference = _dijkstra_scalar_path(
        inputs.G,
        query,
        full,
        _reference_spec(query),
    )

    retained_shortest_validation = _validate_path(inputs.G, retained_shortest)
    retained_reference_validation = _validate_path(inputs.G, retained_reference)
    full_reference_validation = _validate_path(inputs.G, full_reference)

    checks: list[dict[str, Any]] = []
    passed, deltas = _approx_metric_check(
        retained_shortest.metrics,
        EXPECTED_RETAINED_SHORTEST,
    )
    checks.append(
        {
            "name": "retained_shortest_physical_length_metrics",
            "passed": passed and retained_shortest_validation.passed,
            "expected": EXPECTED_RETAINED_SHORTEST,
            "actual": _metric_dict(retained_shortest.metrics),
            "deltas": deltas,
            "validation": retained_shortest_validation.as_dict(),
        }
    )
    passed, deltas = _approx_metric_check(
        retained_reference.metrics,
        EXPECTED_RETAINED_REFERENCE,
    )
    checks.append(
        {
            "name": "retained_reference_scalarization_metrics",
            "passed": (
                passed
                and retained_reference_validation.passed
                and retained_reference.metrics is not None
                and query.is_feasible(retained_reference.metrics)
            ),
            "expected": EXPECTED_RETAINED_REFERENCE,
            "actual": _metric_dict(retained_reference.metrics),
            "deltas": deltas,
            "feasible": (
                retained_reference.metrics is not None
                and query.is_feasible(retained_reference.metrics)
            ),
            "validation": retained_reference_validation.as_dict(),
        }
    )
    passed, deltas = _approx_metric_check(
        full_reference.metrics,
        EXPECTED_FULL_REFERENCE,
    )
    checks.append(
        {
            "name": "full_reference_scalarization_metrics",
            "passed": (
                passed
                and full_reference_validation.passed
                and full_reference.metrics is not None
                and not query.is_feasible(full_reference.metrics)
                and full_reference.metrics.elevation < query.Hmin
            ),
            "expected": EXPECTED_FULL_REFERENCE,
            "actual": _metric_dict(full_reference.metrics),
            "deltas": deltas,
            "feasible": (
                full_reference.metrics is not None
                and query.is_feasible(full_reference.metrics)
            ),
            "validation": full_reference_validation.as_dict(),
        }
    )

    preserved_routes: list[dict[str, Any]] = []
    for name, result in (
        ("retained_shortest", retained_shortest),
        ("retained_reference", retained_reference),
        ("full_reference", full_reference),
    ):
        if result.metrics is None or result.metrics.length > query.Lmax:
            continue
        preserved_routes.append(
            {
                "route": name,
                "length": result.metrics.length,
                "preserved": _route_edges_preserved(certified.edge_mask, result),
            }
        )
    checks.append(
        {
            "name": "certified_corridor_preserves_known_routes_with_L_le_Lmax",
            "passed": all(row["preserved"] for row in preserved_routes),
            "routes": preserved_routes,
        }
    )

    all_valid = all(
        validation.passed
        for validation in (
            retained_shortest_validation,
            retained_reference_validation,
            full_reference_validation,
        )
    )
    checks.append(
        {
            "name": "known_routes_validate_edge_by_edge_on_directed_csr",
            "passed": all_valid,
        }
    )

    forward_length = graph_comparison["certified_length_corridor"][
        "shortest_length_s_to_t"
    ]
    reverse_length = graph_comparison["certified_length_corridor"][
        "shortest_length_t_reverse_to_s"
    ]
    reverse_agrees = (
        forward_length is not None
        and reverse_length is not None
        and abs(float(forward_length) - float(reverse_length)) <= 1e-6
    )
    checks.append(
        {
            "name": "forward_reverse_full_length_distances_agree_for_query",
            "passed": reverse_agrees,
            "s_to_t": forward_length,
            "reverse_t_to_s": reverse_length,
        }
    )
    checks.append(
        {
            "name": "reverse_distance_respects_edge_direction_tiny_graph",
            "passed": _tiny_direction_regression(),
        }
    )
    return checks


def _print_regressions(checks: Sequence[dict[str, Any]]) -> None:
    print("Regression checks:", flush=True)
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  {status} {check['name']}", flush=True)
        actual = check.get("actual")
        if isinstance(actual, dict):
            print(
                "    actual "
                f"L={actual['length']:.1f} H={actual['elevation']:.1f} "
                f"pop={actual['avg_pop']:.2f} width={actual['avg_width']:.2f}",
                flush=True,
            )


def _load_static_inputs(args: argparse.Namespace) -> StaticInputs:
    xy = load_xy_graph(args.graph_path)
    G = xy.G
    nodes = xy.nodes
    xy_int = build_local_xy_int(nodes)
    seeds = load_seeds(args.seeds_path, id_mode="xy")
    partition = load_partition(args.partition_path, id_mode="xy")
    boundary_nodes = load_boundary_nodes(args.boundary_nodes_path, id_mode="xy")
    return StaticInputs(
        G=G,
        nodes=nodes,
        xy_int=xy_int,
        seeds=seeds,
        partition=partition,
        boundary_nodes=boundary_nodes,
        edge_sources=_edge_sources(G),
    )


def _default_query(args: argparse.Namespace) -> QueryBox:
    return QueryBox(
        name="paris_bures",
        source=args.source,
        target=args.target,
        Lmin=args.Lmin,
        Lmax=args.Lmax,
        Hmin=args.Hmin,
        Hmax=args.Hmax,
        Pmin=args.Pmin,
        Wmax=args.Wmax,
        corridor_slack_m=args.corridor_slack_m,
        max_hops_from_boundary=args.max_hops_from_boundary,
    )


def _query_from_mapping(
    item: dict[str, Any],
    default: QueryBox,
) -> QueryBox:
    return QueryBox(
        name=str(item.get("name", default.name)),
        source=int(item.get("source", default.source)),
        target=int(item.get("target", default.target)),
        Lmin=float(item.get("Lmin", default.Lmin)),
        Lmax=float(item.get("Lmax", default.Lmax)),
        Hmin=float(item.get("Hmin", default.Hmin)),
        Hmax=float(item.get("Hmax", default.Hmax)),
        Pmin=float(item.get("Pmin", default.Pmin)),
        Wmax=float(item.get("Wmax", default.Wmax)),
        corridor_slack_m=int(item.get("corridor_slack_m", default.corridor_slack_m)),
        max_hops_from_boundary=int(
            item.get("max_hops_from_boundary", default.max_hops_from_boundary)
        ),
    )


def _load_queries(args: argparse.Namespace) -> list[QueryBox]:
    default = _default_query(args)
    if args.boxes_json is None:
        return [default]
    with open(args.boxes_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload["boxes"] if isinstance(payload, dict) and "boxes" in payload else payload
    if not isinstance(items, list):
        raise ValueError("--boxes-json must contain a list or an object with a 'boxes' list")
    return [_query_from_mapping(item, default) for item in items]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent scalar shortest-path feasibility baseline for "
            "four-criterion bicycle routing boxes."
        )
    )
    parser.add_argument("--graph-path", default=GRAPH_PATH)
    parser.add_argument("--seeds-path", default=SEEDS_PATH)
    parser.add_argument("--partition-path", default=PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=BOUNDARY_NODES_PATH)
    parser.add_argument("--source", type=int, default=PARIS_BURES_SOURCE)
    parser.add_argument("--target", type=int, default=PARIS_BURES_TARGET)
    parser.add_argument("--Lmin", type=float, default=PARIS_BURES_LMIN)
    parser.add_argument("--Lmax", type=float, default=PARIS_BURES_LMAX)
    parser.add_argument("--Hmin", type=float, default=PARIS_BURES_HMIN)
    parser.add_argument("--Hmax", type=float, default=PARIS_BURES_HMAX)
    parser.add_argument("--Pmin", type=float, default=PARIS_BURES_PMIN)
    parser.add_argument("--Wmax", type=float, default=PARIS_BURES_WMAX)
    parser.add_argument("--corridor-slack-m", type=int, default=CORRIDOR_SLACK_M)
    parser.add_argument("--max-hops-from-boundary", type=int, default=MAX_HOPS_FROM_BOUNDARY)
    parser.add_argument(
        "--graph-mode",
        choices=("all", "full", "certified", "geometric"),
        default="all",
    )
    parser.add_argument(
        "--portfolio",
        choices=("reference", "fixed", "adaptive"),
        default="fixed",
    )
    parser.add_argument("--adaptive-rounds", type=int, default=8)
    parser.add_argument("--stop-on-first", action="store_true")
    parser.add_argument("--boxes-json")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--skip-regressions", action="store_true")
    parser.add_argument("--strict-regressions", action="store_true")
    parser.add_argument("--include-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _load_static_inputs(args)
    queries = _load_queries(args)
    payload: dict[str, Any] = {
        "experiment": "scalar_feasibility_baseline",
        "formulas": {
            "route_metrics": {
                "length": "sum l(e)",
                "elevation": "sum g(e)",
                "avg_pop": "sum l(e)*p(e) / sum l(e)",
                "avg_width": "sum l(e)*w(e) / sum l(e)",
            },
            "certified_length_corridor": "d_s(u)+l(u,v)+d_t(v) <= Lmax",
            "hinge": (
                "alpha_L*l + l*(alpha_w*max(0,w-Wref)/Wscale + "
                "alpha_p*max(0,Pref-p)/Pscale)"
            ),
            "low_elevation": "alpha_L*l + alpha_h*g + optional hinge penalties",
            "high_elevation": (
                "alpha_L*l + alpha_H*max(0,rho_H*l-g) + optional hinge penalties"
            ),
            "pop_complement": (
                "alpha_L*l + alpha_pop*l*max(0,p_star-p)/max(p_star,1) "
                "+ optional width hinge"
            ),
            "width_linear": "alpha_L*l + alpha_w*l*w/Wscale + optional hinge penalties",
            "nonlinear_high_elevation": (
                "l/(1+beta*g/l) or l*exp(-beta*g/l), plus optional hinge penalties"
            ),
        },
        "violation_score_scales": (
            "length/elevation use box widths; popularity and width use their "
            "thresholds, with a lower bound of 1."
        ),
        "road_id_mode": "normal vertex-state Dijkstra; road_id only breaks exact scalar ties",
        "graph_path": args.graph_path,
        "portfolio_mode": args.portfolio,
        "graph_mode": args.graph_mode,
        "queries": [],
        "runs": [],
        "regressions": [],
    }
    csv_rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries, start=1):
        if not (0 <= query.source < inputs.G.n_nodes and 0 <= query.target < inputs.G.n_nodes):
            raise ValueError(
                f"query {query.name}: source/target outside graph range 0..{inputs.G.n_nodes - 1}"
            )
        contexts, graph_comparison = _build_graph_contexts(
            inputs,
            query,
            args.graph_mode,
        )
        payload["queries"].append(
            {
                "query": query.as_dict(),
                "graph_comparison": graph_comparison,
            }
        )
        print(
            f"Query {query_index}: {query.name} source={query.source} target={query.target}",
            flush=True,
        )
        print(
            "Graph sizes: "
            f"full |V|={graph_comparison['full']['nodes']} |E|={graph_comparison['full']['edges']}; "
            f"certified |V|={graph_comparison['certified_length_corridor']['nodes']} "
            f"|E|={graph_comparison['certified_length_corridor']['edges']}; "
            f"geometric kept |V|={graph_comparison['geometric']['kept_nodes']} "
            f"|E|={graph_comparison['geometric']['edges']}",
            flush=True,
        )
        print(
            "Set differences: "
            f"certified_not_geometric_edges="
            f"{graph_comparison['set_differences']['certified_edges_not_geometric']} "
            f"geometric_not_certified_edges="
            f"{graph_comparison['set_differences']['geometric_edges_not_certified']}",
            flush=True,
        )

        if not args.skip_regressions and query.name == "paris_bures":
            regression_contexts = contexts
            if args.graph_mode != "all":
                regression_contexts, _ = _build_graph_contexts(inputs, query, "all")
            checks = _run_regressions(
                inputs,
                query,
                graph_comparison,
                regression_contexts,
            )
            _print_regressions(checks)
            payload["regressions"].extend(checks)
            if args.strict_regressions and not all(check["passed"] for check in checks):
                _write_json(args.output_json, payload)
                raise SystemExit("one or more regression checks failed")

        for context in contexts:
            run_payload = _run_one_context(
                inputs,
                query,
                context,
                args.portfolio,
                args,
            )
            payload["runs"].append(run_payload)
            csv_rows.extend(_csv_rows_from_run_payload(run_payload))
            _write_json(args.output_json, payload)
            _write_csv(args.output_csv, csv_rows)

    print(f"Output JSON: {args.output_json}", flush=True)
    print(f"Output CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
