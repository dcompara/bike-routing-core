from __future__ import annotations

import heapq
import importlib.util
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

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

SOURCE = 127
TARGET = 4433
CORRIDOR_SLACK_M = 1500
MAX_HOPS_FROM_BOUNDARY = 1

LENGTH_LOW = 30000.0
LENGTH_HIGH = 35000.0
ELEVATION_LOW = 400.0
ELEVATION_HIGH = 500.0
POPULARITY_LOW = 150.0
WIDTH_HIGH = 15.0


@dataclass(frozen=True)
class RetainedEdge:
    edge_id: int
    u: int
    v: int
    length: float
    elevation: float
    popularity: float
    width: float
    road_id: int

    @property
    def popularity_length(self) -> float:
        return self.length * self.popularity

    @property
    def width_length(self) -> float:
        return self.length * self.width


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


@dataclass(frozen=True)
class PathResult:
    name: str
    status: str
    cost: float
    elapsed_s: float
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    metrics: RouteMetrics | None


@dataclass(frozen=True)
class PathValidation:
    directed_edges_ok: bool
    missing_edges: int
    metric_deltas: tuple[float, float, float, float]
    road_changes_delta: int
    feasible: bool


@dataclass(frozen=True)
class LabelRecord:
    node: int
    length: float
    elevation: float
    popularity_length: float
    width_length: float
    last_road_id: int | None
    road_changes: int
    parent: int | None
    edge_id: int | None


def _edge_indices(G: CompactDiGraph, u: int, v: int) -> list[int]:
    start = int(G.offsets[u])
    end = int(G.offsets[u + 1])
    return [idx for idx in range(start, end) if int(G.to[idx]) == v]


def _build_retained_graph(
    G: CompactDiGraph,
    kept_nodes: set[int],
) -> tuple[list[RetainedEdge], dict[int, list[int]], dict[int, list[int]]]:
    edges: list[RetainedEdge] = []
    adjacency: dict[int, list[int]] = {node: [] for node in kept_nodes}
    reverse_adjacency: dict[int, list[int]] = {node: [] for node in kept_nodes}

    for u in sorted(kept_nodes):
        to, weights, road_ids = G.neighbors(u)
        for local_idx, v_raw in enumerate(to):
            v = int(v_raw)
            if v not in kept_nodes:
                continue
            row = weights[local_idx]
            edge = RetainedEdge(
                edge_id=int(G.offsets[u]) + local_idx,
                u=u,
                v=v,
                length=float(row[0]),
                elevation=float(row[1]),
                popularity=float(row[2]),
                width=float(row[3]),
                road_id=int(road_ids[local_idx]),
            )
            adjacency[u].append(len(edges))
            reverse_adjacency[v].append(len(edges))
            edges.append(edge)

    return edges, adjacency, reverse_adjacency


def _metrics_from_edge_ids(
    edges: list[RetainedEdge],
    edge_ids: Iterable[int],
) -> RouteMetrics:
    length = 0.0
    elevation = 0.0
    popularity_length = 0.0
    width_length = 0.0
    road_changes = 0
    previous_road_id: int | None = None
    for edge_id in edge_ids:
        edge = edges[edge_id]
        length += edge.length
        elevation += edge.elevation
        popularity_length += edge.popularity_length
        width_length += edge.width_length
        if previous_road_id is not None and edge.road_id != previous_road_id:
            road_changes += 1
        previous_road_id = edge.road_id
    return RouteMetrics(
        length=length,
        elevation=elevation,
        popularity_length=popularity_length,
        width_length=width_length,
        road_changes=road_changes,
    )


def _feasible(
    metrics: RouteMetrics,
    *,
    width_high: float = WIDTH_HIGH,
    popularity_low: float = POPULARITY_LOW,
    elevation_low: float = ELEVATION_LOW,
    elevation_high: float = ELEVATION_HIGH,
) -> bool:
    return (
        LENGTH_LOW <= metrics.length <= LENGTH_HIGH
        and elevation_low <= metrics.elevation <= elevation_high
        and metrics.popularity_length >= popularity_low * metrics.length
        and metrics.width_length <= width_high * metrics.length
    )


def _constraint_misses(
    metrics: RouteMetrics,
    *,
    width_high: float = WIDTH_HIGH,
    popularity_low: float = POPULARITY_LOW,
    elevation_low: float = ELEVATION_LOW,
    elevation_high: float = ELEVATION_HIGH,
) -> list[str]:
    misses: list[str] = []
    if metrics.length < LENGTH_LOW:
        misses.append(f"length_low_by={LENGTH_LOW - metrics.length:.1f}")
    if metrics.length > LENGTH_HIGH:
        misses.append(f"length_high_by={metrics.length - LENGTH_HIGH:.1f}")
    if metrics.elevation < elevation_low:
        misses.append(f"elevation_low_by={elevation_low - metrics.elevation:.1f}")
    if metrics.elevation > elevation_high:
        misses.append(f"elevation_high_by={metrics.elevation - elevation_high:.1f}")
    if metrics.avg_popularity < popularity_low:
        misses.append(
            f"avg_popularity_low_by={popularity_low - metrics.avg_popularity:.2f}"
        )
    if metrics.avg_width > width_high:
        misses.append(f"avg_width_high_by={metrics.avg_width - width_high:.2f}")
    return misses


def _dijkstra_path(
    name: str,
    edges: list[RetainedEdge],
    adjacency: dict[int, list[int]],
    source: int,
    target: int,
    weight: Callable[[RetainedEdge], float],
) -> PathResult:
    start = time.perf_counter()
    dist: dict[int, float] = {source: 0.0}
    parent: dict[int, tuple[int, int]] = {}
    queue: list[tuple[float, int]] = [(0.0, source)]

    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist != dist.get(node):
            continue
        if node == target:
            break
        for edge_id in adjacency.get(node, []):
            edge = edges[edge_id]
            step = weight(edge)
            if not math.isfinite(step) or step < 0.0:
                continue
            next_dist = current_dist + step
            if next_dist < dist.get(edge.v, float("inf")):
                dist[edge.v] = next_dist
                parent[edge.v] = (node, edge_id)
                heapq.heappush(queue, (next_dist, edge.v))

    elapsed = time.perf_counter() - start
    if target not in dist:
        return PathResult(name, "not_found", float("inf"), elapsed, (), (), None)

    path_nodes: list[int] = []
    edge_ids: list[int] = []
    cur = target
    while cur != source:
        path_nodes.append(cur)
        prev, edge_id = parent[cur]
        edge_ids.append(edge_id)
        cur = prev
    path_nodes.append(source)
    path_nodes.reverse()
    edge_ids.reverse()
    metrics = _metrics_from_edge_ids(edges, edge_ids)
    return PathResult(
        name=name,
        status="found",
        cost=dist[target],
        elapsed_s=elapsed,
        path_nodes=tuple(path_nodes),
        edge_ids=tuple(edge_ids),
        metrics=metrics,
    )


def _reverse_dijkstra_bounds(
    edges: list[RetainedEdge],
    reverse_adjacency: dict[int, list[int]],
    target: int,
    attr: str,
) -> dict[int, float]:
    dist: dict[int, float] = {target: 0.0}
    queue: list[tuple[float, int]] = [(0.0, target)]
    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist != dist.get(node):
            continue
        for edge_id in reverse_adjacency.get(node, []):
            edge = edges[edge_id]
            value = float(getattr(edge, attr))
            next_dist = current_dist + value
            if next_dist < dist.get(edge.u, float("inf")):
                dist[edge.u] = next_dist
                heapq.heappush(queue, (next_dist, edge.u))
    return dist


def _portfolio_specs(
    *,
    width_high: float,
    popularity_low: float,
) -> list[tuple[str, Callable[[RetainedEdge], float]]]:
    specs: list[tuple[str, Callable[[RetainedEdge], float]]] = []
    specs.append(("shortest_length", lambda e: e.length))
    specs.append(("low_width_linear", lambda e: e.length * max(e.width / width_high, 0.01)))
    specs.append(
        (
            "pop_width_reference",
            lambda e: e.length
            * (
                1.0
                + 3.0 * max(0.0, e.width - width_high) / max(width_high, 1.0)
                + 2.0
                * max(0.0, popularity_low - e.popularity)
                / max(popularity_low, 1.0)
            ),
        )
    )
    for width_penalty in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        for popularity_penalty in (1.0, 2.0, 4.0, 8.0, 12.0):
            specs.append(
                (
                    f"wp={width_penalty:g},pp={popularity_penalty:g}",
                    lambda e, wp=width_penalty, pp=popularity_penalty: e.length
                    * (
                        1.0
                        + wp * max(0.0, e.width - width_high) / max(width_high, 1.0)
                        + pp
                        * max(0.0, popularity_low - e.popularity)
                        / max(popularity_low, 1.0)
                    ),
                )
            )
    return specs


def _scalar_portfolio_search(
    edges: list[RetainedEdge],
    adjacency: dict[int, list[int]],
    source: int,
    target: int,
    *,
    width_high: float,
    popularity_low: float,
    elevation_low: float = ELEVATION_LOW,
    elevation_high: float = ELEVATION_HIGH,
) -> tuple[PathResult | None, list[PathResult]]:
    results: list[PathResult] = []
    best: PathResult | None = None
    seen_paths: set[tuple[int, ...]] = set()
    for name, weight in _portfolio_specs(
        width_high=width_high,
        popularity_low=popularity_low,
    ):
        result = _dijkstra_path(name, edges, adjacency, source, target, weight)
        if result.path_nodes in seen_paths:
            continue
        seen_paths.add(result.path_nodes)
        results.append(result)
        if result.metrics is None:
            continue
        if _feasible(
            result.metrics,
            width_high=width_high,
            popularity_low=popularity_low,
            elevation_low=elevation_low,
            elevation_high=elevation_high,
        ):
            best_metrics = best.metrics if best is not None else None
            if best is None or (
                best_metrics is not None
                and result.metrics.length < best_metrics.length
            ):
                best = result
    return best, results


def _label_contains_node(labels: list[LabelRecord], label_idx: int, node: int) -> bool:
    cur: int | None = label_idx
    while cur is not None:
        label = labels[cur]
        if label.node == node:
            return True
        cur = label.parent
    return False


def _label_path(labels: list[LabelRecord], label_idx: int) -> tuple[list[int], list[int]]:
    nodes: list[int] = []
    edge_ids: list[int] = []
    cur: int | None = label_idx
    while cur is not None:
        label = labels[cur]
        nodes.append(label.node)
        if label.edge_id is not None:
            edge_ids.append(label.edge_id)
        cur = label.parent
    nodes.reverse()
    edge_ids.reverse()
    return nodes, edge_ids


def _label_priority(
    label: LabelRecord,
    min_len_to_target: dict[int, float],
    *,
    width_high: float,
    popularity_low: float,
) -> float:
    remaining = min_len_to_target.get(label.node, float("inf"))
    lower_total = label.length + remaining
    width_excess = max(0.0, label.width_length - width_high * label.length)
    pop_deficit = max(0.0, popularity_low * label.length - label.popularity_length)
    return (
        1_000_000.0 * max(0.0, lower_total - LENGTH_HIGH)
        + abs(lower_total - 0.5 * (LENGTH_LOW + LENGTH_HIGH))
        + 8.0 * width_excess / max(label.length, 1.0)
        + 5.0 * pop_deficit / max(label.length, 1.0)
        + 20.0 * max(0.0, ELEVATION_LOW - label.elevation)
    )


def _retain_label_set(
    labels: list[LabelRecord],
    node_label_indices: list[int],
    min_len_to_target: dict[int, float],
    *,
    width_high: float,
    popularity_low: float,
    max_labels_per_node: int,
) -> list[int]:
    if len(node_label_indices) <= max_labels_per_node:
        return node_label_indices

    def key_priority(idx: int) -> tuple[float, float, int]:
        label = labels[idx]
        return (
            _label_priority(
                label,
                min_len_to_target,
                width_high=width_high,
                popularity_low=popularity_low,
            ),
            label.length,
            idx,
        )

    selectors = [
        lambda idx: key_priority(idx),
        lambda idx: (labels[idx].length, labels[idx].elevation, idx),
        lambda idx: (
            abs(labels[idx].elevation - 0.5 * (ELEVATION_LOW + ELEVATION_HIGH)),
            labels[idx].length,
            idx,
        ),
        lambda idx: (
            labels[idx].width_length - width_high * labels[idx].length,
            labels[idx].length,
            idx,
        ),
        lambda idx: (
            popularity_low * labels[idx].length - labels[idx].popularity_length,
            labels[idx].length,
            idx,
        ),
        lambda idx: (
            abs(labels[idx].length - 0.5 * (LENGTH_LOW + LENGTH_HIGH)),
            abs(labels[idx].elevation - 0.5 * (ELEVATION_LOW + ELEVATION_HIGH)),
            idx,
        ),
    ]
    kept: list[int] = []
    for selector in selectors:
        for idx in sorted(node_label_indices, key=selector):
            if idx not in kept:
                kept.append(idx)
            if len(kept) >= max_labels_per_node:
                return kept
    return kept[:max_labels_per_node]


def _bounded_label_setting_search(
    edges: list[RetainedEdge],
    adjacency: dict[int, list[int]],
    reverse_adjacency: dict[int, list[int]],
    source: int,
    target: int,
    *,
    width_high: float,
    popularity_low: float,
    max_expansions: int = 250_000,
    max_labels_per_node: int = 80,
) -> tuple[PathResult | None, dict[str, int | float | str]]:
    start = time.perf_counter()
    min_len_to_target = _reverse_dijkstra_bounds(edges, reverse_adjacency, target, "length")
    min_elev_to_target = _reverse_dijkstra_bounds(
        edges,
        reverse_adjacency,
        target,
        "elevation",
    )
    labels: list[LabelRecord] = [
        LabelRecord(source, 0.0, 0.0, 0.0, 0.0, None, 0, None, None)
    ]
    node_labels: dict[int, list[int]] = {source: [0]}
    queue: list[tuple[float, int, int]] = [
        (
            _label_priority(
                labels[0],
                min_len_to_target,
                width_high=width_high,
                popularity_low=popularity_low,
            ),
            0,
            0,
        )
    ]
    pushes = 1
    expansions = 0
    pruned = 0

    while queue and expansions < max_expansions:
        _, _, label_idx = heapq.heappop(queue)
        label = labels[label_idx]
        if label_idx not in node_labels.get(label.node, []):
            continue
        expansions += 1

        metrics = RouteMetrics(
            label.length,
            label.elevation,
            label.popularity_length,
            label.width_length,
            label.road_changes,
        )
        if label.node == target and _feasible(
            metrics,
            width_high=width_high,
            popularity_low=popularity_low,
        ):
            nodes, edge_ids = _label_path(labels, label_idx)
            return (
                PathResult(
                    "bounded_multi_resource_label_setting",
                    "found",
                    label.length,
                    time.perf_counter() - start,
                    tuple(nodes),
                    tuple(edge_ids),
                    metrics,
                ),
                {
                    "status": "found",
                    "expansions": expansions,
                    "labels_created": len(labels),
                    "queue_remaining": len(queue),
                    "pruned": pruned,
                },
            )
        if label.node == target:
            continue

        for edge_id in adjacency.get(label.node, []):
            edge = edges[edge_id]
            if _label_contains_node(labels, label_idx, edge.v):
                pruned += 1
                continue
            next_length = label.length + edge.length
            next_elevation = label.elevation + edge.elevation
            if next_length > LENGTH_HIGH or next_elevation > ELEVATION_HIGH:
                pruned += 1
                continue
            remaining_length = min_len_to_target.get(edge.v, float("inf"))
            remaining_elevation = min_elev_to_target.get(edge.v, float("inf"))
            if not math.isfinite(remaining_length) or next_length + remaining_length > LENGTH_HIGH:
                pruned += 1
                continue
            if not math.isfinite(remaining_elevation) or next_elevation + remaining_elevation > ELEVATION_HIGH:
                pruned += 1
                continue

            road_changes = label.road_changes
            if label.last_road_id is not None and edge.road_id != label.last_road_id:
                road_changes += 1
            next_label = LabelRecord(
                node=edge.v,
                length=next_length,
                elevation=next_elevation,
                popularity_length=label.popularity_length + edge.popularity_length,
                width_length=label.width_length + edge.width_length,
                last_road_id=edge.road_id,
                road_changes=road_changes,
                parent=label_idx,
                edge_id=edge_id,
            )
            labels.append(next_label)
            next_idx = len(labels) - 1
            current = node_labels.setdefault(edge.v, [])
            current.append(next_idx)
            retained = _retain_label_set(
                labels,
                current,
                min_len_to_target,
                width_high=width_high,
                popularity_low=popularity_low,
                max_labels_per_node=max_labels_per_node,
            )
            node_labels[edge.v] = retained
            if next_idx not in retained:
                continue
            priority = _label_priority(
                next_label,
                min_len_to_target,
                width_high=width_high,
                popularity_low=popularity_low,
            )
            heapq.heappush(queue, (priority, pushes, next_idx))
            pushes += 1

    return (
        None,
        {
            "status": "exhausted" if not queue else "budget_reached",
            "expansions": expansions,
            "labels_created": len(labels),
            "queue_remaining": len(queue),
            "pruned": pruned,
            "elapsed_s": time.perf_counter() - start,
        },
    )


def _validate_path(
    G: CompactDiGraph,
    retained_edges: list[RetainedEdge],
    result: PathResult,
) -> PathValidation:
    if result.metrics is None:
        return PathValidation(False, 0, (0.0, 0.0, 0.0, 0.0), 0, False)

    missing = 0
    original_edge_ids: list[int] = []
    for u, v, retained_edge_id in zip(
        result.path_nodes,
        result.path_nodes[1:],
        result.edge_ids,
    ):
        edge = retained_edges[retained_edge_id]
        if edge.u != u or edge.v != v:
            missing += 1
            continue
        if retained_edge_id < 0 or retained_edge_id >= len(retained_edges):
            missing += 1
            continue
        matches = _edge_indices(G, u, v)
        if edge.edge_id not in matches:
            missing += 1
        original_edge_ids.append(retained_edge_id)

    recomputed = _metrics_from_edge_ids(retained_edges, original_edge_ids)
    deltas = (
        recomputed.length - result.metrics.length,
        recomputed.elevation - result.metrics.elevation,
        recomputed.avg_popularity - result.metrics.avg_popularity,
        recomputed.avg_width - result.metrics.avg_width,
    )
    return PathValidation(
        directed_edges_ok=missing == 0,
        missing_edges=missing,
        metric_deltas=deltas,
        road_changes_delta=recomputed.road_changes - result.metrics.road_changes,
        feasible=_feasible(result.metrics),
    )


def _compressed_cell_sequence(
    partition: dict[int, int],
    path_nodes: Iterable[int],
) -> list[int]:
    out: list[int] = []
    previous: int | None = None
    for node in path_nodes:
        cell = partition.get(int(node), -(int(node) + 1))
        if cell != previous:
            out.append(cell)
            previous = cell
    return out


def _repeated_cells(sequence: list[int]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for idx, cell in enumerate(sequence):
        positions.setdefault(cell, []).append(idx)
    return {cell: pos for cell, pos in positions.items() if len(pos) > 1}


def _road_runs(
    retained_edges: list[RetainedEdge],
    edge_ids: Iterable[int],
    limit: int = 8,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    current_road: int | None = None
    current_len = 0
    for edge_id in edge_ids:
        road_id = retained_edges[edge_id].road_id
        if road_id == current_road:
            current_len += 1
        else:
            if current_road is not None:
                runs.append((current_road, current_len))
            current_road = road_id
            current_len = 1
    if current_road is not None:
        runs.append((current_road, current_len))
    return sorted(runs, key=lambda item: (-item[1], item[0]))[:limit]


def _print_route_summary(prefix: str, result: PathResult) -> None:
    if result.metrics is None:
        print(f"{prefix}: status={result.status}")
        return
    m = result.metrics
    print(
        f"{prefix}: name={result.name} L={m.length:.1f} H={m.elevation:.1f} "
        f"avg_pop={m.avg_popularity:.2f} avg_width={m.avg_width:.2f} "
        f"road_changes={m.road_changes} path_nodes={len(result.path_nodes)} "
        f"elapsed_s={result.elapsed_s:.3f}"
    )
    misses = _constraint_misses(m)
    print(f"  misses={misses if misses else 'none'}")


def _run_sensitivity(
    edges: list[RetainedEdge],
    adjacency: dict[int, list[int]],
    source: int,
    target: int,
) -> None:
    print("Sensitivity:")
    first_feasible: tuple[float, float, PathResult] | None = None
    for width_high in (15.0, 16.0, 17.0, 18.0, 19.0):
        for popularity_low in (150.0, 145.0, 140.0):
            best, _ = _scalar_portfolio_search(
                edges,
                adjacency,
                source,
                target,
                width_high=width_high,
                popularity_low=popularity_low,
            )
            if best is None or best.metrics is None:
                print(
                    f"  width<={width_high:.0f} pop>={popularity_low:.0f}: not_found"
                )
                continue
            m = best.metrics
            print(
                f"  width<={width_high:.0f} pop>={popularity_low:.0f}: "
                f"found L={m.length:.1f} H={m.elevation:.1f} "
                f"pop={m.avg_popularity:.2f} width={m.avg_width:.2f} "
                f"name={best.name}"
            )
            if first_feasible is None:
                first_feasible = (width_high, popularity_low, best)
    if first_feasible is None:
        print("  smallest relaxed combination found: none")
    else:
        width_high, popularity_low, _ = first_feasible
        print(
            "  smallest relaxed combination found by scan: "
            f"width<={width_high:.0f}, pop>={popularity_low:.0f}"
        )


def main() -> None:
    setup_start = time.perf_counter()
    xy = load_xy_graph(GRAPH_PATH)
    G = xy.G
    nodes = xy.nodes
    xy_int = build_local_xy_int(nodes)
    seeds = load_seeds(SEEDS_PATH, id_mode="xy")
    partition = load_partition(PARTITION_PATH, id_mode="xy")
    boundary_nodes = load_boundary_nodes(BOUNDARY_NODES_PATH, id_mode="xy")
    kept_cells, kept_nodes = search_space_reduction(
        G=G,
        xy_int=xy_int,
        seeds=seeds,
        partition=partition,
        boundary_nodes=boundary_nodes,
        s=SOURCE,
        t=TARGET,
        corridor_slack_m=CORRIDOR_SLACK_M,
        max_hops_from_boundary=MAX_HOPS_FROM_BOUNDARY,
    )
    retained_edges, adjacency, reverse_adjacency = _build_retained_graph(
        G,
        kept_nodes,
    )
    setup_s = time.perf_counter() - setup_start

    print("Paris-Bures independent retained-graph feasibility study")
    print(f"  scipy_milp_available={importlib.util.find_spec('scipy') is not None}")
    print("  exact_method_used=False")
    print("  method=scalarized Dijkstra portfolio plus bounded label-setting fallback")
    print(f"  source={SOURCE} target={TARGET}")
    print(f"  kept_cells={len(kept_cells)} kept_nodes={len(kept_nodes)}")
    print(f"  retained_directed_edges={len(retained_edges)} setup_s={setup_s:.3f}")
    print(
        "  constraints: "
        f"{LENGTH_LOW:.0f}<=L<={LENGTH_HIGH:.0f}, "
        f"{ELEVATION_LOW:.0f}<=H<={ELEVATION_HIGH:.0f}, "
        f"avg_pop>={POPULARITY_LOW:.0f}, avg_width<={WIDTH_HIGH:.0f}"
    )

    shortest = _dijkstra_path(
        "shortest_length",
        retained_edges,
        adjacency,
        SOURCE,
        TARGET,
        lambda edge: edge.length,
    )
    print("Reference route:")
    _print_route_summary("  shortest retained length", shortest)

    portfolio_start = time.perf_counter()
    best, portfolio_results = _scalar_portfolio_search(
        retained_edges,
        adjacency,
        SOURCE,
        TARGET,
        width_high=WIDTH_HIGH,
        popularity_low=POPULARITY_LOW,
    )
    portfolio_s = time.perf_counter() - portfolio_start
    print("Portfolio search:")
    print(
        f"  scalar_paths_tested={len(portfolio_results)} "
        f"solver_search_s={portfolio_s:.3f}"
    )
    if best is not None:
        _print_route_summary("  best feasible portfolio route", best)
    else:
        print("  best feasible portfolio route: not_found")

    label_result, label_stats = _bounded_label_setting_search(
        retained_edges,
        adjacency,
        reverse_adjacency,
        SOURCE,
        TARGET,
        width_high=WIDTH_HIGH,
        popularity_low=POPULARITY_LOW,
    )
    print("Bounded multi-resource label-setting:")
    print(f"  stats={label_stats}")
    if label_result is not None:
        _print_route_summary("  label-setting feasible route", label_result)

    chosen = best or label_result
    if chosen is None:
        print("Feasibility result: no feasible route found by diagnostic search")
        print(
            "Conclusion C: no feasible route found, but the fallback diagnostic "
            "is not exact enough to prove infeasibility."
        )
        _run_sensitivity(retained_edges, adjacency, SOURCE, TARGET)
        return

    validation = _validate_path(G, retained_edges, chosen)
    compressed_cells = _compressed_cell_sequence(partition, chosen.path_nodes)
    repeated = _repeated_cells(compressed_cells)
    full_cells = [partition.get(node, -(node + 1)) for node in chosen.path_nodes]
    print("Chosen feasible route validation:")
    print(
        f"  directed_edges_ok={validation.directed_edges_ok} "
        f"missing_edges={validation.missing_edges} "
        f"metric_deltas={validation.metric_deltas} "
        f"road_changes_delta={validation.road_changes_delta} "
        f"constraint_feasible={validation.feasible}"
    )
    print(f"  full_cell_sequence={full_cells}")
    print(f"  compressed_cell_sequence={compressed_cells}")
    print(f"  repeated_cells={repeated if repeated else 'none'}")
    print(f"  no_cell_revisit_compatible={not repeated}")
    print(f"  longest_road_id_runs={_road_runs(retained_edges, chosen.edge_ids)}")

    _run_sensitivity(retained_edges, adjacency, SOURCE, TARGET)

    if validation.feasible and not repeated:
        print(
            "Conclusion A: a fully feasible retained-graph Paris-Bures route "
            "exists and it does not revisit a partition cell."
        )
    elif validation.feasible:
        print(
            "Conclusion B: a feasible retained-graph Paris-Bures route exists, "
            "but it violates the production no-cell-revisit approximation."
        )
    else:
        print(
            "Conclusion C: a candidate route was found, but validation did not "
            "certify feasibility."
        )
    print("Full graph comparison: not run because the retained graph is feasible.")


if __name__ == "__main__":
    main()
