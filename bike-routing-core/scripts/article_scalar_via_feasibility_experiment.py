from __future__ import annotations

import argparse
import csv
import json
import heapq
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
from brcore.graph.compact import CompactDiGraph


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_experiment.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_experiment.csv"
EXPECTED_FULL_REFERENCE_VIA = {
    "via_vertex": 13683,
    "length": 30217.0,
    "elevation": 413.9,
    "avg_pop": 182.05,
    "avg_width": 14.63,
}
REGRESSION_TOLERANCES = {
    "length": 5.0,
    "elevation": 1.0,
    "avg_pop": 0.5,
    "avg_width": 0.2,
}
PSEUDO_POP_UPPER = 255.0
PSEUDO_WIDTH_LOWER = 5.0


@dataclass(frozen=True)
class TreeTieDiagnostics:
    equal_cost_relaxations: int = 0
    equal_cost_resource_distinct_relaxations: int = 0
    parent_changes_due_to_road_tiebreak: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "equal_cost_relaxations": self.equal_cost_relaxations,
            "equal_cost_resource_distinct_relaxations": (
                self.equal_cost_resource_distinct_relaxations
            ),
            "parent_changes_due_to_road_tiebreak": (
                self.parent_changes_due_to_road_tiebreak
            ),
        }


@dataclass(frozen=True)
class TreeResult:
    direction: str
    dist: np.ndarray
    parent_node: np.ndarray
    parent_edge: np.ndarray
    length: np.ndarray
    elevation: np.ndarray
    popularity_length: np.ndarray
    width_length: np.ndarray
    road_changes: np.ndarray
    first_road_id: np.ndarray
    last_road_id: np.ndarray
    stats: base.DijkstraStats
    tie_diagnostics: TreeTieDiagnostics
    settled_vertices: int

    def resource_metrics(self, node: int) -> base.RouteMetrics:
        return base.RouteMetrics(
            length=float(self.length[node]),
            elevation=float(self.elevation[node]),
            popularity_length=float(self.popularity_length[node]),
            width_length=float(self.width_length[node]),
            road_changes=int(self.road_changes[node]),
        )

    def as_summary_dict(self) -> dict[str, Any]:
        out = self.stats.as_dict()
        out["settled_vertices"] = self.settled_vertices
        out["tie_diagnostics"] = self.tie_diagnostics.as_dict()
        return out


@dataclass(frozen=True)
class ProfileCandidate:
    via_vertex: int
    metrics: base.RouteMetrics

    def as_dict(self) -> dict[str, Any]:
        out = {"via_vertex": self.via_vertex}
        out.update(self.metrics.as_dict())
        return out


@dataclass(frozen=True)
class ExactViaCandidate:
    via_vertex: int
    metrics: base.RouteMetrics
    profile_metrics: base.RouteMetrics
    box_score: float
    scalar_cost: float
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    validation: base.RouteValidation
    repeated_vertex_count: int
    profile_exact_deltas: dict[str, float]

    @property
    def elementary(self) -> bool:
        return self.repeated_vertex_count == 0

    def as_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "via_vertex": self.via_vertex,
            "metrics": self.metrics.as_dict(),
            "profile_metrics": self.profile_metrics.as_dict(),
            "box_score": self.box_score,
            "scalar_cost": self.scalar_cost,
            "path_nodes": len(self.path_nodes),
            "elementary": self.elementary,
            "repeated_vertex_count": self.repeated_vertex_count,
            "validation": self.validation.as_dict(),
            "profile_exact_deltas": self.profile_exact_deltas,
        }
        if include_paths:
            out["path_node_ids"] = list(self.path_nodes)
            out["csr_edge_ids"] = list(self.edge_ids)
        return out


@dataclass
class ReconstructionCounters:
    reconstructed_count: int = 0
    rejected_non_elementary_count: int = 0
    rejected_validation_count: int = 0
    rejected_exact_box_count: int = 0
    exact_feasible_count: int = 0
    time_to_first_feasible_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reconstructed_count": self.reconstructed_count,
            "rejected_non_elementary_count": self.rejected_non_elementary_count,
            "rejected_validation_count": self.rejected_validation_count,
            "rejected_exact_box_count": self.rejected_exact_box_count,
            "exact_feasible_count": self.exact_feasible_count,
            "time_to_first_feasible_s": self.time_to_first_feasible_s,
        }


def _edge_resources(
    G: CompactDiGraph,
    edge_id: int,
) -> tuple[float, float, float, float, int]:
    row = G.w[int(edge_id)]
    length = float(row[0])
    return (
        length,
        float(row[1]),
        length * float(row[2]),
        length * float(row[3]),
        int(G.road_id[int(edge_id)]),
    )


def _resource_distinct(
    tree: TreeResult | None,
    node: int,
    length: float,
    elevation: float,
    popularity_length: float,
    width_length: float,
    road_changes: int,
    first_road_id: int,
    last_road_id: int,
) -> bool:
    if tree is None or not math.isfinite(float(tree.dist[node])):
        return False
    return (
        abs(float(tree.length[node]) - length) > 1e-6
        or abs(float(tree.elevation[node]) - elevation) > 1e-6
        or abs(float(tree.popularity_length[node]) - popularity_length) > 1e-3
        or abs(float(tree.width_length[node]) - width_length) > 1e-3
        or int(tree.road_changes[node]) != int(road_changes)
        or int(tree.first_road_id[node]) != int(first_road_id)
        or int(tree.last_road_id[node]) != int(last_road_id)
    )


def _build_reverse_edge_adjacency(
    G: CompactDiGraph,
    edge_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    reverse_adj: list[list[tuple[int, int]]] = [[] for _ in range(G.n_nodes)]
    for u in range(G.n_nodes):
        start = int(G.offsets[u])
        end = int(G.offsets[u + 1])
        for edge_id in range(start, end):
            if bool(edge_mask[edge_id]):
                reverse_adj[int(G.to[edge_id])].append((u, edge_id))
    return reverse_adj


def _masked_out_edges(
    G: CompactDiGraph,
    edge_mask: np.ndarray,
    node: int,
) -> list[tuple[int, int]]:
    start = int(G.offsets[node])
    end = int(G.offsets[node + 1])
    return [
        (int(G.to[edge_id]), edge_id)
        for edge_id in range(start, end)
        if bool(edge_mask[edge_id])
    ]


def _run_scalar_tree(
    G: CompactDiGraph,
    query: base.QueryBox,
    context: base.GraphContext,
    spec: base.ScalarizationSpec,
    *,
    reverse: bool,
    reverse_adj: list[list[tuple[int, int]]] | None = None,
) -> TreeResult:
    origin = query.target if reverse else query.source
    direction = "backward" if reverse else "forward"
    if reverse and reverse_adj is None:
        raise ValueError("reverse_adj is required for backward tree computation")

    start_time = time.perf_counter()
    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    parent_node = np.full(G.n_nodes, -1, dtype=np.int32)
    parent_edge = np.full(G.n_nodes, -1, dtype=np.int32)
    length = np.zeros(G.n_nodes, dtype=np.float64)
    elevation = np.zeros(G.n_nodes, dtype=np.float64)
    popularity_length = np.zeros(G.n_nodes, dtype=np.float64)
    width_length = np.zeros(G.n_nodes, dtype=np.float64)
    road_changes = np.zeros(G.n_nodes, dtype=np.int32)
    first_road_id = np.full(G.n_nodes, -1, dtype=np.int32)
    last_road_id = np.full(G.n_nodes, -1, dtype=np.int32)
    best_tie: list[tuple[int, int, int, int]] = [
        (2**31 - 1, 2**31 - 1, 2**31 - 1, 2**31 - 1)
        for _ in range(G.n_nodes)
    ]
    origin_tie = (0, -1, -1, -1)
    dist[origin] = 0.0
    best_tie[origin] = origin_tie
    heap: list[tuple[float, int, int, int, int, int, int]] = [
        (0.0, *origin_tie, 0, origin)
    ]
    serial = 1
    heap_pops = 0
    expanded_nodes = 0
    edge_scans = 0
    raw_edge_rows_checked = 0
    settled = np.zeros(G.n_nodes, dtype=bool)
    settled_vertices = 0
    equal_cost_relaxations = 0
    equal_cost_resource_distinct_relaxations = 0
    parent_changes_due_to_road_tiebreak = 0
    partial_tree: TreeResult | None = None

    while heap:
        item = heapq.heappop(heap)
        current_dist = float(item[0])
        current_tie = (int(item[1]), int(item[2]), int(item[3]), int(item[4]))
        node = int(item[6])
        heap_pops += 1
        if abs(current_dist - float(dist[node])) > base.EPS or current_tie != best_tie[node]:
            continue

        expanded_nodes += 1
        if not bool(settled[node]):
            settled[node] = True
            settled_vertices += 1

        if reverse:
            assert reverse_adj is not None
            neighbors = reverse_adj[node]
            raw_edge_rows_checked += len(neighbors)
        else:
            start_idx = int(G.offsets[node])
            end_idx = int(G.offsets[node + 1])
            raw_edge_rows_checked += end_idx - start_idx
            neighbors = _masked_out_edges(G, context.edge_mask, node)

        for next_node, edge_id in neighbors:
            edge_scans += 1
            step_cost = base._edge_cost(G, edge_id, query, context.constants, spec)
            if not math.isfinite(step_cost):
                raise ValueError(
                    f"{spec.name} produced a non-finite cost on edge {edge_id}"
                )
            if step_cost < -base.EPS:
                raise ValueError(
                    f"{spec.name} produced a negative cost {step_cost} on edge {edge_id}"
                )
            if step_cost < 0.0:
                step_cost = 0.0

            edge_length, edge_elevation, edge_pop_len, edge_width_len, edge_road = (
                _edge_resources(G, edge_id)
            )
            if reverse:
                target_node = int(next_node)
                candidate_length = edge_length + float(length[node])
                candidate_elevation = edge_elevation + float(elevation[node])
                candidate_popularity_length = (
                    edge_pop_len + float(popularity_length[node])
                )
                candidate_width_length = edge_width_len + float(width_length[node])
                candidate_road_changes = (
                    base._road_change_delta(edge_road, int(first_road_id[node]))
                    + int(road_changes[node])
                )
                candidate_first_road = edge_road
                candidate_last_road = (
                    int(last_road_id[node])
                    if int(last_road_id[node]) >= 0
                    else edge_road
                )
                tie_road = candidate_first_road
                tie_parent = node
            else:
                target_node = int(next_node)
                candidate_length = float(length[node]) + edge_length
                candidate_elevation = float(elevation[node]) + edge_elevation
                candidate_popularity_length = (
                    float(popularity_length[node]) + edge_pop_len
                )
                candidate_width_length = float(width_length[node]) + edge_width_len
                candidate_road_changes = int(road_changes[node]) + base._road_change_delta(
                    int(last_road_id[node]),
                    edge_road,
                )
                candidate_first_road = (
                    int(first_road_id[node])
                    if int(first_road_id[node]) >= 0
                    else edge_road
                )
                candidate_last_road = edge_road
                tie_road = candidate_last_road
                tie_parent = node

            candidate = current_dist + step_cost
            next_tie = (candidate_road_changes, tie_road, tie_parent, edge_id)
            old = float(dist[target_node])
            equal_cost = abs(candidate - old) <= base.EPS
            if equal_cost:
                equal_cost_relaxations += 1
                if partial_tree is None:
                    partial_tree = TreeResult(
                        direction=direction,
                        dist=dist,
                        parent_node=parent_node,
                        parent_edge=parent_edge,
                        length=length,
                        elevation=elevation,
                        popularity_length=popularity_length,
                        width_length=width_length,
                        road_changes=road_changes,
                        first_road_id=first_road_id,
                        last_road_id=last_road_id,
                        stats=base.DijkstraStats(0, 0, 0, 0, 0.0),
                        tie_diagnostics=TreeTieDiagnostics(),
                        settled_vertices=settled_vertices,
                    )
                if _resource_distinct(
                    partial_tree,
                    target_node,
                    candidate_length,
                    candidate_elevation,
                    candidate_popularity_length,
                    candidate_width_length,
                    candidate_road_changes,
                    candidate_first_road,
                    candidate_last_road,
                ):
                    equal_cost_resource_distinct_relaxations += 1

            if candidate < old - base.EPS or (
                equal_cost and next_tie < best_tie[target_node]
            ):
                if equal_cost and next_tie < best_tie[target_node]:
                    parent_changes_due_to_road_tiebreak += 1
                dist[target_node] = candidate
                best_tie[target_node] = next_tie
                parent_node[target_node] = node
                parent_edge[target_node] = edge_id
                length[target_node] = candidate_length
                elevation[target_node] = candidate_elevation
                popularity_length[target_node] = candidate_popularity_length
                width_length[target_node] = candidate_width_length
                road_changes[target_node] = candidate_road_changes
                first_road_id[target_node] = candidate_first_road
                last_road_id[target_node] = candidate_last_road
                heapq.heappush(heap, (candidate, *next_tie, serial, target_node))
                serial += 1

    elapsed_s = time.perf_counter() - start_time
    return TreeResult(
        direction=direction,
        dist=dist,
        parent_node=parent_node,
        parent_edge=parent_edge,
        length=length,
        elevation=elevation,
        popularity_length=popularity_length,
        width_length=width_length,
        road_changes=road_changes,
        first_road_id=first_road_id,
        last_road_id=last_road_id,
        stats=base.DijkstraStats(
            heap_pops=heap_pops,
            expanded_nodes=expanded_nodes,
            edge_scans=edge_scans,
            raw_edge_rows_checked=raw_edge_rows_checked,
            elapsed_s=elapsed_s,
        ),
        tie_diagnostics=TreeTieDiagnostics(
            equal_cost_relaxations=equal_cost_relaxations,
            equal_cost_resource_distinct_relaxations=(
                equal_cost_resource_distinct_relaxations
            ),
            parent_changes_due_to_road_tiebreak=(
                parent_changes_due_to_road_tiebreak
            ),
        ),
        settled_vertices=settled_vertices,
    )


def _combined_profile(
    query: base.QueryBox,
    forward: TreeResult,
    backward: TreeResult,
    via_vertex: int,
) -> base.RouteMetrics | None:
    if not math.isfinite(float(forward.dist[via_vertex])) or not math.isfinite(
        float(backward.dist[via_vertex])
    ):
        return None
    length = float(forward.length[via_vertex]) + float(backward.length[via_vertex])
    if length <= 0.0 and query.source != query.target:
        return None
    elevation = float(forward.elevation[via_vertex]) + float(
        backward.elevation[via_vertex]
    )
    popularity_length = float(forward.popularity_length[via_vertex]) + float(
        backward.popularity_length[via_vertex]
    )
    width_length = float(forward.width_length[via_vertex]) + float(
        backward.width_length[via_vertex]
    )
    road_changes = (
        int(forward.road_changes[via_vertex])
        + int(backward.road_changes[via_vertex])
        + base._road_change_delta(
            int(forward.last_road_id[via_vertex]),
            int(backward.first_road_id[via_vertex]),
        )
    )
    return base.RouteMetrics(
        length=length,
        elevation=elevation,
        popularity_length=popularity_length,
        width_length=width_length,
        road_changes=road_changes,
    )


def _scan_profile_feasible_vertices(
    query: base.QueryBox,
    forward: TreeResult,
    backward: TreeResult,
) -> tuple[list[ProfileCandidate], int, float]:
    start = time.perf_counter()
    profile_candidates: list[ProfileCandidate] = []
    via_vertices_scanned = 0
    for via_vertex in range(len(forward.dist)):
        metrics = _combined_profile(query, forward, backward, via_vertex)
        if metrics is None:
            continue
        via_vertices_scanned += 1
        if query.is_feasible(metrics):
            profile_candidates.append(ProfileCandidate(via_vertex, metrics))
    return profile_candidates, via_vertices_scanned, time.perf_counter() - start


def _reconstruct_forward_branch(
    G: CompactDiGraph,
    source: int,
    via_vertex: int,
    tree: TreeResult,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if source == via_vertex:
        return (source,), ()
    nodes: list[int] = []
    edge_ids: list[int] = []
    cur = via_vertex
    for _ in range(G.n_nodes + 1):
        if cur == source:
            nodes.append(source)
            nodes.reverse()
            edge_ids.reverse()
            return tuple(nodes), tuple(edge_ids)
        edge_id = int(tree.parent_edge[cur])
        prev = int(tree.parent_node[cur])
        if edge_id < 0 or prev < 0:
            return None
        nodes.append(cur)
        edge_ids.append(edge_id)
        cur = prev
    return None


def _reconstruct_backward_branch(
    G: CompactDiGraph,
    target: int,
    via_vertex: int,
    tree: TreeResult,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if target == via_vertex:
        return (target,), ()
    nodes = [via_vertex]
    edge_ids: list[int] = []
    cur = via_vertex
    for _ in range(G.n_nodes + 1):
        if cur == target:
            return tuple(nodes), tuple(edge_ids)
        edge_id = int(tree.parent_edge[cur])
        nxt = int(tree.parent_node[cur])
        if edge_id < 0 or nxt < 0:
            return None
        start = int(G.offsets[cur])
        end = int(G.offsets[cur + 1])
        if not (start <= edge_id < end) or int(G.to[edge_id]) != nxt:
            return None
        edge_ids.append(edge_id)
        cur = nxt
        nodes.append(cur)
    return None


def _repeated_vertex_count(path_nodes: Sequence[int]) -> int:
    counts: dict[int, int] = {}
    for node in path_nodes:
        counts[int(node)] = counts.get(int(node), 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _box_center_score(query: base.QueryBox, metrics: base.RouteMetrics) -> float:
    length_half = max(0.5 * (query.Lmax - query.Lmin), 1.0)
    elevation_half = max(0.5 * (query.Hmax - query.Hmin), 1.0)
    length_mid = 0.5 * (query.Lmin + query.Lmax)
    elevation_mid = 0.5 * (query.Hmin + query.Hmax)
    pop_upper = max(PSEUDO_POP_UPPER, query.Pmin + 1.0)
    pop_half = max(0.5 * (pop_upper - query.Pmin), 1.0)
    pop_mid = 0.5 * (query.Pmin + pop_upper)
    width_lower = min(PSEUDO_WIDTH_LOWER, query.Wmax - 1.0)
    width_half = max(0.5 * (query.Wmax - width_lower), 1.0)
    width_mid = 0.5 * (query.Wmax + width_lower)
    return max(
        abs(metrics.length - length_mid) / length_half,
        abs(metrics.elevation - elevation_mid) / elevation_half,
        abs(metrics.avg_popularity - pop_mid) / pop_half,
        abs(metrics.avg_width - width_mid) / width_half,
    )


def _profile_exact_deltas(
    profile: base.RouteMetrics,
    exact: base.RouteMetrics,
) -> dict[str, float]:
    return {
        "length": exact.length - profile.length,
        "elevation": exact.elevation - profile.elevation,
        "popularity_length": exact.popularity_length - profile.popularity_length,
        "width_length": exact.width_length - profile.width_length,
        "avg_pop": exact.avg_popularity - profile.avg_popularity,
        "avg_width": exact.avg_width - profile.avg_width,
        "road_changes": float(exact.road_changes - profile.road_changes),
    }


def _make_path_result(
    scalar_cost: float,
    path_nodes: tuple[int, ...],
    edge_ids: tuple[int, ...],
    metrics: base.RouteMetrics,
) -> base.ScalarPathResult:
    return base.ScalarPathResult(
        route_found=True,
        scalar_cost=scalar_cost,
        path_nodes=path_nodes,
        edge_ids=edge_ids,
        metrics=metrics,
        stats=base.DijkstraStats(0, 0, 0, 0, 0.0),
    )


def _reconstruct_validate_candidates(
    G: CompactDiGraph,
    query: base.QueryBox,
    forward: TreeResult,
    backward: TreeResult,
    profile_candidates: Sequence[ProfileCandidate],
    *,
    total_start_s: float,
) -> tuple[list[ExactViaCandidate], ReconstructionCounters, float]:
    start = time.perf_counter()
    counters = ReconstructionCounters()
    exact_candidates: list[ExactViaCandidate] = []

    for profile in profile_candidates:
        counters.reconstructed_count += 1
        via_vertex = profile.via_vertex
        forward_branch = _reconstruct_forward_branch(
            G,
            query.source,
            via_vertex,
            forward,
        )
        backward_branch = _reconstruct_backward_branch(
            G,
            query.target,
            via_vertex,
            backward,
        )
        if forward_branch is None or backward_branch is None:
            counters.rejected_validation_count += 1
            continue

        forward_nodes, forward_edges = forward_branch
        backward_nodes, backward_edges = backward_branch
        if not forward_nodes or not backward_nodes or forward_nodes[-1] != via_vertex:
            counters.rejected_validation_count += 1
            continue
        if backward_nodes[0] != via_vertex:
            counters.rejected_validation_count += 1
            continue
        path_nodes = tuple(forward_nodes + backward_nodes[1:])
        edge_ids = tuple(forward_edges + backward_edges)
        metrics = base._metrics_from_edge_ids(G, edge_ids)
        scalar_cost = float(forward.dist[via_vertex]) + float(backward.dist[via_vertex])
        result = _make_path_result(scalar_cost, path_nodes, edge_ids, metrics)
        validation = base._validate_path(G, result)
        if not validation.passed:
            counters.rejected_validation_count += 1
            continue

        repeated_count = _repeated_vertex_count(path_nodes)
        if repeated_count > 0:
            counters.rejected_non_elementary_count += 1
            continue
        if not query.is_feasible(metrics):
            counters.rejected_exact_box_count += 1
            continue

        counters.exact_feasible_count += 1
        if counters.time_to_first_feasible_s is None:
            counters.time_to_first_feasible_s = time.perf_counter() - total_start_s
        exact_candidates.append(
            ExactViaCandidate(
                via_vertex=via_vertex,
                metrics=metrics,
                profile_metrics=profile.metrics,
                box_score=_box_center_score(query, metrics),
                scalar_cost=scalar_cost,
                path_nodes=path_nodes,
                edge_ids=edge_ids,
                validation=validation,
                repeated_vertex_count=repeated_count,
                profile_exact_deltas=_profile_exact_deltas(profile.metrics, metrics),
            )
        )

    return exact_candidates, counters, time.perf_counter() - start


def _validate_backward_branch_samples(
    G: CompactDiGraph,
    query: base.QueryBox,
    backward: TreeResult,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    reachable = [
        int(node)
        for node in np.flatnonzero(np.isfinite(backward.dist))
        if int(node) != query.target
    ]
    rng = random.Random(seed)
    samples = rng.sample(reachable, min(sample_count, len(reachable)))
    if query.source in reachable and query.source not in samples:
        samples.append(query.source)

    rows: list[dict[str, Any]] = []
    passed = 0
    for node in samples:
        branch = _reconstruct_backward_branch(G, query.target, node, backward)
        if branch is None:
            rows.append({"node": node, "passed": False, "reason": "not_reconstructable"})
            continue
        path_nodes, edge_ids = branch
        metrics = base._metrics_from_edge_ids(G, edge_ids)
        result = _make_path_result(float(backward.dist[node]), path_nodes, edge_ids, metrics)
        validation = base._validate_path(G, result)
        tree_metrics = backward.resource_metrics(node)
        deltas = _profile_exact_deltas(tree_metrics, metrics)
        close = (
            validation.passed
            and abs(deltas["length"]) <= 1e-6
            and abs(deltas["elevation"]) <= 1e-6
            and abs(deltas["avg_pop"]) <= 1e-6
            and abs(deltas["avg_width"]) <= 1e-6
            and abs(deltas["road_changes"]) <= 1e-6
        )
        if close:
            passed += 1
        rows.append(
            {
                "node": node,
                "passed": close,
                "path_nodes": len(path_nodes),
                "validation": validation.as_dict(),
                "tree_recompute_deltas": deltas,
            }
        )
    return {
        "sample_count": len(samples),
        "passed_count": passed,
        "all_passed": passed == len(samples),
        "samples": rows,
    }


def _best_miss_spec(query: base.QueryBox) -> base.ScalarizationSpec:
    for spec in base._fixed_portfolio(query):
        if spec.name == "slope_exp_beta_150_width":
            return spec
    raise RuntimeError("slope_exp_beta_150_width not found in fixed portfolio")


def _selected_scalarizations(
    query: base.QueryBox,
    scalar_set: str,
) -> list[base.ScalarizationSpec]:
    reference = base._reference_spec(query)
    best_miss = _best_miss_spec(query)
    if scalar_set == "reference":
        return [reference]
    if scalar_set == "best-miss":
        return [best_miss]
    if scalar_set == "fixed":
        return base._fixed_portfolio(query)
    return [reference, best_miss]


def _candidate_sort_key(candidate: ExactViaCandidate) -> tuple[float, float, float, int]:
    return (
        candidate.box_score,
        candidate.metrics.length,
        candidate.scalar_cost,
        candidate.via_vertex,
    )


def _check_previous_full_reference_target(
    candidates: Sequence[ExactViaCandidate],
    target_profile: base.RouteMetrics | None,
) -> dict[str, Any]:
    matched = next(
        (
            candidate
            for candidate in candidates
            if candidate.via_vertex == EXPECTED_FULL_REFERENCE_VIA["via_vertex"]
        ),
        None,
    )
    if matched is None:
        return {
            "expected": EXPECTED_FULL_REFERENCE_VIA,
            "reproduced": False,
            "target_profile": None
            if target_profile is None
            else target_profile.as_dict(),
            "exact_candidate": None,
        }

    actual = {
        "via_vertex": matched.via_vertex,
        "length": matched.metrics.length,
        "elevation": matched.metrics.elevation,
        "avg_pop": matched.metrics.avg_popularity,
        "avg_width": matched.metrics.avg_width,
    }
    deltas = {
        key: actual[key] - EXPECTED_FULL_REFERENCE_VIA[key]
        for key in ("length", "elevation", "avg_pop", "avg_width")
    }
    reproduced = all(
        abs(deltas[key]) <= REGRESSION_TOLERANCES[key]
        for key in deltas
    )
    return {
        "expected": EXPECTED_FULL_REFERENCE_VIA,
        "reproduced": reproduced,
        "actual": actual,
        "deltas": deltas,
        "exact_candidate": matched.as_dict(),
    }


def _run_via_scalarization(
    inputs: base.StaticInputs,
    query: base.QueryBox,
    context: base.GraphContext,
    spec: base.ScalarizationSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    reverse_build_start = time.perf_counter()
    reverse_adj = _build_reverse_edge_adjacency(inputs.G, context.edge_mask)
    reverse_adjacency_build_s = time.perf_counter() - reverse_build_start

    forward = _run_scalar_tree(
        inputs.G,
        query,
        context,
        spec,
        reverse=False,
    )
    backward = _run_scalar_tree(
        inputs.G,
        query,
        context,
        spec,
        reverse=True,
        reverse_adj=reverse_adj,
    )
    branch_validation = _validate_backward_branch_samples(
        inputs.G,
        query,
        backward,
        sample_count=args.branch_samples,
        seed=args.branch_sample_seed,
    )
    profile_candidates, via_vertices_scanned, profile_scan_s = (
        _scan_profile_feasible_vertices(query, forward, backward)
    )
    exact_candidates, reconstruction_counters, reconstruction_s = (
        _reconstruct_validate_candidates(
            inputs.G,
            query,
            forward,
            backward,
            profile_candidates,
            total_start_s=total_start,
        )
    )
    total_s = time.perf_counter() - total_start

    best_candidates = sorted(exact_candidates, key=_candidate_sort_key)
    first_feasible = exact_candidates[0] if exact_candidates else None
    best_feasible = best_candidates[0] if best_candidates else None
    target_profile = _combined_profile(
        query,
        forward,
        backward,
        EXPECTED_FULL_REFERENCE_VIA["via_vertex"],
    )
    previous_target = None
    if context.mode == "full" and spec.name == "pop_width_reference":
        previous_target = _check_previous_full_reference_target(
            exact_candidates,
            target_profile,
        )

    tie_diagnostics = {
        "forward": forward.tie_diagnostics.as_dict(),
        "backward": backward.tie_diagnostics.as_dict(),
        "total_equal_cost_relaxations": (
            forward.tie_diagnostics.equal_cost_relaxations
            + backward.tie_diagnostics.equal_cost_relaxations
        ),
        "total_equal_cost_resource_distinct_relaxations": (
            forward.tie_diagnostics.equal_cost_resource_distinct_relaxations
            + backward.tie_diagnostics.equal_cost_resource_distinct_relaxations
        ),
        "total_parent_changes_due_to_road_tiebreak": (
            forward.tie_diagnostics.parent_changes_due_to_road_tiebreak
            + backward.tie_diagnostics.parent_changes_due_to_road_tiebreak
        ),
    }
    summary = {
        "scalar_name": spec.name,
        "scalar_family": spec.family,
        "scalar_parameters": spec.parameters,
        "graph_mode": context.mode,
        "reverse_adjacency_build_s": reverse_adjacency_build_s,
        "forward_tree_s": forward.stats.elapsed_s,
        "backward_tree_s": backward.stats.elapsed_s,
        "profile_scan_s": profile_scan_s,
        "reconstruction_s": reconstruction_s,
        "total_s": total_s,
        "via_vertices_scanned": via_vertices_scanned,
        "profile_feasible_count": len(profile_candidates),
        **reconstruction_counters.as_dict(),
        "forward_tree": forward.as_summary_dict(),
        "backward_tree": backward.as_summary_dict(),
        "tie_diagnostics": tie_diagnostics,
        "backward_branch_validation": branch_validation,
        "first_feasible_route": None
        if first_feasible is None
        else first_feasible.as_dict(include_paths=args.include_paths),
        "best_box_centered_feasible_route": None
        if best_feasible is None
        else best_feasible.as_dict(include_paths=args.include_paths),
        "best_5_exact_feasible_candidates": [
            candidate.as_dict(include_paths=args.include_paths)
            for candidate in best_candidates[:5]
        ],
        "target_via_13683_profile": None
        if target_profile is None
        else target_profile.as_dict(),
        "previous_full_reference_target": previous_target,
    }
    return summary


def _csv_row(query_name: str, run: dict[str, Any]) -> dict[str, Any]:
    best = run["best_box_centered_feasible_route"]
    best_metrics = None if best is None else best["metrics"]
    return {
        "query": query_name,
        "scalar_name": run["scalar_name"],
        "graph_mode": run["graph_mode"],
        "forward_tree_s": run["forward_tree_s"],
        "backward_tree_s": run["backward_tree_s"],
        "profile_scan_s": run["profile_scan_s"],
        "reconstruction_s": run["reconstruction_s"],
        "total_s": run["total_s"],
        "forward_heap_pops": run["forward_tree"]["heap_pops"],
        "forward_expanded_nodes": run["forward_tree"]["expanded_nodes"],
        "forward_edge_scans": run["forward_tree"]["edge_scans"],
        "backward_heap_pops": run["backward_tree"]["heap_pops"],
        "backward_expanded_nodes": run["backward_tree"]["expanded_nodes"],
        "backward_edge_scans": run["backward_tree"]["edge_scans"],
        "via_vertices_scanned": run["via_vertices_scanned"],
        "profile_feasible_count": run["profile_feasible_count"],
        "reconstructed_count": run["reconstructed_count"],
        "rejected_non_elementary_count": run["rejected_non_elementary_count"],
        "rejected_validation_count": run["rejected_validation_count"],
        "exact_feasible_count": run["exact_feasible_count"],
        "time_to_first_feasible_s": run["time_to_first_feasible_s"],
        "best_via_vertex": None if best is None else best["via_vertex"],
        "best_L": None if best_metrics is None else best_metrics["length"],
        "best_H": None if best_metrics is None else best_metrics["elevation"],
        "best_avg_pop": None if best_metrics is None else best_metrics["avg_pop"],
        "best_avg_width": None if best_metrics is None else best_metrics["avg_width"],
        "best_road_changes": None if best_metrics is None else best_metrics["road_changes"],
        "best_box_score": None if best is None else best["box_score"],
        "equal_cost_relaxations": run["tie_diagnostics"]["total_equal_cost_relaxations"],
        "equal_cost_resource_distinct_relaxations": run["tie_diagnostics"][
            "total_equal_cost_resource_distinct_relaxations"
        ],
        "parent_changes_due_to_road_tiebreak": run["tie_diagnostics"][
            "total_parent_changes_due_to_road_tiebreak"
        ],
        "backward_branch_validation_passed": run["backward_branch_validation"][
            "all_passed"
        ],
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_run(run: dict[str, Any]) -> None:
    best = run["best_box_centered_feasible_route"]
    if best is None:
        best_text = "best=none"
    else:
        metrics = best["metrics"]
        best_text = (
            f"best_via={best['via_vertex']} "
            f"L={metrics['length']:.1f} H={metrics['elevation']:.1f} "
            f"pop={metrics['avg_pop']:.2f} width={metrics['avg_width']:.2f} "
            f"score={best['box_score']:.4f}"
        )
    print(
        f"{run['graph_mode']} {run['scalar_name']}: "
        f"profile={run['profile_feasible_count']} "
        f"exact={run['exact_feasible_count']} "
        f"non_elem={run['rejected_non_elementary_count']} "
        f"invalid={run['rejected_validation_count']} "
        f"ties={run['tie_diagnostics']['total_equal_cost_relaxations']} "
        f"road_tie_parent_changes="
        f"{run['tie_diagnostics']['total_parent_changes_due_to_road_tiebreak']} "
        f"total={run['total_s']:.3f}s {best_text}",
        flush=True,
    )


def _material_difference_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_scalar_mode = {(run["scalar_name"], run["graph_mode"]): run for run in runs}
    scalar_names = sorted({run["scalar_name"] for run in runs})
    for scalar_name in scalar_names:
        full = by_scalar_mode.get((scalar_name, "full"))
        certified = by_scalar_mode.get((scalar_name, "certified"))
        if full is None or certified is None:
            continue
        full_best = full["best_box_centered_feasible_route"]
        cert_best = certified["best_box_centered_feasible_route"]
        full_via = None if full_best is None else full_best["via_vertex"]
        cert_via = None if cert_best is None else cert_best["via_vertex"]
        rows.append(
            {
                "scalar_name": scalar_name,
                "full_exact_feasible_count": full["exact_feasible_count"],
                "certified_exact_feasible_count": certified["exact_feasible_count"],
                "full_best_via": full_via,
                "certified_best_via": cert_via,
                "best_via_differs": full_via != cert_via,
                "profile_count_diff": (
                    full["profile_feasible_count"] - certified["profile_feasible_count"]
                ),
                "exact_count_diff": (
                    full["exact_feasible_count"] - certified["exact_feasible_count"]
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-scalar via-node feasibility experiment."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--source", type=int, default=base.PARIS_BURES_SOURCE)
    parser.add_argument("--target", type=int, default=base.PARIS_BURES_TARGET)
    parser.add_argument("--Lmin", type=float, default=base.PARIS_BURES_LMIN)
    parser.add_argument("--Lmax", type=float, default=base.PARIS_BURES_LMAX)
    parser.add_argument("--Hmin", type=float, default=base.PARIS_BURES_HMIN)
    parser.add_argument("--Hmax", type=float, default=base.PARIS_BURES_HMAX)
    parser.add_argument("--Pmin", type=float, default=base.PARIS_BURES_PMIN)
    parser.add_argument("--Wmax", type=float, default=base.PARIS_BURES_WMAX)
    parser.add_argument("--corridor-slack-m", type=int, default=base.CORRIDOR_SLACK_M)
    parser.add_argument(
        "--max-hops-from-boundary",
        type=int,
        default=base.MAX_HOPS_FROM_BOUNDARY,
    )
    parser.add_argument(
        "--graph-mode",
        choices=("all", "full", "certified", "geometric"),
        default="all",
    )
    parser.add_argument(
        "--scalar-set",
        choices=("first-two", "reference", "best-miss", "fixed"),
        default="first-two",
    )
    parser.add_argument("--boxes-json")
    parser.add_argument("--branch-samples", type=int, default=12)
    parser.add_argument("--branch-sample-seed", type=int, default=20260829)
    parser.add_argument("--include-paths", action="store_true")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = base._load_static_inputs(args)
    queries = base._load_queries(args)
    payload: dict[str, Any] = {
        "experiment": "same_scalar_via_node_feasibility",
        "method": (
            "For each scalar metric, compute one selected shortest-path tree "
            "from s and one selected shortest-path tree to t on the reversed "
            "directed graph, scan additive via profiles for all vertices, and "
            "reconstruct only profile-feasible concatenations."
        ),
        "road_id_mode": (
            "normal vertex-state Dijkstra; road_id only breaks exact scalar ties"
        ),
        "box_center_score": (
            "ranking-only Chebyshev distance to L/H box centers and diagnostic "
            "Paris-style popularity/width pseudo-centers P in [Pmin,255], "
            "W in [5,Wmax]; never used for pruning"
        ),
        "graph_mode": args.graph_mode,
        "scalar_set": args.scalar_set,
        "queries": [],
        "runs": [],
        "full_certified_comparison": [],
    }
    csv_rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries, start=1):
        contexts, graph_comparison = base._build_graph_contexts(
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
        specs = _selected_scalarizations(query, args.scalar_set)
        query_runs: list[dict[str, Any]] = []
        for context in contexts:
            for spec in specs:
                run = _run_via_scalarization(inputs, query, context, spec, args)
                payload["runs"].append(run)
                query_runs.append(run)
                csv_rows.append(_csv_row(query.name, run))
                _print_run(run)
                _write_json(args.output_json, payload)
                _write_csv(args.output_csv, csv_rows)
        comparisons = _material_difference_rows(query_runs)
        payload["full_certified_comparison"].extend(comparisons)
        _write_json(args.output_json, payload)
        _write_csv(args.output_csv, csv_rows)

    print(f"Output JSON: {args.output_json}", flush=True)
    print(f"Output CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
