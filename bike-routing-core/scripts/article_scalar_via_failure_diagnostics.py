from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_benchmark as bench
import article_scalar_via_feasibility_experiment as via
import article_scalar_via_mixed_benchmark as mixed
from brcore.graph.compact import CompactDiGraph


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_failure_diagnostics_results.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_failure_diagnostics_results.csv"
DEFAULT_HOLDOUT_BOXES_JSON = mixed.DEFAULT_HOLDOUT_BOXES_JSON
DEFAULT_DEVELOPMENT_BOXES_JSON = mixed.DEFAULT_DEVELOPMENT_BOXES_JSON

LOOP_ERASURE_TARGETS = {
    "holdout": {
        "holdout_anchor_south_north_09_multi_tight",
        "holdout_anchor_south_north_28_quality_conflict",
    },
    "development": {"paris_bures_02_multi_tight"},
}
EAST_WEST_TARGET = "holdout_anchor_east_west_09_multi_tight"
PAIR_TYPES = mixed.PAIR_TYPES
EVALUATED_SCALAR_NAMES = {
    "shortest_length",
    bench.SCALAR_PHYSICAL,
    bench.SCALAR_REFERENCE,
    bench.SCALAR_SLOPE,
}
INTERPOLATION_LAMBDAS = (0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875)


@dataclass(frozen=True)
class ErasureResult:
    nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    removed_loops: int
    removed_edge_ids: tuple[int, ...]

    @property
    def removed_edges(self) -> int:
        return len(self.removed_edge_ids)


def _json_default(value: Any) -> Any:
    return mixed._json_default(value)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False, default=_json_default)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None:
        return ""
    return value


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _load_items_by_label(
    inputs: base.StaticInputs,
    holdout_path: str,
    development_path: str,
) -> dict[str, dict[str, bench.BenchmarkItem]]:
    holdout_items, _ = bench._load_items_from_boxes_json(inputs, holdout_path)
    dev_items, _ = bench._load_items_from_boxes_json(inputs, development_path)
    return {
        "holdout": {item.query.name: item for item in holdout_items},
        "development": {item.query.name: item for item in dev_items},
    }


def _route_result(
    path_nodes: tuple[int, ...],
    edge_ids: tuple[int, ...],
    metrics: base.RouteMetrics,
) -> base.ScalarPathResult:
    return via._make_path_result(0.0, path_nodes, edge_ids, metrics)


def _loop_erase_path(
    nodes_in: Sequence[int],
    edge_ids_in: Sequence[int],
) -> ErasureResult:
    nodes = [int(v) for v in nodes_in]
    edge_ids = [int(e) for e in edge_ids_in]
    removed: list[int] = []
    removed_loops = 0
    while True:
        first_seen: dict[int, int] = {}
        remove_span: tuple[int, int] | None = None
        for index, node in enumerate(nodes):
            previous = first_seen.get(node)
            if previous is not None:
                remove_span = (previous, index)
                break
            first_seen[node] = index
        if remove_span is None:
            break
        left, right = remove_span
        removed.extend(edge_ids[left:right])
        nodes = nodes[: left + 1] + nodes[right + 1 :]
        edge_ids = edge_ids[:left] + edge_ids[right:]
        removed_loops += 1
    return ErasureResult(tuple(nodes), tuple(edge_ids), removed_loops, tuple(removed))


def _constraint_loss_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        violations = row.get("after_violations") or {}
        if not isinstance(violations, dict):
            continue
        for key, value in violations.items():
            if float(value) > 1e-6:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _build_ps_trees(
    prepared: bench.PreparedGraph,
    query: base.QueryBox,
) -> tuple[dict[str, dict[str, via.TreeResult]], float]:
    specs = mixed._specs(query)
    reverse_adj = via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)
    trees: dict[str, dict[str, via.TreeResult]] = {}
    start = time.perf_counter()
    for key in ("P", "S"):
        forward = via._run_scalar_tree(
            prepared.G,
            query,
            prepared.context,
            specs[key],
            reverse=False,
        )
        backward = via._run_scalar_tree(
            prepared.G,
            query,
            prepared.context,
            specs[key],
            reverse=True,
            reverse_adj=reverse_adj,
        )
        trees[key] = {"forward": forward, "backward": backward}
    return trees, time.perf_counter() - start


def _scan_pair_profiles(
    query: base.QueryBox,
    trees: dict[str, dict[str, via.TreeResult]],
) -> tuple[dict[str, list[via.ProfileCandidate]], dict[str, dict[str, Any] | None], dict[str, float]]:
    profiles_by_pair: dict[str, list[via.ProfileCandidate]] = {}
    nearest_by_pair: dict[str, dict[str, Any] | None] = {}
    scan_times: dict[str, float] = {}
    for pair_type in PAIR_TYPES:
        forward_key, backward_key = mixed._tree_key(pair_type)
        profiles, _, nearest, elapsed = mixed._scan_pair_profiles(
            query,
            pair_type,
            trees[forward_key]["forward"],
            trees[backward_key]["backward"],
        )
        profiles_by_pair[pair_type] = profiles
        nearest_by_pair[pair_type] = nearest
        scan_times[pair_type] = elapsed
    return profiles_by_pair, nearest_by_pair, scan_times


def _loop_erasure_for_item(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    benchmark_label: str,
    item: bench.BenchmarkItem,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = item.query
    trees, tree_s = _build_ps_trees(prepared, query)
    profiles_by_pair, _, scan_times = _scan_pair_profiles(query, trees)
    rows: list[dict[str, Any]] = []
    profile_counts = {pair_type: len(profiles_by_pair[pair_type]) for pair_type in PAIR_TYPES}
    non_elementary_counts = {pair_type: 0 for pair_type in PAIR_TYPES}
    feasible_count = 0
    loop_start = time.perf_counter()
    for pair_type in PAIR_TYPES:
        forward_key, backward_key = mixed._tree_key(pair_type)
        forward = trees[forward_key]["forward"]
        backward = trees[backward_key]["backward"]
        for profile in profiles_by_pair[pair_type]:
            candidate, reason, validation, repeated = mixed._reconstruct_candidate(
                prepared,
                original_G,
                query,
                pair_type,
                forward,
                backward,
                profile,
            )
            if candidate is None or candidate.elementary:
                continue
            non_elementary_counts[pair_type] += 1
            erased = _loop_erase_path(candidate.path_nodes, candidate.edge_ids)
            after_metrics = base._metrics_from_edge_ids(original_G, erased.edge_ids)
            after_result = _route_result(erased.nodes, erased.edge_ids, after_metrics)
            after_validation = base._validate_path(original_G, after_result)
            after_repeated = via._repeated_vertex_count(erased.nodes)
            after_feasible = (
                after_validation.passed
                and after_repeated == 0
                and query.is_feasible(after_metrics)
            )
            if after_feasible:
                feasible_count += 1
            removed_metrics = base._metrics_from_edge_ids(original_G, erased.removed_edge_ids)
            rows.append(
                {
                    "part": "A_loop_erasure",
                    "benchmark": benchmark_label,
                    "query_id": query.name,
                    "pair_type": pair_type,
                    "via_vertex": candidate.via_vertex,
                    "before_repeated_vertex_count": candidate.repeated_vertex_count,
                    "before_L": candidate.metrics.length,
                    "before_H": candidate.metrics.elevation,
                    "before_avg_pop": candidate.metrics.avg_popularity,
                    "before_avg_width": candidate.metrics.avg_width,
                    "after_removed_loops": erased.removed_loops,
                    "after_removed_edges": erased.removed_edges,
                    "after_removed_length": removed_metrics.length,
                    "after_removed_elevation": removed_metrics.elevation,
                    "after_elementary": after_repeated == 0,
                    "after_L": after_metrics.length,
                    "after_H": after_metrics.elevation,
                    "after_avg_pop": after_metrics.avg_popularity,
                    "after_avg_width": after_metrics.avg_width,
                    "after_road_changes": after_metrics.road_changes,
                    "after_violations": query.violations(after_metrics),
                    "after_normalized_violation_score": query.normalized_violation_score(after_metrics),
                    "after_validation_passed": after_validation.passed,
                    "after_repeated_vertex_count": after_repeated,
                    "after_feasible": after_feasible,
                }
            )
    elapsed_s = time.perf_counter() - loop_start
    artifact = {
        "part": "A_loop_erasure",
        "benchmark": benchmark_label,
        "query_id": query.name,
        "profile_feasible_by_pair": profile_counts,
        "non_elementary_by_pair": non_elementary_counts,
        "profile_feasible_non_elementary_candidates": sum(non_elementary_counts.values()),
        "loop_erased_candidates": len(rows),
        "loop_erased_feasible_candidates": feasible_count,
        "constraint_losses_after_erasure": _constraint_loss_counts(rows),
        "tree_computation_s": tree_s,
        "profile_scan_s": sum(scan_times.values()),
        "loop_erasure_s": elapsed_s,
    }
    return rows, artifact


def _existing_non_evaluated_specs(query: base.QueryBox) -> list[base.ScalarizationSpec]:
    specs = []
    for spec in base._fixed_portfolio(query):
        if spec.name in EVALUATED_SCALAR_NAMES:
            continue
        specs.append(spec)
    return specs


def _reconstruct_same_scalar(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    scalar_name: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profiles: Sequence[via.ProfileCandidate],
) -> tuple[list[mixed.ExactCandidate], dict[str, Any] | None, int, int, int, int, float]:
    return mixed._reconstruct_exhaustive(
        prepared,
        original_G,
        query,
        scalar_name,
        forward,
        backward,
        profiles,
    )


def _best_candidate(candidates: Sequence[mixed.ExactCandidate]) -> mixed.ExactCandidate | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda c: (c.box_score, c.metrics.length, c.pair_type, c.via_vertex),
    )[0]


def _scan_same_scalar_profiles(
    query: base.QueryBox,
    scalar_name: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
) -> tuple[list[via.ProfileCandidate], dict[str, Any] | None, float]:
    profiles, _, nearest, elapsed = mixed._scan_pair_profiles(
        query,
        scalar_name,
        forward,
        backward,
    )
    return profiles, nearest, elapsed


def _run_existing_scalar_probe(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    spec: base.ScalarizationSpec,
    reverse_adj: list[list[tuple[int, int]]],
) -> dict[str, Any]:
    forward_start = time.perf_counter()
    forward = via._run_scalar_tree(
        prepared.G,
        query,
        prepared.context,
        spec,
        reverse=False,
    )
    forward_s = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    backward = via._run_scalar_tree(
        prepared.G,
        query,
        prepared.context,
        spec,
        reverse=True,
        reverse_adj=reverse_adj,
    )
    backward_s = time.perf_counter() - backward_start
    profiles, nearest, scan_s = _scan_same_scalar_profiles(
        query,
        spec.name,
        forward,
        backward,
    )
    exact, nearest_exact, reconstructed, non_elementary, validation, exact_box, recon_s = (
        _reconstruct_same_scalar(
            prepared,
            original_G,
            query,
            spec.name,
            forward,
            backward,
            profiles,
        )
    )
    best = _best_candidate(exact)
    return {
        "scalar_name": spec.name,
        "family": spec.family,
        "parameters": spec.parameters,
        "forward_tree_time_s": forward_s,
        "backward_tree_time_s": backward_s,
        "profile_scan_s": scan_s,
        "profile_feasible_count": len(profiles),
        "reconstructed_count": reconstructed,
        "non_elementary_count": non_elementary,
        "validation_failure_count": validation,
        "exact_box_rejection_count": exact_box,
        "exact_elementary_feasible_count": len(exact),
        "nearest_profile": nearest,
        "nearest_exact": nearest_exact,
        "solved": best is not None,
        "best_exact_route": None if best is None else best.as_dict(),
        "reconstruction_s": recon_s,
    }


def _edge_cost_interpolation(
    G: CompactDiGraph,
    edge_id: int,
    query: base.QueryBox,
    constants: base.MetricConstants,
    p_spec: base.ScalarizationSpec,
    s_spec: base.ScalarizationSpec,
    lambda_value: float,
) -> float:
    p_cost = base._edge_cost(G, edge_id, query, constants, p_spec)
    s_cost = base._edge_cost(G, edge_id, query, constants, s_spec)
    cost = (1.0 - lambda_value) * p_cost + lambda_value * s_cost
    if not math.isfinite(cost):
        raise ValueError(f"interpolated cost is non-finite on edge {edge_id}")
    if cost < -base.EPS:
        raise ValueError(f"interpolated cost is negative on edge {edge_id}: {cost}")
    return max(0.0, cost)


def _run_custom_tree(
    G: CompactDiGraph,
    query: base.QueryBox,
    context: base.GraphContext,
    *,
    reverse: bool,
    reverse_adj: list[list[tuple[int, int]]] | None,
    edge_cost: Callable[[int], float],
    name: str,
) -> via.TreeResult:
    origin = query.target if reverse else query.source
    direction = "backward" if reverse else "forward"
    if reverse and reverse_adj is None:
        raise ValueError("reverse_adj is required for backward interpolation tree")
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
            neighbors = via._masked_out_edges(G, context.edge_mask, node)

        for next_node, edge_id in neighbors:
            edge_scans += 1
            step_cost = edge_cost(int(edge_id))
            edge_length, edge_elevation, edge_pop_len, edge_width_len, edge_road = (
                via._edge_resources(G, int(edge_id))
            )
            if reverse:
                target_node = int(next_node)
                candidate_length = edge_length + float(length[node])
                candidate_elevation = edge_elevation + float(elevation[node])
                candidate_popularity_length = edge_pop_len + float(popularity_length[node])
                candidate_width_length = edge_width_len + float(width_length[node])
                candidate_road_changes = (
                    base._road_change_delta(edge_road, int(first_road_id[node]))
                    + int(road_changes[node])
                )
                candidate_first_road = edge_road
                candidate_last_road = (
                    int(last_road_id[node]) if int(last_road_id[node]) >= 0 else edge_road
                )
                tie_road = candidate_first_road
                tie_parent = node
            else:
                target_node = int(next_node)
                candidate_length = float(length[node]) + edge_length
                candidate_elevation = float(elevation[node]) + edge_elevation
                candidate_popularity_length = float(popularity_length[node]) + edge_pop_len
                candidate_width_length = float(width_length[node]) + edge_width_len
                candidate_road_changes = int(road_changes[node]) + base._road_change_delta(
                    int(last_road_id[node]),
                    edge_road,
                )
                candidate_first_road = (
                    int(first_road_id[node]) if int(first_road_id[node]) >= 0 else edge_road
                )
                candidate_last_road = edge_road
                tie_road = candidate_last_road
                tie_parent = node
            candidate = current_dist + step_cost
            next_tie = (candidate_road_changes, tie_road, tie_parent, int(edge_id))
            old = float(dist[target_node])
            if candidate < old - base.EPS or (
                abs(candidate - old) <= base.EPS and next_tie < best_tie[target_node]
            ):
                dist[target_node] = candidate
                best_tie[target_node] = next_tie
                parent_node[target_node] = node
                parent_edge[target_node] = int(edge_id)
                length[target_node] = candidate_length
                elevation[target_node] = candidate_elevation
                popularity_length[target_node] = candidate_popularity_length
                width_length[target_node] = candidate_width_length
                road_changes[target_node] = candidate_road_changes
                first_road_id[target_node] = candidate_first_road
                last_road_id[target_node] = candidate_last_road
                heapq.heappush(heap, (candidate, *next_tie, serial, target_node))
                serial += 1

    stats = base.DijkstraStats(
        heap_pops=heap_pops,
        expanded_nodes=expanded_nodes,
        edge_scans=edge_scans,
        raw_edge_rows_checked=raw_edge_rows_checked,
        elapsed_s=time.perf_counter() - start_time,
    )
    return via.TreeResult(
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
        stats=stats,
        tie_diagnostics=via.TreeTieDiagnostics(),
        settled_vertices=settled_vertices,
    )


def _run_interpolation_probe(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    lambda_value: float,
    reverse_adj: list[list[tuple[int, int]]],
) -> dict[str, Any]:
    specs = mixed._specs(query)
    p_spec = specs["P"]
    s_spec = specs["S"]
    name = f"interp_P_S_lambda_{lambda_value:.3f}".rstrip("0").rstrip(".")

    def cost(edge_id: int) -> float:
        return _edge_cost_interpolation(
            prepared.G,
            edge_id,
            query,
            prepared.context.constants,
            p_spec,
            s_spec,
            lambda_value,
        )

    forward = _run_custom_tree(
        prepared.G,
        query,
        prepared.context,
        reverse=False,
        reverse_adj=None,
        edge_cost=cost,
        name=name,
    )
    backward = _run_custom_tree(
        prepared.G,
        query,
        prepared.context,
        reverse=True,
        reverse_adj=reverse_adj,
        edge_cost=cost,
        name=name,
    )
    profiles, nearest, scan_s = _scan_same_scalar_profiles(
        query,
        name,
        forward,
        backward,
    )
    exact, nearest_exact, reconstructed, non_elementary, validation, exact_box, recon_s = (
        _reconstruct_same_scalar(
            prepared,
            original_G,
            query,
            name,
            forward,
            backward,
            profiles,
        )
    )
    best = _best_candidate(exact)
    return {
        "lambda": lambda_value,
        "scalar_name": name,
        "forward_tree_time_s": forward.stats.elapsed_s,
        "backward_tree_time_s": backward.stats.elapsed_s,
        "profile_scan_s": scan_s,
        "profile_feasible_count": len(profiles),
        "reconstructed_count": reconstructed,
        "non_elementary_count": non_elementary,
        "validation_failure_count": validation,
        "exact_box_rejection_count": exact_box,
        "exact_elementary_feasible_count": len(exact),
        "nearest_profile": nearest,
        "nearest_exact": nearest_exact,
        "solved": best is not None,
        "best_exact_route": None if best is None else best.as_dict(),
        "reconstruction_s": recon_s,
    }


def _witness_provenance_from_raw(path: str, query_id: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    boxes = payload["boxes"] if isinstance(payload, dict) and "boxes" in payload else payload
    for raw in boxes:
        query = raw.get("query", raw)
        if query.get("name", query.get("query_id")) != query_id:
            continue
        witness = raw.get("witness", raw)
        return {
            "route_id": witness.get("route_id"),
            "generator": witness.get("generator"),
            "scalar_name": witness.get("scalar_name"),
            "via_vertex": witness.get("via_vertex"),
            "metrics": witness.get("metrics"),
            "witness_source": raw.get("witness_source") or witness.get("witness_source"),
            "witness_method": raw.get("witness_method") or witness.get("witness_method"),
            "witness_creation_time": raw.get("witness_creation_time") or witness.get("witness_creation_time"),
            "witness_historical_file": raw.get("witness_historical_file") or witness.get("witness_historical_file"),
        }
    raise ValueError(f"{query_id} not found in {path}")


def _east_west_diagnostic(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    item: bench.BenchmarkItem,
    holdout_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = item.query
    reverse_adj = via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)
    rows: list[dict[str, Any]] = []
    witness = _witness_provenance_from_raw(holdout_path, query.name)
    existing_results: list[dict[str, Any]] = []
    for spec in _existing_non_evaluated_specs(query):
        print(f"  existing scalar {spec.name}", flush=True)
        result = _run_existing_scalar_probe(prepared, original_G, query, spec, reverse_adj)
        existing_results.append(result)
        rows.append(
            {
                "part": "B_existing_scalar",
                "benchmark": "holdout",
                "query_id": query.name,
                "scalar_name": spec.name,
                "family": spec.family,
                "profile_feasible_count": result["profile_feasible_count"],
                "exact_elementary_feasible_count": result["exact_elementary_feasible_count"],
                "non_elementary_count": result["non_elementary_count"],
                "forward_tree_time_s": result["forward_tree_time_s"],
                "backward_tree_time_s": result["backward_tree_time_s"],
                "profile_scan_s": result["profile_scan_s"],
                "reconstruction_s": result["reconstruction_s"],
                "solved": result["solved"],
                "best_exact_route": result["best_exact_route"],
                "nearest_profile": result["nearest_profile"],
            }
        )
    interpolation_results: list[dict[str, Any]] = []
    for lambda_value in INTERPOLATION_LAMBDAS:
        print(f"  interpolation lambda={lambda_value}", flush=True)
        result = _run_interpolation_probe(
            prepared,
            original_G,
            query,
            lambda_value,
            reverse_adj,
        )
        interpolation_results.append(result)
        rows.append(
            {
                "part": "B_interpolation",
                "benchmark": "holdout",
                "query_id": query.name,
                "lambda": lambda_value,
                "scalar_name": result["scalar_name"],
                "profile_feasible_count": result["profile_feasible_count"],
                "exact_elementary_feasible_count": result["exact_elementary_feasible_count"],
                "non_elementary_count": result["non_elementary_count"],
                "forward_tree_time_s": result["forward_tree_time_s"],
                "backward_tree_time_s": result["backward_tree_time_s"],
                "profile_scan_s": result["profile_scan_s"],
                "reconstruction_s": result["reconstruction_s"],
                "solved": result["solved"],
                "best_exact_route": result["best_exact_route"],
                "nearest_profile": result["nearest_profile"],
            }
        )
    artifact = {
        "part": "B_tree_reachability",
        "benchmark": "holdout",
        "query_id": query.name,
        "witness_provenance": witness,
        "existing_non_evaluated_scalarizations": existing_results,
        "fixed_interpolation_grid": interpolation_results,
        "existing_solvers": [
            result["scalar_name"] for result in existing_results if result["solved"]
        ],
        "interpolation_solvers": [
            result["scalar_name"] for result in interpolation_results if result["solved"]
        ],
    }
    return rows, artifact


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": ordered[p90_idx],
        "max": max(values),
    }


def _build_summary(artifacts: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    loop_artifacts = [a for a in artifacts if a["part"] == "A_loop_erasure"]
    part_b = next(a for a in artifacts if a["part"] == "B_tree_reachability")
    return {
        "part_a": {
            artifact["query_id"]: {
                "profile_feasible_non_elementary_candidates": artifact[
                    "profile_feasible_non_elementary_candidates"
                ],
                "loop_erased_candidates": artifact["loop_erased_candidates"],
                "loop_erased_feasible_candidates": artifact[
                    "loop_erased_feasible_candidates"
                ],
                "constraint_losses_after_erasure": artifact[
                    "constraint_losses_after_erasure"
                ],
            }
            for artifact in loop_artifacts
        },
        "part_b": {
            "query_id": part_b["query_id"],
            "witness_provenance": part_b["witness_provenance"],
            "existing_solvers": part_b["existing_solvers"],
            "interpolation_solvers": part_b["interpolation_solvers"],
            "existing_profile_feasible": {
                r["scalar_name"]: r["profile_feasible_count"]
                for r in rows
                if r["part"] == "B_existing_scalar"
            },
            "interpolation_profile_feasible": {
                str(r["lambda"]): r["profile_feasible_count"]
                for r in rows
                if r["part"] == "B_interpolation"
            },
        },
        "timing": {
            "loop_erasure_s": _stats(
                [
                    float(a["loop_erasure_s"])
                    for a in loop_artifacts
                ]
            ),
            "part_b_existing_total_s": sum(
                float(r["forward_tree_time_s"])
                + float(r["backward_tree_time_s"])
                + float(r["profile_scan_s"])
                + float(r["reconstruction_s"])
                for r in rows
                if r["part"] == "B_existing_scalar"
            ),
            "part_b_interpolation_total_s": sum(
                float(r["forward_tree_time_s"])
                + float(r["backward_tree_time_s"])
                + float(r["profile_scan_s"])
                + float(r["reconstruction_s"])
                for r in rows
                if r["part"] == "B_interpolation"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Failure-mechanism diagnostics for scalar via experiments."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--holdout-boxes-json", default=DEFAULT_HOLDOUT_BOXES_JSON)
    parser.add_argument("--development-boxes-json", default=DEFAULT_DEVELOPMENT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = bench._compute_global_metric_constants(inputs)
    prepared = mixed._full_prepared(inputs, global_constants)
    items_by_label = _load_items_by_label(
        inputs,
        args.holdout_boxes_json,
        args.development_boxes_json,
    )
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    for label, targets in LOOP_ERASURE_TARGETS.items():
        for query_id in sorted(targets):
            print(f"[part A {label}] {query_id}", flush=True)
            item = items_by_label[label][query_id]
            part_rows, artifact = _loop_erasure_for_item(
                prepared,
                inputs.G,
                label,
                item,
            )
            rows.extend(part_rows)
            artifacts.append(artifact)

    print(f"[part B holdout] {EAST_WEST_TARGET}", flush=True)
    east_west_item = items_by_label["holdout"][EAST_WEST_TARGET]
    part_rows, artifact = _east_west_diagnostic(
        prepared,
        inputs.G,
        east_west_item,
        args.holdout_boxes_json,
    )
    rows.extend(part_rows)
    artifacts.append(artifact)

    summary = _build_summary(artifacts, rows)
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_mode": "full",
            "holdout_boxes_json": args.holdout_boxes_json,
            "development_boxes_json": args.development_boxes_json,
            "frozen_boxes_policy": "boxes are loaded from existing JSON files; thresholds and witnesses are not modified",
            "validation_set_note": "the 12-box holdout is treated here as validation because its failures guide algorithm development",
            "part_a_policy": "deterministic loop erasure on reconstructed directed CSR walks; no Dijkstra",
            "part_b_policy": "existing non-evaluated scalar portfolio plus fixed non-adaptive P/S interpolation grid",
            "rho_H_global": rho_info,
        },
        "rows": rows,
        "artifacts": artifacts,
        "summary": summary,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)

    print("\nFailure diagnostics summary", flush=True)
    for query_id, item in summary["part_a"].items():
        print(
            f"  loop erasure {query_id}: {item['loop_erased_feasible_candidates']}/"
            f"{item['loop_erased_candidates']} feasible",
            flush=True,
        )
    print(
        f"  east_west existing solvers: "
        f"{summary['part_b']['existing_solvers'] or 'none'}",
        flush=True,
    )
    print(
        f"  east_west interpolation solvers: "
        f"{summary['part_b']['interpolation_solvers'] or 'none'}",
        flush=True,
    )
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
