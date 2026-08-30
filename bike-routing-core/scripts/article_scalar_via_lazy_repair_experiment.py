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
from typing import Any, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_benchmark as bench
import article_scalar_via_feasibility_experiment as via
import article_scalar_via_mixed_benchmark as mixed
from brcore.graph.compact import CompactDiGraph


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_lazy_repair_results.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_lazy_repair_results.csv"
DEFAULT_HOLDOUT_BOXES_JSON = mixed.DEFAULT_HOLDOUT_BOXES_JSON
DEFAULT_DEVELOPMENT_BOXES_JSON = mixed.DEFAULT_DEVELOPMENT_BOXES_JSON

PAIR_TYPES = mixed.PAIR_TYPES
SAME_PAIR_TYPES = mixed.SAME_PAIR_TYPES
MIXED_PAIR_TYPES = mixed.MIXED_PAIR_TYPES
VIA_UNION_2 = mixed.VIA_UNION_2
LAZY_VIA_REPAIR = "LAZY-VIA-REPAIR"

HOLDOUT_TARGET_IDS = mixed.HOLDOUT_FAILURE_QUERY_IDS
DEVELOPMENT_TARGET_IDS = ("paris_bures_02_multi_tight",)


@dataclass(frozen=True)
class RepairPathResult:
    route_found: bool
    scalar_cost: float
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    elapsed_s: float
    heap_pops: int
    expanded_nodes: int
    edge_scans: int
    raw_edge_rows_checked: int
    endpoint_forbidden: bool = False


@dataclass(frozen=True)
class RepairCandidate:
    original_pair_type: str
    via_vertex: int
    repair_direction: str
    metrics: base.RouteMetrics
    profile_metrics: base.RouteMetrics
    box_score: float
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    validation: base.RouteValidation
    repeated_vertex_count: int
    original_repeated_vertex_count: int
    original_overlap_vertex_count: int
    original_overlap_vertices_sample: tuple[int, ...]

    @property
    def elementary(self) -> bool:
        return self.repeated_vertex_count == 0

    def as_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "original_pair_type": self.original_pair_type,
            "via_vertex": self.via_vertex,
            "repair_direction": self.repair_direction,
            "metrics": self.metrics.as_dict(),
            "box_score": self.box_score,
            "profile_metrics": self.profile_metrics.as_dict(),
            "path_nodes": len(self.path_nodes),
            "edges": len(self.edge_ids),
            "elementary": self.elementary,
            "repeated_vertex_count": self.repeated_vertex_count,
            "validation": self.validation.as_dict(),
            "original_repeated_vertex_count": self.original_repeated_vertex_count,
            "original_overlap_vertex_count": self.original_overlap_vertex_count,
            "original_overlap_vertices_sample": list(self.original_overlap_vertices_sample),
        }
        if include_paths:
            out["path_node_ids"] = list(self.path_nodes)
            out["csr_edge_ids"] = list(self.edge_ids)
        return out


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


def _scalar_key_spec(
    query: base.QueryBox,
) -> dict[str, base.ScalarizationSpec]:
    return mixed._specs(query)


def _branch_pair(
    prepared: bench.PreparedGraph,
    query: base.QueryBox,
    pair_type: str,
    trees: dict[str, dict[str, via.TreeResult]],
    via_vertex: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    forward_key, backward_key = mixed._tree_key(pair_type)
    forward = trees[forward_key]["forward"]
    backward = trees[backward_key]["backward"]
    forward_branch = via._reconstruct_forward_branch(
        prepared.G,
        query.source,
        via_vertex,
        forward,
    )
    backward_branch = via._reconstruct_backward_branch(
        prepared.G,
        query.target,
        via_vertex,
        backward,
    )
    if forward_branch is None or backward_branch is None:
        return None
    return (*forward_branch, *backward_branch)


def _make_result(
    route_found: bool,
    scalar_cost: float,
    path_nodes: tuple[int, ...],
    edge_ids: tuple[int, ...],
    elapsed_s: float,
    heap_pops: int,
    expanded_nodes: int,
    edge_scans: int,
    raw_edge_rows_checked: int,
    endpoint_forbidden: bool = False,
) -> RepairPathResult:
    return RepairPathResult(
        route_found=route_found,
        scalar_cost=scalar_cost,
        path_nodes=path_nodes,
        edge_ids=edge_ids,
        elapsed_s=elapsed_s,
        heap_pops=heap_pops,
        expanded_nodes=expanded_nodes,
        edge_scans=edge_scans,
        raw_edge_rows_checked=raw_edge_rows_checked,
        endpoint_forbidden=endpoint_forbidden,
    )


def _dijkstra_scalar_path_forbidden(
    G: CompactDiGraph,
    query: base.QueryBox,
    context: base.GraphContext,
    spec: base.ScalarizationSpec,
    source: int,
    target: int,
    forbidden_vertices: set[int],
) -> RepairPathResult:
    start_time = time.perf_counter()
    if not (0 <= source < G.n_nodes and 0 <= target < G.n_nodes):
        raise ValueError("repair source or target is outside graph node id range")
    if source in forbidden_vertices or target in forbidden_vertices:
        return _make_result(
            False,
            float("inf"),
            (),
            (),
            time.perf_counter() - start_time,
            0,
            0,
            0,
            0,
            True,
        )
    if source == target:
        return _make_result(
            True,
            0.0,
            (source,),
            (),
            time.perf_counter() - start_time,
            0,
            0,
            0,
            0,
        )

    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    parent_node = np.full(G.n_nodes, -1, dtype=np.int32)
    parent_edge = np.full(G.n_nodes, -1, dtype=np.int32)
    best_tie: list[tuple[int, int, int, int]] = [
        (2**31 - 1, 2**31 - 1, 2**31 - 1, 2**31 - 1)
        for _ in range(G.n_nodes)
    ]
    source_tie = (0, -1, -1, -1)
    dist[source] = 0.0
    best_tie[source] = source_tie
    heap: list[tuple[float, int, int, int, int, int, int]] = [
        (0.0, *source_tie, 0, source)
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
        if abs(current_dist - float(dist[node])) > base.EPS or current_tie != best_tie[node]:
            continue
        expanded_nodes += 1
        if node == target:
            break

        start_idx = int(G.offsets[node])
        end_idx = int(G.offsets[node + 1])
        raw_edge_rows_checked += end_idx - start_idx
        current_road_changes = current_tie[0]
        current_last_road = current_tie[1]
        for edge_id in range(start_idx, end_idx):
            if not bool(context.edge_mask[edge_id]):
                continue
            nxt = int(G.to[edge_id])
            if nxt in forbidden_vertices:
                continue
            edge_scans += 1
            step_cost = base._edge_cost(G, edge_id, query, context.constants, spec)
            if not math.isfinite(step_cost):
                raise ValueError(f"{spec.name} produced a non-finite cost on edge {edge_id}")
            if step_cost < -base.EPS:
                raise ValueError(f"{spec.name} produced a negative cost {step_cost} on edge {edge_id}")
            if step_cost < 0.0:
                step_cost = 0.0

            edge_road_id = int(G.road_id[edge_id])
            next_road_changes = current_road_changes + base._road_change_delta(
                current_last_road,
                edge_road_id,
            )
            next_tie = (next_road_changes, edge_road_id, node, edge_id)
            candidate = current_dist + step_cost
            old = float(dist[nxt])
            if candidate < old - base.EPS or (
                abs(candidate - old) <= base.EPS and next_tie < best_tie[nxt]
            ):
                dist[nxt] = candidate
                best_tie[nxt] = next_tie
                parent_node[nxt] = node
                parent_edge[nxt] = edge_id
                heapq.heappush(heap, (candidate, *next_tie, serial, nxt))
                serial += 1

    elapsed_s = time.perf_counter() - start_time
    if not math.isfinite(float(dist[target])):
        return _make_result(
            False,
            float("inf"),
            (),
            (),
            elapsed_s,
            heap_pops,
            expanded_nodes,
            edge_scans,
            raw_edge_rows_checked,
        )

    nodes: list[int] = []
    edge_ids: list[int] = []
    cur = target
    while cur != source:
        edge_id = int(parent_edge[cur])
        prev = int(parent_node[cur])
        if edge_id < 0 or prev < 0:
            return _make_result(
                False,
                float("inf"),
                (),
                (),
                elapsed_s,
                heap_pops,
                expanded_nodes,
                edge_scans,
                raw_edge_rows_checked,
            )
        nodes.append(cur)
        edge_ids.append(edge_id)
        cur = prev
    nodes.append(source)
    nodes.reverse()
    edge_ids.reverse()
    return _make_result(
        True,
        float(dist[target]),
        tuple(nodes),
        tuple(edge_ids),
        elapsed_s,
        heap_pops,
        expanded_nodes,
        edge_scans,
        raw_edge_rows_checked,
    )


def _candidate_from_repair(
    *,
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_type: str,
    repair_direction: str,
    profile: via.ProfileCandidate,
    path_nodes: tuple[int, ...],
    run_edges: tuple[int, ...],
    original_repeated_vertex_count: int,
    overlap_vertices: Sequence[int],
) -> RepairCandidate | None:
    original_edges = tuple(
        int(prepared.edge_id_to_original[int(edge_id)]) for edge_id in run_edges
    )
    metrics = base._metrics_from_edge_ids(original_G, original_edges)
    result = via._make_path_result(0.0, path_nodes, original_edges, metrics)
    validation = base._validate_path(original_G, result)
    repeated = via._repeated_vertex_count(path_nodes)
    if not validation.passed:
        return None
    return RepairCandidate(
        original_pair_type=pair_type,
        via_vertex=int(profile.via_vertex),
        repair_direction=repair_direction,
        metrics=metrics,
        profile_metrics=profile.metrics,
        box_score=via._box_center_score(query, metrics),
        path_nodes=path_nodes,
        edge_ids=original_edges,
        validation=validation,
        repeated_vertex_count=repeated,
        original_repeated_vertex_count=original_repeated_vertex_count,
        original_overlap_vertex_count=len(overlap_vertices),
        original_overlap_vertices_sample=tuple(int(v) for v in overlap_vertices[:20]),
    )


def _repair_non_elementary_candidate(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    specs: dict[str, base.ScalarizationSpec],
    pair_type: str,
    profile: via.ProfileCandidate,
    branch_data: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    original_repeated_vertex_count: int,
) -> tuple[list[RepairCandidate], dict[str, Any]]:
    forward_nodes, forward_edges, backward_nodes, backward_edges = branch_data
    via_vertex = int(profile.via_vertex)
    overlap_vertices = sorted((set(forward_nodes) & set(backward_nodes)) - {via_vertex})
    forward_key, backward_key = mixed._tree_key(pair_type)
    candidates: list[RepairCandidate] = []
    call_records: list[dict[str, Any]] = []

    forbidden_a = set(int(v) for v in forward_nodes)
    forbidden_a.discard(via_vertex)
    repair_a = _dijkstra_scalar_path_forbidden(
        prepared.G,
        query,
        prepared.context,
        specs[backward_key],
        via_vertex,
        query.target,
        forbidden_a,
    )
    call_records.append(
        {
            "repair_direction": "A",
            "route_found": repair_a.route_found,
            "elapsed_s": repair_a.elapsed_s,
            "endpoint_forbidden": repair_a.endpoint_forbidden,
            "heap_pops": repair_a.heap_pops,
            "expanded_nodes": repair_a.expanded_nodes,
            "edge_scans": repair_a.edge_scans,
        }
    )
    if repair_a.route_found:
        path_nodes = tuple(forward_nodes + repair_a.path_nodes[1:])
        run_edges = tuple(forward_edges + repair_a.edge_ids)
        candidate = _candidate_from_repair(
            prepared=prepared,
            original_G=original_G,
            query=query,
            pair_type=pair_type,
            repair_direction="A",
            profile=profile,
            path_nodes=path_nodes,
            run_edges=run_edges,
            original_repeated_vertex_count=original_repeated_vertex_count,
            overlap_vertices=overlap_vertices,
        )
        if candidate is not None:
            candidates.append(candidate)

    forbidden_b = set(int(v) for v in backward_nodes)
    forbidden_b.discard(via_vertex)
    repair_b = _dijkstra_scalar_path_forbidden(
        prepared.G,
        query,
        prepared.context,
        specs[forward_key],
        query.source,
        via_vertex,
        forbidden_b,
    )
    call_records.append(
        {
            "repair_direction": "B",
            "route_found": repair_b.route_found,
            "elapsed_s": repair_b.elapsed_s,
            "endpoint_forbidden": repair_b.endpoint_forbidden,
            "heap_pops": repair_b.heap_pops,
            "expanded_nodes": repair_b.expanded_nodes,
            "edge_scans": repair_b.edge_scans,
        }
    )
    if repair_b.route_found:
        path_nodes = tuple(repair_b.path_nodes + backward_nodes[1:])
        run_edges = tuple(repair_b.edge_ids + backward_edges)
        candidate = _candidate_from_repair(
            prepared=prepared,
            original_G=original_G,
            query=query,
            pair_type=pair_type,
            repair_direction="B",
            profile=profile,
            path_nodes=path_nodes,
            run_edges=run_edges,
            original_repeated_vertex_count=original_repeated_vertex_count,
            overlap_vertices=overlap_vertices,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates, {
        "overlap_vertex_count": len(overlap_vertices),
        "overlap_vertices_sample": overlap_vertices[:20],
        "calls": call_records,
    }


def _build_trees(
    prepared: bench.PreparedGraph,
    query: base.QueryBox,
    specs: dict[str, base.ScalarizationSpec],
) -> tuple[dict[str, dict[str, via.TreeResult]], float]:
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


def _run_normal_union2(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    item: bench.BenchmarkItem,
    trees: dict[str, dict[str, via.TreeResult]],
    tree_total_s: float,
) -> tuple[dict[str, mixed.PairRun], mixed.ExactCandidate | None, dict[str, Any]]:
    pair_runs = {
        pair_type: mixed._run_pair_type(prepared, original_G, item.query, pair_type, trees)
        for pair_type in SAME_PAIR_TYPES
    }
    first, reconstructed, non_elementary, validation, exact_box, first_recon_s = (
        mixed._first_hit_union(
            prepared,
            original_G,
            item.query,
            pair_runs,
            SAME_PAIR_TYPES,
            trees,
        )
    )
    timing = {
        "tree_computation_s": tree_total_s,
        "same_scalar_profile_scan_s": sum(pair_runs[p].profile_scan_s for p in SAME_PAIR_TYPES),
        "same_scalar_exhaustive_reconstruction_s": sum(
            pair_runs[p].exhaustive_reconstruction_s for p in SAME_PAIR_TYPES
        ),
        "first_hit_reconstruction_s": first_recon_s,
        "normal_time_to_first_feasible_s": None
        if first is None
        else tree_total_s
        + sum(pair_runs[p].profile_scan_s for p in SAME_PAIR_TYPES)
        + first_recon_s,
        "normal_exhaustive_total_s": tree_total_s
        + sum(pair_runs[p].profile_scan_s for p in SAME_PAIR_TYPES)
        + sum(pair_runs[p].exhaustive_reconstruction_s for p in SAME_PAIR_TYPES),
        "first_hit_reconstructed": reconstructed,
        "first_hit_non_elementary_rejections": non_elementary,
        "first_hit_validation_rejections": validation,
        "first_hit_exact_box_rejections": exact_box,
    }
    return pair_runs, first, timing


def _scan_mixed_profiles(
    query: base.QueryBox,
    trees: dict[str, dict[str, via.TreeResult]],
) -> tuple[dict[str, list[via.ProfileCandidate]], dict[str, dict[str, Any] | None], dict[str, float]]:
    profiles_by_pair: dict[str, list[via.ProfileCandidate]] = {}
    nearest_by_pair: dict[str, dict[str, Any] | None] = {}
    scan_times: dict[str, float] = {}
    for pair_type in MIXED_PAIR_TYPES:
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


def _best_nearest_from_maps(
    query: base.QueryBox,
    pair_runs: dict[str, mixed.PairRun],
    mixed_nearest: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for pair_type in SAME_PAIR_TYPES:
        nearest = pair_runs[pair_type].nearest_profile
        if nearest is not None:
            candidates.append(nearest)
    for nearest in mixed_nearest.values():
        if nearest is not None:
            candidates.append(nearest)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            float(c["normalized_violation_score"]),
            via._box_center_score(query, base.RouteMetrics(
                length=float(c["metrics"]["length"]),
                elevation=float(c["metrics"]["elevation"]),
                popularity_length=float(c["metrics"]["popularity_length"]),
                width_length=float(c["metrics"]["width_length"]),
                road_changes=int(c["metrics"]["road_changes"]),
            )),
            str(c["pair_type"]),
            int(c["via_vertex"]),
        ),
    )


def _run_repair_fallback(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    item: bench.BenchmarkItem,
    specs: dict[str, base.ScalarizationSpec],
    trees: dict[str, dict[str, via.TreeResult]],
    same_pair_runs: dict[str, mixed.PairRun],
) -> dict[str, Any]:
    query = item.query
    fallback_start = time.perf_counter()
    mixed_profiles, mixed_nearest, mixed_scan_times = _scan_mixed_profiles(query, trees)
    profiles_by_pair: dict[str, list[via.ProfileCandidate]] = {
        pair_type: same_pair_runs[pair_type].profile_candidates
        for pair_type in SAME_PAIR_TYPES
    }
    profiles_by_pair.update(mixed_profiles)

    non_elementary_records: list[dict[str, Any]] = []
    repaired_candidates: list[RepairCandidate] = []
    repair_call_count_a = 0
    repair_call_count_b = 0
    repair_dijkstra_s = 0.0
    validated_repair_count = 0
    repaired_non_elementary_count = 0
    repaired_exact_box_rejected_count = 0
    repair_validation_failure_count = 0
    first_repaired_feasible: RepairCandidate | None = None
    time_to_first_repaired_s: float | None = None
    identify_start = time.perf_counter()

    ordered: list[tuple[str, via.ProfileCandidate]] = []
    for pair_type in PAIR_TYPES:
        ordered.extend((pair_type, profile) for profile in profiles_by_pair[pair_type])
    ordered.sort(key=lambda item_pair: mixed._profile_sort_key(query, item_pair[0], item_pair[1]))

    reconstructed_for_repair = 0
    branch_validation_failures = 0
    non_elementary_count = 0
    overlap_repairable_count = 0
    for pair_type, profile in ordered:
        reconstructed_for_repair += 1
        forward_key, backward_key = mixed._tree_key(pair_type)
        candidate, reason, validation, repeated = mixed._reconstruct_candidate(
            prepared,
            original_G,
            query,
            pair_type,
            trees[forward_key]["forward"],
            trees[backward_key]["backward"],
            profile,
        )
        if candidate is None:
            branch_validation_failures += 1
            continue
        if candidate.elementary:
            continue
        non_elementary_count += 1
        branch_data = _branch_pair(
            prepared,
            query,
            pair_type,
            trees,
            int(profile.via_vertex),
        )
        if branch_data is None:
            branch_validation_failures += 1
            continue
        forward_nodes, _, backward_nodes, _ = branch_data
        overlap_vertices = sorted((set(forward_nodes) & set(backward_nodes)) - {int(profile.via_vertex)})
        record: dict[str, Any] = {
            "pair_type": pair_type,
            "via_vertex": int(profile.via_vertex),
            "profile_metrics": profile.metrics.as_dict(),
            "original_repeated_vertex_count": repeated,
            "overlap_vertex_count": len(overlap_vertices),
            "overlap_vertices_sample": overlap_vertices[:20],
        }
        if not overlap_vertices:
            non_elementary_records.append(record)
            continue
        overlap_repairable_count += 1
        repaired, repair_info = _repair_non_elementary_candidate(
            prepared,
            original_G,
            query,
            specs,
            pair_type,
            profile,
            branch_data,
            int(repeated or 0),
        )
        record["repair"] = repair_info
        for call in repair_info["calls"]:
            if call["repair_direction"] == "A":
                repair_call_count_a += 1
            else:
                repair_call_count_b += 1
            repair_dijkstra_s += float(call["elapsed_s"])
        for repaired_candidate in repaired:
            if not repaired_candidate.validation.passed:
                repair_validation_failure_count += 1
                continue
            validated_repair_count += 1
            if not repaired_candidate.elementary:
                repaired_non_elementary_count += 1
                continue
            if not query.is_feasible(repaired_candidate.metrics):
                repaired_exact_box_rejected_count += 1
                continue
            repaired_candidates.append(repaired_candidate)
            if first_repaired_feasible is None:
                first_repaired_feasible = repaired_candidate
                time_to_first_repaired_s = time.perf_counter() - fallback_start
        non_elementary_records.append(record)

    identify_s = time.perf_counter() - identify_start
    fallback_total_s = time.perf_counter() - fallback_start
    best = (
        sorted(repaired_candidates, key=lambda c: (c.box_score, c.metrics.length, c.original_pair_type, c.via_vertex, c.repair_direction))[0]
        if repaired_candidates
        else None
    )
    first_or_best = first_repaired_feasible or best
    return {
        "profile_feasible_by_pair": {
            pair_type: len(profiles_by_pair[pair_type]) for pair_type in PAIR_TYPES
        },
        "nearest_profile": _best_nearest_from_maps(query, same_pair_runs, mixed_nearest),
        "non_elementary_candidates": non_elementary_count,
        "overlap_repairable_candidates": overlap_repairable_count,
        "branch_validation_failures": branch_validation_failures,
        "reconstructed_for_repair": reconstructed_for_repair,
        "repair_A_dijkstra_calls": repair_call_count_a,
        "repair_B_dijkstra_calls": repair_call_count_b,
        "repair_dijkstra_s": repair_dijkstra_s,
        "mixed_profile_scan_s": sum(mixed_scan_times.values()),
        "same_profile_scan_s_reused": sum(
            same_pair_runs[pair_type].profile_scan_s for pair_type in SAME_PAIR_TYPES
        ),
        "identify_non_elementary_s": identify_s,
        "fallback_total_s": fallback_total_s,
        "time_to_first_repaired_feasible_s": time_to_first_repaired_s,
        "validated_repair_count": validated_repair_count,
        "repaired_exact_feasible_count": len(repaired_candidates),
        "repaired_non_elementary_count": repaired_non_elementary_count,
        "repair_validation_failure_count": repair_validation_failure_count,
        "repaired_exact_box_rejected_count": repaired_exact_box_rejected_count,
        "solved_by_repair": first_or_best is not None,
        "first_repaired_feasible": None
        if first_repaired_feasible is None
        else first_repaired_feasible.as_dict(),
        "best_repaired_feasible": None if best is None else best.as_dict(),
        "non_elementary_records_sample": non_elementary_records[:25],
    }


def _row_for_item(
    benchmark_label: str,
    item: bench.BenchmarkItem,
    normal_first: mixed.ExactCandidate | None,
    normal_timing: dict[str, Any],
    same_pair_runs: dict[str, mixed.PairRun],
    repair: dict[str, Any] | None,
) -> dict[str, Any]:
    repaired = None if repair is None else repair.get("first_repaired_feasible") or repair.get("best_repaired_feasible")
    solved_by_repair = bool(repair and repair["solved_by_repair"])
    metrics = None if repaired is None else repaired["metrics"]
    row = {
        "benchmark": benchmark_label,
        "query_id": item.query.name,
        "source": item.query.source,
        "target": item.query.target,
        "normal_union2_feasible": normal_first is not None,
        "lazy_repair_feasible": normal_first is not None or solved_by_repair,
        "profile_feasible_P_P": len(same_pair_runs["P,P"].profile_candidates),
        "profile_feasible_S_S": len(same_pair_runs["S,S"].profile_candidates),
        "normal_union2_cost_s": normal_timing["normal_exhaustive_total_s"],
        "normal_time_to_first_feasible_s": normal_timing["normal_time_to_first_feasible_s"],
        "tree_computation_s": normal_timing["tree_computation_s"],
        "same_scalar_profile_scan_s": normal_timing["same_scalar_profile_scan_s"],
        "same_scalar_exhaustive_reconstruction_s": normal_timing[
            "same_scalar_exhaustive_reconstruction_s"
        ],
    }
    if repair is not None:
        row.update(
            {
                "profile_feasible_P_S": repair["profile_feasible_by_pair"]["P,S"],
                "profile_feasible_S_P": repair["profile_feasible_by_pair"]["S,P"],
                "non_elementary_candidates": repair["non_elementary_candidates"],
                "overlap_repairable_candidates": repair["overlap_repairable_candidates"],
                "repair_A_dijkstra_calls": repair["repair_A_dijkstra_calls"],
                "repair_B_dijkstra_calls": repair["repair_B_dijkstra_calls"],
                "repaired_exact_feasible_count": repair["repaired_exact_feasible_count"],
                "time_to_first_repaired_feasible_s": repair["time_to_first_repaired_feasible_s"],
                "total_repair_time_s": repair["fallback_total_s"],
                "marginal_mixed_profile_scan_s": repair["mixed_profile_scan_s"],
                "marginal_repair_dijkstra_s": repair["repair_dijkstra_s"],
                "total_fallback_cost_s": repair["fallback_total_s"],
                "solved_original_pair_type": None if repaired is None else repaired["original_pair_type"],
                "solved_via_vertex": None if repaired is None else repaired["via_vertex"],
                "solved_repair_direction": None if repaired is None else repaired["repair_direction"],
                "solved_L": None if metrics is None else metrics["length"],
                "solved_H": None if metrics is None else metrics["elevation"],
                "solved_avg_pop": None if metrics is None else metrics["avg_pop"],
                "solved_avg_width": None if metrics is None else metrics["avg_width"],
                "solved_road_changes": None if metrics is None else metrics["road_changes"],
                "solved_repeated_vertices": None if repaired is None else repaired["repeated_vertex_count"],
                "solved_validation_passed": None
                if repaired is None
                else repaired["validation"]["passed"],
                "nearest_profile": repair["nearest_profile"],
            }
        )
    return row


def _load_target_items(
    inputs: base.StaticInputs,
    benchmark_specs: Sequence[tuple[str, str, set[str]]],
) -> tuple[dict[str, list[bench.BenchmarkItem]], dict[str, int]]:
    out: dict[str, list[bench.BenchmarkItem]] = {}
    totals: dict[str, int] = {}
    for label, path, target_ids in benchmark_specs:
        items, _ = bench._load_items_from_boxes_json(inputs, path)
        totals[label] = len(items)
        by_name = {item.query.name: item for item in items}
        missing = sorted(target_ids - set(by_name))
        if missing:
            raise ValueError(f"{label} missing target boxes: {missing}")
        out[label] = [by_name[name] for name in sorted(target_ids)]
    return out, totals


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


def _build_summary(
    rows: Sequence[dict[str, Any]],
    benchmark_totals: dict[str, int],
) -> dict[str, Any]:
    normal_failures = {
        "holdout": len(HOLDOUT_TARGET_IDS),
        "development": len(DEVELOPMENT_TARGET_IDS),
    }
    success: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    for label, total in benchmark_totals.items():
        subset = [row for row in rows if row["benchmark"] == label]
        repaired = sum(
            1
            for row in subset
            if (not row["normal_union2_feasible"]) and row["lazy_repair_feasible"]
        )
        normal_solved = total - normal_failures[label]
        lazy_solved = normal_solved + repaired
        success[label] = {
            VIA_UNION_2: {
                "solved": normal_solved,
                "attempted": total,
                "percent": 100.0 * normal_solved / total,
            },
            LAZY_VIA_REPAIR: {
                "solved": lazy_solved,
                "attempted": total,
                "percent": 100.0 * lazy_solved / total,
                "additional_solved_on_failures": repaired,
            },
        }
        timing[label] = {
            "normal_union2_cost_s": _stats(
                [float(row["normal_union2_cost_s"]) for row in subset]
            ),
            "marginal_mixed_profile_scan_s": _stats(
                [
                    float(row["marginal_mixed_profile_scan_s"])
                    for row in subset
                    if row.get("marginal_mixed_profile_scan_s") not in (None, "")
                ]
            ),
            "marginal_repair_dijkstra_s": _stats(
                [
                    float(row["marginal_repair_dijkstra_s"])
                    for row in subset
                    if row.get("marginal_repair_dijkstra_s") not in (None, "")
                ]
            ),
            "total_fallback_cost_s": _stats(
                [
                    float(row["total_fallback_cost_s"])
                    for row in subset
                    if row.get("total_fallback_cost_s") not in (None, "")
                ]
            ),
            "time_to_first_repaired_feasible_s": _stats(
                [
                    float(row["time_to_first_repaired_feasible_s"])
                    for row in subset
                    if row.get("time_to_first_repaired_feasible_s") not in (None, "")
                ]
            ),
        }
    return {"success": success, "timing": timing}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lazy overlap repair for failed scalar via boxes."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--holdout-boxes-json", default=DEFAULT_HOLDOUT_BOXES_JSON)
    parser.add_argument("--development-boxes-json", default=DEFAULT_DEVELOPMENT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--include-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = bench._compute_global_metric_constants(inputs)
    prepared = mixed._full_prepared(inputs, global_constants)
    target_specs = [
        ("holdout", args.holdout_boxes_json, set(HOLDOUT_TARGET_IDS)),
        ("development", args.development_boxes_json, set(DEVELOPMENT_TARGET_IDS)),
    ]
    benchmark_items, benchmark_totals = _load_target_items(inputs, target_specs)

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for label, items in benchmark_items.items():
        for index, item in enumerate(items, start=1):
            print(f"[{label} {index}/{len(items)}] {item.query.name}", flush=True)
            specs = _scalar_key_spec(item.query)
            trees, tree_total_s = _build_trees(prepared, item.query, specs)
            same_pair_runs, normal_first, normal_timing = _run_normal_union2(
                prepared,
                inputs.G,
                item,
                trees,
                tree_total_s,
            )
            repair = None
            if normal_first is None:
                repair = _run_repair_fallback(
                    prepared,
                    inputs.G,
                    item,
                    specs,
                    trees,
                    same_pair_runs,
                )
            row = _row_for_item(
                label,
                item,
                normal_first,
                normal_timing,
                same_pair_runs,
                repair,
            )
            rows.append(row)
            artifacts.append(
                {
                    "benchmark": label,
                    "query_id": item.query.name,
                    "query": item.query.as_dict(),
                    "normal": {
                        "feasible": normal_first is not None,
                        "first_hit": None if normal_first is None else normal_first.as_dict(),
                        "timing": normal_timing,
                        "profile_feasible_by_pair": {
                            pair_type: len(same_pair_runs[pair_type].profile_candidates)
                            for pair_type in SAME_PAIR_TYPES
                        },
                        "non_elementary_by_pair": {
                            pair_type: same_pair_runs[pair_type].rejected_non_elementary
                            for pair_type in SAME_PAIR_TYPES
                        },
                    },
                    "repair": repair,
                }
            )

    summary = _build_summary(rows, benchmark_totals)
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_mode": "full",
            "holdout_boxes_json": args.holdout_boxes_json,
            "development_boxes_json": args.development_boxes_json,
            "frozen_boxes_policy": "boxes are loaded from existing JSON files; thresholds and witnesses are not modified",
            "evaluated_scalars": {"P": mixed.P_NAME, "S": mixed.S_NAME},
            "normal_fast_path": VIA_UNION_2,
            "fallback": (
                "only failed normal VIA-UNION-2 boxes receive mixed profile scan "
                "and exhaustive branch-overlap repair"
            ),
            "rho_H_global": rho_info,
        },
        "rows": rows,
        "artifacts": artifacts,
        "summary": summary,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)

    print("\nLazy repair summary", flush=True)
    for label, success in summary["success"].items():
        normal = success[VIA_UNION_2]
        lazy = success[LAZY_VIA_REPAIR]
        print(
            f"  {label}: {VIA_UNION_2} {normal['solved']}/{normal['attempted']} "
            f"({normal['percent']:.1f}%), {LAZY_VIA_REPAIR} "
            f"{lazy['solved']}/{lazy['attempted']} ({lazy['percent']:.1f}%)",
            flush=True,
        )
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
