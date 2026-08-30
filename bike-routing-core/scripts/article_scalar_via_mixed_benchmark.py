from __future__ import annotations

import argparse
import csv
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
from brcore.graph.compact import CompactDiGraph


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_mixed_results.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_mixed_results.csv"
DEFAULT_HOLDOUT_BOXES_JSON = "tmp_scalar_via_holdout_boxes.json"
DEFAULT_DEVELOPMENT_BOXES_JSON = "tmp_scalar_via_benchmark_boxes.json"

P_NAME = bench.SCALAR_REFERENCE
S_NAME = bench.SCALAR_SLOPE
PAIR_TYPES = ("P,P", "P,S", "S,P", "S,S")
SAME_PAIR_TYPES = ("P,P", "S,S")
MIXED_PAIR_TYPES = ("P,S", "S,P")
VIA_UNION_2 = bench.UNION_2_NAME
MIXED_VIA_2 = "MIXED-VIA-2"

HOLDOUT_FAILURE_QUERY_IDS = (
    "holdout_anchor_south_north_28_quality_conflict",
    "holdout_anchor_east_west_09_multi_tight",
    "holdout_anchor_south_north_09_multi_tight",
)


@dataclass(frozen=True)
class ExactCandidate:
    pair_type: str
    via_vertex: int
    metrics: base.RouteMetrics
    profile_metrics: base.RouteMetrics
    box_score: float
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
            "pair_type": self.pair_type,
            "via_vertex": self.via_vertex,
            "metrics": self.metrics.as_dict(),
            "profile_metrics": self.profile_metrics.as_dict(),
            "box_score": self.box_score,
            "path_nodes": len(self.path_nodes),
            "edges": len(self.edge_ids),
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
class PairRun:
    pair_type: str
    profile_candidates: list[via.ProfileCandidate]
    exact_candidates: list[ExactCandidate]
    first_hit: ExactCandidate | None
    nearest_profile: dict[str, Any] | None
    nearest_exact: dict[str, Any] | None
    via_vertices_scanned: int
    profile_scan_s: float
    first_hit_reconstruction_s: float
    exhaustive_reconstruction_s: float
    reconstructed_first_hit: int
    reconstructed_exhaustive: int
    rejected_non_elementary: int
    rejected_validation: int
    rejected_exact_box: int


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


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


def _metric_fields(prefix: str, metrics: base.RouteMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {
            f"{prefix}L": None,
            f"{prefix}H": None,
            f"{prefix}avg_pop": None,
            f"{prefix}avg_width": None,
            f"{prefix}road_changes": None,
        }
    return {
        f"{prefix}L": metrics.length,
        f"{prefix}H": metrics.elevation,
        f"{prefix}avg_pop": metrics.avg_popularity,
        f"{prefix}avg_width": metrics.avg_width,
        f"{prefix}road_changes": metrics.road_changes,
    }


def _full_prepared(
    inputs: base.StaticInputs,
    global_constants: base.MetricConstants,
) -> bench.PreparedGraph:
    full_mask = np.ones(inputs.G.n_edges, dtype=bool)
    context = base.GraphContext(
        mode="full",
        edge_mask=full_mask,
        node_count=inputs.G.n_nodes,
        edge_count=inputs.G.n_edges,
        constants=global_constants,
        metadata={"rho_H_policy": "global_full_graph_reused_across_modes"},
    )
    return bench.PreparedGraph(
        mode="full",
        G=inputs.G,
        context=context,
        edge_id_to_original=np.arange(inputs.G.n_edges, dtype=np.int32),
        graph_prep_s=0.0,
        corridor_construction_s=0.0,
        compaction_s=0.0,
        metadata={"graph_storage": "native_full_csr"},
    )


def _specs(query: base.QueryBox) -> dict[str, base.ScalarizationSpec]:
    p_spec = base._reference_spec(query)
    s_spec = bench._slope_spec(query)
    for spec in (p_spec, s_spec):
        bench._effective_refs(query, spec)
    return {"P": p_spec, "S": s_spec}


def _tree_key(pair_type: str) -> tuple[str, str]:
    left, right = pair_type.split(",")
    return left, right


def _profile_sort_key(
    query: base.QueryBox,
    pair_type: str,
    profile: via.ProfileCandidate,
) -> tuple[float, float, float, str, int]:
    return (
        via._box_center_score(query, profile.metrics),
        query.normalized_violation_score(profile.metrics),
        profile.metrics.length,
        pair_type,
        profile.via_vertex,
    )


def _candidate_sort_key(candidate: ExactCandidate) -> tuple[float, float, str, int]:
    return (
        candidate.box_score,
        candidate.metrics.length,
        candidate.pair_type,
        candidate.via_vertex,
    )


def _scan_pair_profiles(
    query: base.QueryBox,
    pair_type: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
) -> tuple[list[via.ProfileCandidate], int, dict[str, Any] | None, float]:
    start = time.perf_counter()
    profiles: list[via.ProfileCandidate] = []
    via_vertices_scanned = 0
    nearest: dict[str, Any] | None = None
    nearest_score: float | None = None
    for via_vertex in range(len(forward.dist)):
        metrics = via._combined_profile(query, forward, backward, via_vertex)
        if metrics is None:
            continue
        via_vertices_scanned += 1
        score = query.normalized_violation_score(metrics)
        if nearest_score is None or score < nearest_score:
            nearest = {
                "pair_type": pair_type,
                "via_vertex": int(via_vertex),
                "normalized_violation_score": score,
                "violations": query.violations(metrics),
                "metrics": metrics.as_dict(),
            }
            nearest_score = score
        if query.is_feasible(metrics):
            profiles.append(via.ProfileCandidate(int(via_vertex), metrics))
    return profiles, via_vertices_scanned, nearest, time.perf_counter() - start


def _reconstruct_candidate(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_type: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profile: via.ProfileCandidate,
) -> tuple[ExactCandidate | None, str | None, base.RouteValidation | None, int | None]:
    via_vertex = int(profile.via_vertex)
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
        return None, "branch_not_reconstructable", None, None

    forward_nodes, forward_edges = forward_branch
    backward_nodes, backward_edges = backward_branch
    if not forward_nodes or not backward_nodes or forward_nodes[-1] != via_vertex:
        return None, "forward_branch_bad_endpoint", None, None
    if backward_nodes[0] != via_vertex:
        return None, "backward_branch_bad_endpoint", None, None

    path_nodes = tuple(forward_nodes + backward_nodes[1:])
    run_edges = tuple(forward_edges + backward_edges)
    original_edges = tuple(
        int(prepared.edge_id_to_original[int(edge_id)]) for edge_id in run_edges
    )
    metrics = base._metrics_from_edge_ids(original_G, original_edges)
    result = via._make_path_result(0.0, path_nodes, original_edges, metrics)
    validation = base._validate_path(original_G, result)
    repeated = via._repeated_vertex_count(path_nodes)
    if not validation.passed:
        return None, "validation_failed", validation, repeated
    candidate = ExactCandidate(
        pair_type=pair_type,
        via_vertex=via_vertex,
        metrics=metrics,
        profile_metrics=profile.metrics,
        box_score=via._box_center_score(query, metrics),
        path_nodes=path_nodes,
        edge_ids=original_edges,
        validation=validation,
        repeated_vertex_count=repeated,
        profile_exact_deltas=via._profile_exact_deltas(profile.metrics, metrics),
    )
    return candidate, None, validation, repeated


def _reconstruct_first_hit(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_type: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profiles: Sequence[via.ProfileCandidate],
) -> tuple[ExactCandidate | None, int, int, int, int, float]:
    start = time.perf_counter()
    reconstructed = 0
    rejected_non_elementary = 0
    rejected_validation = 0
    rejected_exact_box = 0
    for profile in sorted(profiles, key=lambda p: _profile_sort_key(query, pair_type, p)):
        reconstructed += 1
        candidate, reason, _, repeated = _reconstruct_candidate(
            prepared,
            original_G,
            query,
            pair_type,
            forward,
            backward,
            profile,
        )
        if candidate is None:
            rejected_validation += 1
            continue
        if not candidate.elementary:
            rejected_non_elementary += 1
            continue
        if not query.is_feasible(candidate.metrics):
            rejected_exact_box += 1
            continue
        return (
            candidate,
            reconstructed,
            rejected_non_elementary,
            rejected_validation,
            rejected_exact_box,
            time.perf_counter() - start,
        )
    return (
        None,
        reconstructed,
        rejected_non_elementary,
        rejected_validation,
        rejected_exact_box,
        time.perf_counter() - start,
    )


def _reconstruct_exhaustive(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_type: str,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profiles: Sequence[via.ProfileCandidate],
) -> tuple[list[ExactCandidate], dict[str, Any] | None, int, int, int, int, float]:
    start = time.perf_counter()
    exact: list[ExactCandidate] = []
    nearest_exact: dict[str, Any] | None = None
    nearest_score: float | None = None
    reconstructed = 0
    rejected_non_elementary = 0
    rejected_validation = 0
    rejected_exact_box = 0
    for profile in profiles:
        reconstructed += 1
        candidate, reason, _, _ = _reconstruct_candidate(
            prepared,
            original_G,
            query,
            pair_type,
            forward,
            backward,
            profile,
        )
        if candidate is None:
            rejected_validation += 1
            continue
        if not candidate.elementary:
            rejected_non_elementary += 1
            continue
        score = query.normalized_violation_score(candidate.metrics)
        if nearest_score is None or score < nearest_score:
            nearest_exact = {
                "pair_type": pair_type,
                "via_vertex": candidate.via_vertex,
                "normalized_violation_score": score,
                "violations": query.violations(candidate.metrics),
                "metrics": candidate.metrics.as_dict(),
                "profile_metrics": candidate.profile_metrics.as_dict(),
                "profile_exact_deltas": candidate.profile_exact_deltas,
            }
            nearest_score = score
        if not query.is_feasible(candidate.metrics):
            rejected_exact_box += 1
            continue
        exact.append(candidate)
    return (
        exact,
        nearest_exact,
        reconstructed,
        rejected_non_elementary,
        rejected_validation,
        rejected_exact_box,
        time.perf_counter() - start,
    )


def _run_pair_type(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_type: str,
    trees: dict[str, dict[str, via.TreeResult]],
) -> PairRun:
    forward_key, backward_key = _tree_key(pair_type)
    forward = trees[forward_key]["forward"]
    backward = trees[backward_key]["backward"]
    profiles, scanned, nearest_profile, scan_s = _scan_pair_profiles(
        query,
        pair_type,
        forward,
        backward,
    )
    (
        first_hit,
        reconstructed_first,
        first_non_elementary,
        first_validation,
        first_exact_box,
        first_s,
    ) = _reconstruct_first_hit(
        prepared,
        original_G,
        query,
        pair_type,
        forward,
        backward,
        profiles,
    )
    (
        exact,
        nearest_exact,
        reconstructed_exhaustive,
        non_elementary,
        validation,
        exact_box,
        exhaustive_s,
    ) = _reconstruct_exhaustive(
        prepared,
        original_G,
        query,
        pair_type,
        forward,
        backward,
        profiles,
    )
    return PairRun(
        pair_type=pair_type,
        profile_candidates=profiles,
        exact_candidates=exact,
        first_hit=first_hit,
        nearest_profile=nearest_profile,
        nearest_exact=nearest_exact,
        via_vertices_scanned=scanned,
        profile_scan_s=scan_s,
        first_hit_reconstruction_s=first_s,
        exhaustive_reconstruction_s=exhaustive_s,
        reconstructed_first_hit=reconstructed_first,
        reconstructed_exhaustive=reconstructed_exhaustive,
        rejected_non_elementary=non_elementary,
        rejected_validation=validation,
        rejected_exact_box=exact_box,
    )


def _first_hit_union(
    prepared: bench.PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    pair_runs: dict[str, PairRun],
    pair_types: Sequence[str],
    trees: dict[str, dict[str, via.TreeResult]],
) -> tuple[ExactCandidate | None, int, int, int, int, float]:
    start = time.perf_counter()
    ordered: list[tuple[str, via.ProfileCandidate]] = []
    for pair_type in pair_types:
        ordered.extend((pair_type, profile) for profile in pair_runs[pair_type].profile_candidates)
    ordered.sort(key=lambda item: _profile_sort_key(query, item[0], item[1]))
    reconstructed = 0
    rejected_non_elementary = 0
    rejected_validation = 0
    rejected_exact_box = 0
    for pair_type, profile in ordered:
        forward_key, backward_key = _tree_key(pair_type)
        candidate, _, _, _ = _reconstruct_candidate(
            prepared,
            original_G,
            query,
            pair_type,
            trees[forward_key]["forward"],
            trees[backward_key]["backward"],
            profile,
        )
        reconstructed += 1
        if candidate is None:
            rejected_validation += 1
            continue
        if not candidate.elementary:
            rejected_non_elementary += 1
            continue
        if not query.is_feasible(candidate.metrics):
            rejected_exact_box += 1
            continue
        return (
            candidate,
            reconstructed,
            rejected_non_elementary,
            rejected_validation,
            rejected_exact_box,
            time.perf_counter() - start,
        )
    return (
        None,
        reconstructed,
        rejected_non_elementary,
        rejected_validation,
        rejected_exact_box,
        time.perf_counter() - start,
    )


def _best_nearest(
    query: base.QueryBox,
    pair_runs: dict[str, PairRun],
    pair_types: Sequence[str],
) -> dict[str, Any] | None:
    options: list[dict[str, Any]] = []
    for pair_type in pair_types:
        run = pair_runs[pair_type]
        if run.nearest_exact is not None:
            options.append(run.nearest_exact)
        elif run.nearest_profile is not None:
            options.append(run.nearest_profile)
    if not options:
        return None
    return min(
        options,
        key=lambda item: (
            float(item.get("normalized_violation_score", float("inf"))),
            str(item.get("pair_type", "")),
            int(item.get("via_vertex", 2**31 - 1)),
        ),
    )


def _unique_exact_candidates(
    pair_runs: dict[str, PairRun],
    pair_types: Sequence[str],
) -> list[ExactCandidate]:
    seen: set[tuple[int, ...]] = set()
    out: list[ExactCandidate] = []
    for pair_type in pair_types:
        for candidate in pair_runs[pair_type].exact_candidates:
            key = tuple(candidate.edge_ids)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def _row_base(
    benchmark_label: str,
    item: bench.BenchmarkItem,
    row_type: str,
    pair_type_or_union: str,
) -> dict[str, Any]:
    q = item.query
    return {
        "benchmark": benchmark_label,
        "query_id": q.name,
        "pair_id": item.pair_id,
        "category": item.category,
        "tightness": item.tightness,
        "tags": "|".join(item.tags),
        "source": q.source,
        "target": q.target,
        "row_type": row_type,
        "pair_type": pair_type_or_union,
        "graph_mode": "full",
        "Lmin": q.Lmin,
        "Lmax": q.Lmax,
        "Hmin": q.Hmin,
        "Hmax": q.Hmax,
        "Pmin": q.Pmin,
        "Wmax": q.Wmax,
        "witness_route_id": item.witness.route_id,
        "witness_generator": item.witness.generator,
        "witness_scalar_name": item.witness.scalar_name,
        **_metric_fields("witness_", item.witness.metrics),
        "shortest_violation_score": item.shortest_violation_score,
        "shortest_violated_constraints": "|".join(item.shortest_violated_constraints),
    }


def _pair_run_row(
    benchmark_label: str,
    item: bench.BenchmarkItem,
    pair_run: PairRun,
    tree_total_s: float,
) -> dict[str, Any]:
    first = pair_run.first_hit
    nearest = pair_run.nearest_exact or pair_run.nearest_profile
    feasible = bool(pair_run.exact_candidates)
    route = first or (sorted(pair_run.exact_candidates, key=_candidate_sort_key)[0] if feasible else None)
    score, violations, violated = bench._violation_details(
        item.query,
        None if route is None else route.metrics,
    )
    if not feasible and nearest is not None:
        score = nearest.get("normalized_violation_score")
        violations = nearest.get("violations")
        if isinstance(violations, dict):
            violated = tuple(k for k, v in violations.items() if v > 1e-6)
    row = _row_base(benchmark_label, item, "pair_type", pair_run.pair_type)
    row.update(
        {
            "route_found": route is not None,
            "feasible": feasible,
            **_metric_fields("", None if route is None else route.metrics),
            "via_vertex": None if route is None else route.via_vertex,
            "orientation": None if route is None else route.pair_type,
            "road_changes": None if route is None else route.metrics.road_changes,
            "elementary": None if route is None else route.elementary,
            "repeated_vertex_count": None if route is None else route.repeated_vertex_count,
            "validation_passed": None if route is None else route.validation.passed,
            "profile_feasible_count": len(pair_run.profile_candidates),
            "exact_feasible_count": len(pair_run.exact_candidates),
            "non_elementary_count": pair_run.rejected_non_elementary,
            "rejected_validation_count": pair_run.rejected_validation,
            "rejected_exact_box_count": pair_run.rejected_exact_box,
            "via_vertices_scanned": pair_run.via_vertices_scanned,
            "tree_computation_s": tree_total_s,
            "profile_scan_s": pair_run.profile_scan_s,
            "first_hit_reconstruction_s": pair_run.first_hit_reconstruction_s,
            "exhaustive_reconstruction_s": pair_run.exhaustive_reconstruction_s,
            "time_to_first_feasible": None
            if first is None
            else tree_total_s + pair_run.profile_scan_s + pair_run.first_hit_reconstruction_s,
            "exhaustive_total_s": tree_total_s
            + pair_run.profile_scan_s
            + pair_run.exhaustive_reconstruction_s,
            "reconstructed_before_first_hit": None
            if first is None
            else pair_run.reconstructed_first_hit,
            "reconstructed_exhaustive": pair_run.reconstructed_exhaustive,
            "normalized_violation_score": 0.0 if feasible else score,
            "violations": {} if feasible else violations,
            "violated_constraints": "" if feasible else "|".join(violated),
            "nearest_candidate_or_profile": nearest,
        }
    )
    return row


def _union_row(
    benchmark_label: str,
    item: bench.BenchmarkItem,
    union_name: str,
    pair_types: Sequence[str],
    pair_runs: dict[str, PairRun],
    first_hit: ExactCandidate | None,
    first_hit_counts: tuple[int, int, int, int, float],
    timing: dict[str, float],
) -> dict[str, Any]:
    unique_exact = _unique_exact_candidates(pair_runs, pair_types)
    feasible = bool(unique_exact)
    route = first_hit or (sorted(unique_exact, key=_candidate_sort_key)[0] if unique_exact else None)
    nearest = _best_nearest(item.query, pair_runs, pair_types)
    score, violations, violated = bench._violation_details(
        item.query,
        None if route is None else route.metrics,
    )
    if not feasible and nearest is not None:
        score = nearest.get("normalized_violation_score")
        violations = nearest.get("violations")
        if isinstance(violations, dict):
            violated = tuple(k for k, v in violations.items() if v > 1e-6)
    reconstructed, non_elementary_first, validation_first, exact_box_first, first_recon_s = first_hit_counts
    row = _row_base(benchmark_label, item, "union", union_name)
    row.update(
        {
            "route_found": route is not None,
            "feasible": feasible,
            **_metric_fields("", None if route is None else route.metrics),
            "via_vertex": None if route is None else route.via_vertex,
            "orientation": None if route is None else route.pair_type,
            "road_changes": None if route is None else route.metrics.road_changes,
            "elementary": None if route is None else route.elementary,
            "repeated_vertex_count": None if route is None else route.repeated_vertex_count,
            "validation_passed": None if route is None else route.validation.passed,
            "member_pair_types": "|".join(pair_types),
            "profile_feasible_count": sum(len(pair_runs[p].profile_candidates) for p in pair_types),
            "exact_feasible_count": len(unique_exact),
            "non_elementary_count": sum(pair_runs[p].rejected_non_elementary for p in pair_types),
            "rejected_validation_count": sum(pair_runs[p].rejected_validation for p in pair_types),
            "rejected_exact_box_count": sum(pair_runs[p].rejected_exact_box for p in pair_types),
            "via_vertices_scanned": sum(pair_runs[p].via_vertices_scanned for p in pair_types),
            "tree_computation_s": timing["tree_computation_s"],
            "same_scalar_profile_scan_s": timing["same_scalar_profile_scan_s"],
            "additional_mixed_profile_scan_s": timing["additional_mixed_profile_scan_s"],
            "same_scalar_reconstruction_s": timing["same_scalar_reconstruction_s"],
            "mixed_reconstruction_s": timing["mixed_reconstruction_s"],
            "profile_scan_s": sum(pair_runs[p].profile_scan_s for p in pair_types),
            "first_hit_reconstruction_s": first_recon_s,
            "exhaustive_reconstruction_s": sum(
                pair_runs[p].exhaustive_reconstruction_s for p in pair_types
            ),
            "time_to_first_feasible": None
            if first_hit is None
            else timing["tree_computation_s"]
            + sum(pair_runs[p].profile_scan_s for p in pair_types)
            + first_recon_s,
            "exhaustive_total_s": timing["tree_computation_s"]
            + sum(pair_runs[p].profile_scan_s for p in pair_types)
            + sum(pair_runs[p].exhaustive_reconstruction_s for p in pair_types),
            "reconstructed_before_first_hit": None if first_hit is None else reconstructed,
            "reconstructed_exhaustive": sum(pair_runs[p].reconstructed_exhaustive for p in pair_types),
            "normalized_violation_score": 0.0 if feasible else score,
            "violations": {} if feasible else violations,
            "violated_constraints": "" if feasible else "|".join(violated),
            "nearest_candidate_or_profile": nearest,
            "first_hit_non_elementary_rejections": non_elementary_first,
            "first_hit_validation_rejections": validation_first,
            "first_hit_exact_box_rejections": exact_box_first,
        }
    )
    return row


def _run_one_query(
    inputs: base.StaticInputs,
    prepared: bench.PreparedGraph,
    benchmark_label: str,
    item: bench.BenchmarkItem,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = _specs(item.query)
    reverse_adj = via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)
    trees: dict[str, dict[str, via.TreeResult]] = {}
    tree_start = time.perf_counter()
    for key in ("P", "S"):
        spec = specs[key]
        forward = via._run_scalar_tree(
            prepared.G,
            item.query,
            prepared.context,
            spec,
            reverse=False,
        )
        backward = via._run_scalar_tree(
            prepared.G,
            item.query,
            prepared.context,
            spec,
            reverse=True,
            reverse_adj=reverse_adj,
        )
        trees[key] = {"forward": forward, "backward": backward}
    tree_total_s = time.perf_counter() - tree_start

    pair_runs = {
        pair_type: _run_pair_type(prepared, inputs.G, item.query, pair_type, trees)
        for pair_type in PAIR_TYPES
    }
    via_union_first, *via_union_counts = _first_hit_union(
        prepared,
        inputs.G,
        item.query,
        pair_runs,
        SAME_PAIR_TYPES,
        trees,
    )
    mixed_union_first, *mixed_union_counts = _first_hit_union(
        prepared,
        inputs.G,
        item.query,
        pair_runs,
        PAIR_TYPES,
        trees,
    )
    timing = {
        "tree_computation_s": tree_total_s,
        "same_scalar_profile_scan_s": sum(pair_runs[p].profile_scan_s for p in SAME_PAIR_TYPES),
        "additional_mixed_profile_scan_s": sum(pair_runs[p].profile_scan_s for p in MIXED_PAIR_TYPES),
        "same_scalar_reconstruction_s": sum(
            pair_runs[p].exhaustive_reconstruction_s for p in SAME_PAIR_TYPES
        ),
        "mixed_reconstruction_s": sum(
            pair_runs[p].exhaustive_reconstruction_s for p in MIXED_PAIR_TYPES
        ),
    }
    rows = [
        _pair_run_row(benchmark_label, item, pair_runs[pair_type], tree_total_s)
        for pair_type in PAIR_TYPES
    ]
    rows.append(
        _union_row(
            benchmark_label,
            item,
            VIA_UNION_2,
            SAME_PAIR_TYPES,
            pair_runs,
            via_union_first,
            tuple(via_union_counts),  # type: ignore[arg-type]
            timing,
        )
    )
    rows.append(
        _union_row(
            benchmark_label,
            item,
            MIXED_VIA_2,
            PAIR_TYPES,
            pair_runs,
            mixed_union_first,
            tuple(mixed_union_counts),  # type: ignore[arg-type]
            timing,
        )
    )
    artifact = {
        "benchmark": benchmark_label,
        "query_id": item.query.name,
        "pair_runs": {
            pair_type: {
                "profile_feasible_count": len(run.profile_candidates),
                "exact_feasible_count": len(run.exact_candidates),
                "non_elementary_count": run.rejected_non_elementary,
                "first_hit": None if run.first_hit is None else run.first_hit.as_dict(),
                "best_5_exact": [
                    candidate.as_dict()
                    for candidate in sorted(run.exact_candidates, key=_candidate_sort_key)[:5]
                ],
            }
            for pair_type, run in pair_runs.items()
        },
        "timing": timing,
    }
    return rows, artifact


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[idx]


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _success_by_union(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for benchmark_label in sorted({row["benchmark"] for row in rows}):
        out[benchmark_label] = {}
        for union_name in (VIA_UNION_2, MIXED_VIA_2):
            subset = [
                row
                for row in rows
                if row["benchmark"] == benchmark_label
                and row["row_type"] == "union"
                and row["pair_type"] == union_name
            ]
            solved = sum(1 for row in subset if bool(row["feasible"]))
            out[benchmark_label][union_name] = {
                "solved": solved,
                "attempted": len(subset),
                "percent": 100.0 * solved / len(subset) if subset else None,
            }
    return out


def _counts_by_pair_type(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for benchmark_label in sorted({row["benchmark"] for row in rows}):
        out[benchmark_label] = {}
        for pair_type in PAIR_TYPES:
            subset = [
                row
                for row in rows
                if row["benchmark"] == benchmark_label
                and row["row_type"] == "pair_type"
                and row["pair_type"] == pair_type
            ]
            out[benchmark_label][pair_type] = {
                "profile_feasible_count": sum(int(row["profile_feasible_count"]) for row in subset),
                "exact_feasible_count": sum(int(row["exact_feasible_count"]) for row in subset),
                "non_elementary_count": sum(int(row["non_elementary_count"]) for row in subset),
                "solved_queries": sum(1 for row in subset if bool(row["feasible"])),
            }
    return out


def _timing_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for benchmark_label in sorted({row["benchmark"] for row in rows}):
        out[benchmark_label] = {}
        for row_type, names in (
            ("pair_type", PAIR_TYPES),
            ("union", (VIA_UNION_2, MIXED_VIA_2)),
        ):
            for name in names:
                subset = [
                    row
                    for row in rows
                    if row["benchmark"] == benchmark_label
                    and row["row_type"] == row_type
                    and row["pair_type"] == name
                ]
                out[benchmark_label][name] = {
                    "time_to_first_feasible": _stats(
                        [
                            float(row["time_to_first_feasible"])
                            for row in subset
                            if row.get("time_to_first_feasible") not in (None, "")
                        ]
                    ),
                    "exhaustive_total_s": _stats(
                        [float(row["exhaustive_total_s"]) for row in subset]
                    ),
                    "tree_computation_s": _stats(
                        [float(row["tree_computation_s"]) for row in subset]
                    ),
                    "same_scalar_profile_scan_s": _stats(
                        [
                            float(row.get("same_scalar_profile_scan_s") or 0.0)
                            for row in subset
                            if row_type == "union"
                        ]
                    ),
                    "additional_mixed_profile_scan_s": _stats(
                        [
                            float(row.get("additional_mixed_profile_scan_s") or 0.0)
                            for row in subset
                            if row_type == "union"
                        ]
                    ),
                    "mixed_reconstruction_s": _stats(
                        [
                            float(row.get("mixed_reconstruction_s") or 0.0)
                            for row in subset
                            if row_type == "union"
                        ]
                    ),
                }
    return out


def _new_solved_by_cross(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for benchmark_label in sorted({row["benchmark"] for row in rows}):
        query_ids = sorted(
            {
                row["query_id"]
                for row in rows
                if row["benchmark"] == benchmark_label and row["row_type"] == "union"
            }
        )
        only_ps: list[str] = []
        only_sp: list[str] = []
        both_cross: list[str] = []
        mixed_new: list[str] = []
        for query_id in query_ids:
            def solved_pair(pair_type: str) -> bool:
                return any(
                    row["benchmark"] == benchmark_label
                    and row["query_id"] == query_id
                    and row["row_type"] == "pair_type"
                    and row["pair_type"] == pair_type
                    and bool(row["feasible"])
                    for row in rows
                )

            via_union = any(
                row["benchmark"] == benchmark_label
                and row["query_id"] == query_id
                and row["row_type"] == "union"
                and row["pair_type"] == VIA_UNION_2
                and bool(row["feasible"])
                for row in rows
            )
            mixed_union = any(
                row["benchmark"] == benchmark_label
                and row["query_id"] == query_id
                and row["row_type"] == "union"
                and row["pair_type"] == MIXED_VIA_2
                and bool(row["feasible"])
                for row in rows
            )
            ps = solved_pair("P,S")
            sp = solved_pair("S,P")
            if mixed_union and not via_union:
                mixed_new.append(query_id)
                if ps and not sp:
                    only_ps.append(query_id)
                elif sp and not ps:
                    only_sp.append(query_id)
                elif ps and sp:
                    both_cross.append(query_id)
        out[benchmark_label] = {
            "newly_solved_by_mixed": mixed_new,
            "solved_only_by_P_S": only_ps,
            "solved_only_by_S_P": only_sp,
            "solved_by_both_cross_orientations": both_cross,
        }
    return out


def _classify_mixed_failure(row: dict[str, Any]) -> str:
    if int(row.get("rejected_validation_count") or 0) > 0:
        return "C_VALIDATION_FAILURE"
    profiles = int(row.get("profile_feasible_count") or 0)
    exact = int(row.get("exact_feasible_count") or 0)
    non_elem = int(row.get("non_elementary_count") or 0)
    if profiles == 0:
        return "A_NO_PROFILE_FEASIBLE"
    if exact == 0 and non_elem >= profiles:
        return "B_PROFILE_FEASIBLE_BUT_NON_ELEMENTARY"
    return "D_OTHER"


def _holdout_failure_diagnostics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for query_id in HOLDOUT_FAILURE_QUERY_IDS:
        row = next(
            (
                row
                for row in rows
                if row["benchmark"] == "holdout"
                and row["query_id"] == query_id
                and row["row_type"] == "union"
                and row["pair_type"] == MIXED_VIA_2
            ),
            None,
        )
        if row is None:
            out.append({"query_id": query_id, "status": "not_present"})
            continue
        if row["feasible"]:
            status = "A_FINDS_EXACT_ELEMENTARY_FEASIBLE_ROUTE"
        else:
            status = _classify_mixed_failure(row)
        out.append(
            {
                "query_id": query_id,
                "status": status,
                "orientation": row.get("orientation"),
                "via_vertex": row.get("via_vertex"),
                "metrics": {
                    "L": row.get("L"),
                    "H": row.get("H"),
                    "avg_pop": row.get("avg_pop"),
                    "avg_width": row.get("avg_width"),
                    "road_changes": row.get("road_changes"),
                },
                "repeated_vertex_count": row.get("repeated_vertex_count"),
                "validation_passed": row.get("validation_passed"),
                "profile_feasible_count": row.get("profile_feasible_count"),
                "exact_feasible_count": row.get("exact_feasible_count"),
                "non_elementary_count": row.get("non_elementary_count"),
                "nearest_candidate_or_profile": row.get("nearest_candidate_or_profile"),
            }
        )
    return out


def _build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "success": _success_by_union(rows),
        "counts_by_pair_type": _counts_by_pair_type(rows),
        "timing": _timing_summary(rows),
        "new_solved_by_cross_orientation": _new_solved_by_cross(rows),
        "holdout_target_failure_diagnostics": _holdout_failure_diagnostics(rows),
    }


def _load_benchmark_items(
    inputs: base.StaticInputs,
    specs: Sequence[str],
) -> dict[str, list[bench.BenchmarkItem]]:
    out: dict[str, list[bench.BenchmarkItem]] = {}
    for raw in specs:
        if ":" not in raw:
            raise ValueError(f"benchmark spec must be label:path, got {raw!r}")
        label, path = raw.split(":", 1)
        items, _ = bench._load_items_from_boxes_json(inputs, path)
        out[label] = items
    return out


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nMixed-via summary", flush=True)
    for label, success in summary["success"].items():
        union = success[VIA_UNION_2]
        mixed = success[MIXED_VIA_2]
        print(
            f"  {label}: {VIA_UNION_2} {union['solved']}/{union['attempted']} "
            f"({union['percent']:.1f}%), {MIXED_VIA_2} "
            f"{mixed['solved']}/{mixed['attempted']} ({mixed['percent']:.1f}%)",
            flush=True,
        )
        new = summary["new_solved_by_cross_orientation"][label]
        print(
            "    new by mixed: "
            f"{', '.join(new['newly_solved_by_mixed']) if new['newly_solved_by_mixed'] else 'none'}",
            flush=True,
        )
    print("  holdout target failures:", flush=True)
    for failure in summary["holdout_target_failure_diagnostics"]:
        print(
            f"    {failure['query_id']}: {failure['status']} "
            f"orientation={failure.get('orientation')}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed forward/backward via composition on frozen boxes."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Benchmark label/path as label:path. May be repeated.",
    )
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
    prepared = _full_prepared(inputs, global_constants)
    benchmark_specs = args.benchmark or [
        f"holdout:{DEFAULT_HOLDOUT_BOXES_JSON}",
        f"development:{DEFAULT_DEVELOPMENT_BOXES_JSON}",
    ]
    benchmarks = _load_benchmark_items(inputs, benchmark_specs)

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for label, items in benchmarks.items():
        for index, item in enumerate(items, start=1):
            print(f"[{label} {index}/{len(items)}] {item.query.name}", flush=True)
            query_rows, artifact = _run_one_query(inputs, prepared, label, item)
            rows.extend(query_rows)
            artifacts.append(artifact)

    summary = _build_summary(rows)
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_path": args.graph_path,
            "graph_mode": "full",
            "benchmarks": benchmark_specs,
            "frozen_boxes_policy": "boxes are loaded from existing JSON files; thresholds and witnesses are not modified",
            "evaluated_tree_scalars": {"P": P_NAME, "S": S_NAME},
            "same_scalar_union": SAME_PAIR_TYPES,
            "mixed_union": PAIR_TYPES,
            "rho_H_global": rho_info,
            "heterogeneous_cost_policy": (
                "mixed orientations do not compare or sum heterogeneous scalar costs; "
                "ranking uses resource box-centered score and deterministic ties"
            ),
        },
        "rows": rows,
        "artifacts": artifacts,
        "summary": summary,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    _print_summary(summary)
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
