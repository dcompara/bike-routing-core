from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_feasibility_experiment as via
from brcore.algo.search_space_reduction import search_space_reduction
from brcore.graph.compact import CompactDiGraph


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_benchmark.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_benchmark.csv"
DEFAULT_BOXES_JSON = "tmp_scalar_via_benchmark_boxes.json"

SCALAR_PHYSICAL = "physical_length"
SCALAR_REFERENCE = "pop_width_reference"
SCALAR_SLOPE = "slope_exp_beta_150_width"
BENCHMARK_SCALAR_NAMES = (SCALAR_PHYSICAL, SCALAR_REFERENCE, SCALAR_SLOPE)

UNION_2_NAME = "VIA-UNION-2"
UNION_3_NAME = "VIA-UNION-3"
RHO_MARGIN_NOTE = "np.nextafter(max full-graph positive-edge gain/length, +inf)"


@dataclass(frozen=True)
class PreparedGraph:
    mode: str
    G: CompactDiGraph
    context: base.GraphContext
    edge_id_to_original: np.ndarray
    graph_prep_s: float
    corridor_construction_s: float
    compaction_s: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WitnessRoute:
    route_id: str
    pair_id: str
    generator: str
    scalar_name: str
    via_vertex: int | None
    path_nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    metrics: base.RouteMetrics
    validation: base.RouteValidation
    elementary: bool
    repeated_vertex_count: int

    def as_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "route_id": self.route_id,
            "pair_id": self.pair_id,
            "generator": self.generator,
            "scalar_name": self.scalar_name,
            "via_vertex": self.via_vertex,
            "metrics": self.metrics.as_dict(),
            "validation": self.validation.as_dict(),
            "elementary": self.elementary,
            "repeated_vertex_count": self.repeated_vertex_count,
            "path_nodes": len(self.path_nodes),
            "edges": len(self.edge_ids),
        }
        if include_paths:
            out["path_node_ids"] = list(self.path_nodes)
            out["csr_edge_ids"] = list(self.edge_ids)
        return out


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    source: int
    target: int
    shortest: WitnessRoute
    selection_note: str


@dataclass(frozen=True)
class BenchmarkItem:
    query: base.QueryBox
    pair_id: str
    category: str
    tightness: str
    tags: tuple[str, ...]
    witness: WitnessRoute
    physical_shortest: WitnessRoute
    shortest_violation_score: float
    shortest_violations: dict[str, float]
    shortest_violated_constraints: tuple[str, ...]
    adversarial: bool
    deliberate_adversarial: bool

    def as_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        return {
            "query": self.query.as_dict(),
            "query_id": self.query.name,
            "pair_id": self.pair_id,
            "category": self.category,
            "tightness": self.tightness,
            "tags": list(self.tags),
            "witness": self.witness.as_dict(include_paths=include_paths),
            "physical_shortest": self.physical_shortest.as_dict(
                include_paths=include_paths
            ),
            "shortest_violation_score": self.shortest_violation_score,
            "shortest_violations": self.shortest_violations,
            "shortest_violated_constraints": list(
                self.shortest_violated_constraints
            ),
            "adversarial": self.adversarial,
            "deliberate_adversarial": self.deliberate_adversarial,
        }


@dataclass(frozen=True)
class ReconstructionAttempt:
    candidate: via.ExactViaCandidate | None
    reason: str | None
    validation: base.RouteValidation | None
    repeated_vertex_count: int | None


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


def _violation_details(
    query: base.QueryBox,
    metrics: base.RouteMetrics | None,
) -> tuple[float | None, dict[str, float] | None, tuple[str, ...]]:
    if metrics is None:
        return None, None, ()
    violations = query.violations(metrics)
    violated = tuple(key for key, value in violations.items() if value > 1e-6)
    return query.normalized_violation_score(metrics), violations, violated


def _physical_spec() -> base.ScalarizationSpec:
    return base.ScalarizationSpec(SCALAR_PHYSICAL, "physical_length", {})


def _slope_spec(query: base.QueryBox) -> base.ScalarizationSpec:
    for spec in base._fixed_portfolio(query):
        if spec.name == SCALAR_SLOPE:
            return spec
    raise RuntimeError(f"{SCALAR_SLOPE} not found in fixed portfolio")


def _benchmark_scalarizations(query: base.QueryBox) -> list[base.ScalarizationSpec]:
    specs = [_physical_spec(), base._reference_spec(query), _slope_spec(query)]
    names = [spec.name for spec in specs]
    if names != list(BENCHMARK_SCALAR_NAMES):
        raise AssertionError(f"unexpected benchmark scalar order {names!r}")
    return specs


def _effective_refs(
    query: base.QueryBox,
    spec: base.ScalarizationSpec,
) -> tuple[float | None, float | None]:
    wref = spec.parameters.get("Wref")
    pref = spec.parameters.get("Pref")
    effective_wref = None if wref is None or isinstance(wref, str) else float(wref)
    effective_pref = None if pref is None or isinstance(pref, str) else float(pref)
    if spec.name in {SCALAR_REFERENCE, SCALAR_SLOPE}:
        if effective_wref is None or abs(effective_wref - query.Wmax) > 1e-9:
            raise AssertionError(
                f"{spec.name} Wref={effective_wref} differs from query.Wmax={query.Wmax}"
            )
        if effective_pref is None or abs(effective_pref - query.Pmin) > 1e-9:
            raise AssertionError(
                f"{spec.name} Pref={effective_pref} differs from query.Pmin={query.Pmin}"
            )
    return effective_wref, effective_pref


def _compute_global_metric_constants(
    inputs: base.StaticInputs,
) -> tuple[base.MetricConstants, dict[str, Any]]:
    G = inputs.G
    full_mask = np.ones(G.n_edges, dtype=bool)
    full_constants = base._metric_constants(G, full_mask)
    lengths = G.w[:, 0].astype(np.float64)
    gains = G.w[:, 1].astype(np.float64)
    positive = np.flatnonzero(lengths > base.EPS)
    if len(positive) == 0:
        max_edge = None
        max_slope = 0.0
    else:
        slopes = gains[positive] / lengths[positive]
        local_idx = int(np.argmax(slopes))
        max_edge = int(positive[local_idx])
        max_slope = float(slopes[local_idx])
    info = {
        "definition": RHO_MARGIN_NOTE,
        "rho_H_global": full_constants.rho_H,
        "max_slope_full": max_slope,
        "max_slope_edge_id": max_edge,
        "max_slope_edge_source": None
        if max_edge is None
        else int(inputs.edge_sources[max_edge]),
        "max_slope_edge_target": None if max_edge is None else int(G.to[max_edge]),
        "max_slope_edge_length": None if max_edge is None else float(G.w[max_edge, 0]),
        "max_slope_edge_gain": None if max_edge is None else float(G.w[max_edge, 1]),
        "p_star_full": full_constants.p_star,
        "zero_length_positive_gain_edges_full": (
            full_constants.zero_length_positive_gain_edges
        ),
    }
    return full_constants, info


def _constants_with_global_rho(
    original: base.MetricConstants,
    global_constants: base.MetricConstants,
) -> base.MetricConstants:
    return base.MetricConstants(
        rho_H=global_constants.rho_H,
        p_star=global_constants.p_star,
        max_slope=global_constants.max_slope,
        zero_length_positive_gain_edges=original.zero_length_positive_gain_edges,
        usable_edges=original.usable_edges,
    )


def _context_with_mode_and_constants(
    context: base.GraphContext,
    *,
    mode: str,
    constants: base.MetricConstants,
    metadata: dict[str, Any] | None = None,
) -> base.GraphContext:
    merged = dict(context.metadata)
    if metadata:
        merged.update(metadata)
    merged["rho_H_policy"] = "global_full_graph_reused_across_modes"
    return base.GraphContext(
        mode=mode,
        edge_mask=context.edge_mask,
        node_count=context.node_count,
        edge_count=context.edge_count,
        constants=constants,
        metadata=merged,
    )


def _mode_metric_constant_snapshot(
    G: CompactDiGraph,
    edge_mask: np.ndarray,
    inputs: base.StaticInputs,
) -> dict[str, Any]:
    constants = base._metric_constants(G, edge_mask)
    usable = np.flatnonzero(edge_mask)
    if len(usable) == 0:
        max_edge = None
    else:
        lengths = G.w[usable, 0].astype(np.float64)
        gains = G.w[usable, 1].astype(np.float64)
        positive = lengths > base.EPS
        if np.any(positive):
            positive_edges = usable[positive]
            slopes = gains[positive] / lengths[positive]
            max_edge = int(positive_edges[int(np.argmax(slopes))])
        else:
            max_edge = None
    return {
        "local_constants": constants.as_dict(),
        "local_max_slope_edge_id": max_edge,
        "local_max_slope_edge_source": None
        if max_edge is None
        else int(inputs.edge_sources[max_edge]),
        "local_max_slope_edge_target": None
        if max_edge is None
        else int(G.to[max_edge]),
    }


def _materialize_edge_mask(
    G: CompactDiGraph,
    edge_sources: np.ndarray,
    edge_mask: np.ndarray,
) -> tuple[CompactDiGraph, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    selected = np.flatnonzero(edge_mask).astype(np.int32)
    counts = np.bincount(edge_sources[selected], minlength=G.n_nodes).astype(np.int32)
    offsets = np.zeros(G.n_nodes + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    compact = CompactDiGraph(
        offsets=offsets,
        to=G.to[selected].copy(),
        w=G.w[selected].copy(),
        road_id=G.road_id[selected].copy(),
        n_nodes=G.n_nodes,
        n_edges=int(len(selected)),
        n_obj=G.n_obj,
    )
    incident_nodes = base._node_set_for_mask(
        G,
        edge_sources,
        edge_mask,
        extra_nodes=(),
    )
    metadata = {
        "materialization": "same node id space; edge arrays contain only kept corridor edges",
        "incident_nodes": len(incident_nodes),
        "original_edges": G.n_edges,
        "kept_edges": int(len(selected)),
        "compaction_s": time.perf_counter() - start,
    }
    return compact, selected, metadata


def _build_prepared_graphs(
    inputs: base.StaticInputs,
    query: base.QueryBox,
    global_constants: base.MetricConstants,
    requested_modes: set[str],
    *,
    include_geometric: bool,
) -> tuple[dict[str, PreparedGraph], dict[str, Any]]:
    G = inputs.G
    full_mask = np.ones(G.n_edges, dtype=bool)
    full_original_constants = base._metric_constants(G, full_mask)
    full_context = base.GraphContext(
        mode="full",
        edge_mask=full_mask,
        node_count=G.n_nodes,
        edge_count=G.n_edges,
        constants=_constants_with_global_rho(full_original_constants, global_constants),
        metadata={"rho_H_policy": "global_full_graph_reused_across_modes"},
    )
    prepared: dict[str, PreparedGraph] = {
        "full": PreparedGraph(
            mode="full",
            G=G,
            context=full_context,
            edge_id_to_original=np.arange(G.n_edges, dtype=np.int32),
            graph_prep_s=0.0,
            corridor_construction_s=0.0,
            compaction_s=0.0,
            metadata={"graph_storage": "native_full_csr"},
        )
    }

    corridor_start = time.perf_counter()
    certified_mask, d_s, d_t = base._certified_length_corridor_mask(
        G,
        inputs.edge_sources,
        query,
        full_mask,
    )
    corridor_s = time.perf_counter() - corridor_start
    certified_nodes = base._node_set_for_mask(
        G,
        inputs.edge_sources,
        certified_mask,
        extra_nodes=(query.source, query.target),
    )
    local_cert_constants = base._metric_constants(G, certified_mask)
    certified_metadata = {
        "graph_storage": "full_csr_plus_edge_mask_filter",
        "safe_edge_condition": "d_s(u)+l(u,v)+d_t(v) <= Lmax",
        "shortest_length_s_to_t": base._finite_or_none(float(d_s[query.target])),
        "shortest_length_t_reverse_to_s": base._finite_or_none(float(d_t[query.source])),
        "incident_nodes": len(certified_nodes),
        "corridor_construction_s": corridor_s,
    }
    certified_context = base.GraphContext(
        mode="certified_masked",
        edge_mask=certified_mask,
        node_count=len(certified_nodes),
        edge_count=int(np.count_nonzero(certified_mask)),
        constants=_constants_with_global_rho(local_cert_constants, global_constants),
        metadata={**certified_metadata, "rho_H_policy": "global_full_graph_reused_across_modes"},
    )
    prepared["certified_masked"] = PreparedGraph(
        mode="certified_masked",
        G=G,
        context=certified_context,
        edge_id_to_original=np.arange(G.n_edges, dtype=np.int32),
        graph_prep_s=corridor_s,
        corridor_construction_s=corridor_s,
        compaction_s=0.0,
        metadata=certified_metadata,
    )

    if "certified_compact" in requested_modes:
        compact_G, edge_map, compact_metadata = _materialize_edge_mask(
            G,
            inputs.edge_sources,
            certified_mask,
        )
        compact_context = base.GraphContext(
            mode="certified_compact",
            edge_mask=np.ones(compact_G.n_edges, dtype=bool),
            node_count=len(certified_nodes),
            edge_count=compact_G.n_edges,
            constants=_constants_with_global_rho(local_cert_constants, global_constants),
            metadata={
                **certified_metadata,
                **compact_metadata,
                "rho_H_policy": "global_full_graph_reused_across_modes",
            },
        )
        prepared["certified_compact"] = PreparedGraph(
            mode="certified_compact",
            G=compact_G,
            context=compact_context,
            edge_id_to_original=edge_map,
            graph_prep_s=corridor_s + compact_metadata["compaction_s"],
            corridor_construction_s=corridor_s,
            compaction_s=compact_metadata["compaction_s"],
            metadata={**certified_metadata, **compact_metadata},
        )

    comparison: dict[str, Any] = {
        "full": {"nodes": G.n_nodes, "edges": G.n_edges},
        "certified_length_corridor": {
            "nodes": len(certified_nodes),
            "edges": int(np.count_nonzero(certified_mask)),
            "corridor_construction_s": corridor_s,
            "storage_masked": "full CSR plus edge-mask test in forward relaxations",
            "storage_compact": "optional materialized edge-only CSR over same node ids",
            **_mode_metric_constant_snapshot(G, certified_mask, inputs),
        },
    }

    if include_geometric:
        geo_start = time.perf_counter()
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
        geometric_s = time.perf_counter() - geo_start
        geometric_mask = base._geometric_edge_mask(G, kept_nodes)
        geometric_nodes = base._node_set_for_mask(
            G,
            inputs.edge_sources,
            geometric_mask,
            extra_nodes=(query.source, query.target),
        )
        local_geo_constants = base._metric_constants(G, geometric_mask)
        geometric_context = base.GraphContext(
            mode="geometric_diagnostic",
            edge_mask=geometric_mask,
            node_count=len(geometric_nodes),
            edge_count=int(np.count_nonzero(geometric_mask)),
            constants=_constants_with_global_rho(local_geo_constants, global_constants),
            metadata={
                "graph_storage": "full_csr_plus_edge_mask_filter",
                "kept_cells": len(kept_cells),
                "kept_nodes": len(kept_nodes),
                "incident_nodes": len(geometric_nodes),
                "corridor_slack_m": query.corridor_slack_m,
                "max_hops_from_boundary": query.max_hops_from_boundary,
                "geometric_construction_s": geometric_s,
                "rho_H_policy": "global_full_graph_reused_across_modes",
            },
        )
        prepared["geometric_diagnostic"] = PreparedGraph(
            mode="geometric_diagnostic",
            G=G,
            context=geometric_context,
            edge_id_to_original=np.arange(G.n_edges, dtype=np.int32),
            graph_prep_s=geometric_s,
            corridor_construction_s=0.0,
            compaction_s=0.0,
            metadata=geometric_context.metadata,
        )
        comparison["geometric"] = {
            "kept_cells": len(kept_cells),
            "kept_nodes": len(kept_nodes),
            "incident_nodes": len(geometric_nodes),
            "edges": int(np.count_nonzero(geometric_mask)),
            "geometric_construction_s": geometric_s,
            **_mode_metric_constant_snapshot(G, geometric_mask, inputs),
        }
        comparison["set_differences"] = {
            "certified_edges_not_geometric": int(
                np.count_nonzero(certified_mask & ~geometric_mask)
            ),
            "geometric_edges_not_certified": int(
                np.count_nonzero(geometric_mask & ~certified_mask)
            ),
            "certified_nodes_not_geometric": len(certified_nodes - geometric_nodes),
            "geometric_nodes_not_certified": len(geometric_nodes - certified_nodes),
        }

    return prepared, comparison


def _path_nodes_from_edge_ids(
    G: CompactDiGraph,
    source: int,
    target: int,
    edge_ids: Sequence[int],
) -> tuple[int, ...] | None:
    nodes = [int(source)]
    cur = int(source)
    for edge_id in edge_ids:
        eid = int(edge_id)
        if eid < 0 or eid >= G.n_edges:
            return None
        start = int(G.offsets[cur])
        end = int(G.offsets[cur + 1])
        if not (start <= eid < end):
            return None
        nxt = int(G.to[eid])
        nodes.append(nxt)
        cur = nxt
    if cur != int(target):
        return None
    return tuple(nodes)


def _result_from_original_edges(
    G: CompactDiGraph,
    source: int,
    target: int,
    edge_ids: Sequence[int],
    *,
    scalar_cost: float = 0.0,
) -> base.ScalarPathResult | None:
    path_nodes = _path_nodes_from_edge_ids(G, source, target, edge_ids)
    if path_nodes is None:
        return None
    edge_tuple = tuple(int(edge_id) for edge_id in edge_ids)
    metrics = base._metrics_from_edge_ids(G, edge_tuple)
    return base.ScalarPathResult(
        route_found=True,
        scalar_cost=scalar_cost,
        path_nodes=path_nodes,
        edge_ids=edge_tuple,
        metrics=metrics,
        stats=base.DijkstraStats(0, 0, 0, 0, 0.0),
    )


def _map_result_to_original(
    prepared: PreparedGraph,
    original_G: CompactDiGraph,
    result: base.ScalarPathResult,
) -> base.ScalarPathResult:
    if not result.route_found or result.metrics is None:
        return result
    original_edges = tuple(
        int(prepared.edge_id_to_original[int(edge_id)]) for edge_id in result.edge_ids
    )
    metrics = base._metrics_from_edge_ids(original_G, original_edges)
    return base.ScalarPathResult(
        route_found=True,
        scalar_cost=result.scalar_cost,
        path_nodes=result.path_nodes,
        edge_ids=original_edges,
        metrics=metrics,
        stats=result.stats,
    )


def _make_witness_from_result(
    *,
    route_id: str,
    pair_id: str,
    generator: str,
    scalar_name: str,
    via_vertex: int | None,
    original_G: CompactDiGraph,
    result: base.ScalarPathResult,
) -> WitnessRoute | None:
    if not result.route_found or result.metrics is None:
        return None
    validation = base._validate_path(original_G, result)
    elementary, repeated = base._elementary_status(result.path_nodes)
    if not validation.passed or not elementary:
        return None
    recomputed = base._metrics_from_edge_ids(original_G, result.edge_ids)
    return WitnessRoute(
        route_id=route_id,
        pair_id=pair_id,
        generator=generator,
        scalar_name=scalar_name,
        via_vertex=via_vertex,
        path_nodes=tuple(int(node) for node in result.path_nodes),
        edge_ids=tuple(int(edge_id) for edge_id in result.edge_ids),
        metrics=recomputed,
        validation=validation,
        elementary=elementary,
        repeated_vertex_count=repeated,
    )


def _validate_witness_for_query(
    inputs: base.StaticInputs,
    item: BenchmarkItem,
) -> BenchmarkItem:
    result = _result_from_original_edges(
        inputs.G,
        item.query.source,
        item.query.target,
        item.witness.edge_ids,
    )
    if result is None or result.metrics is None:
        raise ValueError(f"{item.query.name}: witness edge path is invalid")
    validation = base._validate_path(inputs.G, result)
    elementary, repeated = base._elementary_status(result.path_nodes)
    if not validation.passed:
        raise ValueError(f"{item.query.name}: witness failed directed CSR validation")
    if not elementary:
        raise ValueError(f"{item.query.name}: witness is non-elementary")
    if not item.query.is_feasible(result.metrics):
        raise ValueError(f"{item.query.name}: witness does not satisfy declared box")
    witness = WitnessRoute(
        route_id=item.witness.route_id,
        pair_id=item.witness.pair_id,
        generator=item.witness.generator,
        scalar_name=item.witness.scalar_name,
        via_vertex=item.witness.via_vertex,
        path_nodes=tuple(int(node) for node in result.path_nodes),
        edge_ids=tuple(int(edge_id) for edge_id in result.edge_ids),
        metrics=result.metrics,
        validation=validation,
        elementary=elementary,
        repeated_vertex_count=repeated,
    )
    score, violations, violated = _violation_details(
        item.query,
        item.physical_shortest.metrics,
    )
    assert score is not None and violations is not None
    return BenchmarkItem(
        query=item.query,
        pair_id=item.pair_id,
        category=item.category,
        tightness=item.tightness,
        tags=item.tags,
        witness=witness,
        physical_shortest=item.physical_shortest,
        shortest_violation_score=score,
        shortest_violations=violations,
        shortest_violated_constraints=violated,
        adversarial=score > 1e-6,
        deliberate_adversarial=item.deliberate_adversarial,
    )


def _reconstruct_via_candidate(
    prepared: PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profile: via.ProfileCandidate,
) -> ReconstructionAttempt:
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
        return ReconstructionAttempt(None, "branch_not_reconstructable", None, None)

    forward_nodes, forward_edges = forward_branch
    backward_nodes, backward_edges = backward_branch
    if not forward_nodes or not backward_nodes or forward_nodes[-1] != via_vertex:
        return ReconstructionAttempt(None, "forward_branch_bad_endpoint", None, None)
    if backward_nodes[0] != via_vertex:
        return ReconstructionAttempt(None, "backward_branch_bad_endpoint", None, None)

    path_nodes = tuple(forward_nodes + backward_nodes[1:])
    run_edge_ids = tuple(forward_edges + backward_edges)
    original_edges = tuple(
        int(prepared.edge_id_to_original[int(edge_id)]) for edge_id in run_edge_ids
    )
    metrics = base._metrics_from_edge_ids(original_G, original_edges)
    scalar_cost = float(forward.dist[via_vertex]) + float(backward.dist[via_vertex])
    result = via._make_path_result(scalar_cost, path_nodes, original_edges, metrics)
    validation = base._validate_path(original_G, result)
    repeated = via._repeated_vertex_count(path_nodes)
    if not validation.passed:
        return ReconstructionAttempt(
            None,
            "directed_validation_failed",
            validation,
            repeated,
        )
    candidate = via.ExactViaCandidate(
        via_vertex=via_vertex,
        metrics=metrics,
        profile_metrics=profile.metrics,
        box_score=via._box_center_score(query, metrics),
        scalar_cost=scalar_cost,
        path_nodes=path_nodes,
        edge_ids=original_edges,
        validation=validation,
        repeated_vertex_count=repeated,
        profile_exact_deltas=via._profile_exact_deltas(profile.metrics, metrics),
    )
    return ReconstructionAttempt(candidate, None, validation, repeated)


def _profile_sort_key(
    query: base.QueryBox,
    profile: via.ProfileCandidate,
) -> tuple[float, float, float, int]:
    return (
        via._box_center_score(query, profile.metrics),
        profile.metrics.length,
        query.normalized_violation_score(profile.metrics),
        profile.via_vertex,
    )


def _scan_profiles_and_empirical_hmax(
    query: base.QueryBox,
    forward: via.TreeResult,
    backward: via.TreeResult,
) -> tuple[
    list[via.ProfileCandidate],
    int,
    dict[str, Any] | None,
    dict[str, Any] | None,
    float,
]:
    start = time.perf_counter()
    profile_candidates: list[via.ProfileCandidate] = []
    via_vertices_scanned = 0
    best_profile: via.ProfileCandidate | None = None
    best_profile_score: float | None = None
    empirical_hmax: via.ProfileCandidate | None = None

    for via_vertex in range(len(forward.dist)):
        metrics = via._combined_profile(query, forward, backward, via_vertex)
        if metrics is None:
            continue
        via_vertices_scanned += 1
        profile = via.ProfileCandidate(via_vertex, metrics)
        score = query.normalized_violation_score(metrics)
        if best_profile is None or score < float(best_profile_score):
            best_profile = profile
            best_profile_score = score
        if query.is_feasible(metrics):
            profile_candidates.append(profile)
        if metrics.length <= query.Lmax + 1e-6:
            if empirical_hmax is None or (
                metrics.elevation > empirical_hmax.metrics.elevation + 1e-9
                or (
                    abs(metrics.elevation - empirical_hmax.metrics.elevation) <= 1e-9
                    and (metrics.length, via_vertex)
                    < (empirical_hmax.metrics.length, empirical_hmax.via_vertex)
                )
            ):
                empirical_hmax = profile

    nearest = None
    if best_profile is not None and best_profile_score is not None:
        nearest = {
            "via_vertex": best_profile.via_vertex,
            "normalized_violation_score": best_profile_score,
            "violations": query.violations(best_profile.metrics),
            "metrics": best_profile.metrics.as_dict(),
        }
    empirical = None
    if empirical_hmax is not None:
        empirical = {
            "label": "empirical_via_Hmax",
            "scope": "maximum H among this scalar's same-scalar via profiles with L_v <= Lmax; not a certified global envelope",
            "via_vertex": empirical_hmax.via_vertex,
            "profile_metrics": empirical_hmax.metrics.as_dict(),
        }
    return (
        profile_candidates,
        via_vertices_scanned,
        empirical,
        nearest,
        time.perf_counter() - start,
    )


def _reconstruct_empirical_hmax(
    prepared: PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    forward: via.TreeResult,
    backward: via.TreeResult,
    empirical: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if empirical is None:
        return None
    via_vertex = int(empirical["via_vertex"])
    profile = via.ProfileCandidate(
        via_vertex=via_vertex,
        metrics=base.RouteMetrics(
            length=float(empirical["profile_metrics"]["length"]),
            elevation=float(empirical["profile_metrics"]["elevation"]),
            popularity_length=float(empirical["profile_metrics"]["popularity_length"]),
            width_length=float(empirical["profile_metrics"]["width_length"]),
            road_changes=int(empirical["profile_metrics"]["road_changes"]),
        ),
    )
    attempt = _reconstruct_via_candidate(
        prepared,
        original_G,
        query,
        forward,
        backward,
        profile,
    )
    out = dict(empirical)
    if attempt.candidate is None:
        out.update(
            {
                "reconstructed": False,
                "reconstruction_reason": attempt.reason,
                "elementary": False if attempt.repeated_vertex_count else None,
                "exact_H": None,
            }
        )
        if attempt.validation is not None:
            out["validation"] = attempt.validation.as_dict()
        return out
    out.update(
        {
            "reconstructed": True,
            "elementary": attempt.candidate.elementary,
            "repeated_vertex_count": attempt.candidate.repeated_vertex_count,
            "exact_H": attempt.candidate.metrics.elevation,
            "exact_metrics": attempt.candidate.metrics.as_dict(),
            "validation": attempt.candidate.validation.as_dict(),
        }
    )
    return out


def _reconstruct_exhaustive(
    prepared: PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profiles: Sequence[via.ProfileCandidate],
) -> tuple[list[via.ExactViaCandidate], via.ReconstructionCounters, dict[str, Any] | None, float]:
    start = time.perf_counter()
    counters = via.ReconstructionCounters()
    exact_candidates: list[via.ExactViaCandidate] = []
    best_near: dict[str, Any] | None = None
    best_near_score: float | None = None

    for profile in profiles:
        counters.reconstructed_count += 1
        attempt = _reconstruct_via_candidate(
            prepared,
            original_G,
            query,
            forward,
            backward,
            profile,
        )
        if attempt.candidate is None:
            counters.rejected_validation_count += 1
            continue
        candidate = attempt.candidate
        if not candidate.elementary:
            counters.rejected_non_elementary_count += 1
            continue
        score = query.normalized_violation_score(candidate.metrics)
        if best_near is None or score < float(best_near_score):
            best_near = {
                "via_vertex": candidate.via_vertex,
                "normalized_violation_score": score,
                "violations": query.violations(candidate.metrics),
                "metrics": candidate.metrics.as_dict(),
                "profile_metrics": candidate.profile_metrics.as_dict(),
                "profile_exact_deltas": candidate.profile_exact_deltas,
                "box_score": candidate.box_score,
            }
            best_near_score = score
        if not query.is_feasible(candidate.metrics):
            counters.rejected_exact_box_count += 1
            continue
        counters.exact_feasible_count += 1
        exact_candidates.append(candidate)

    return exact_candidates, counters, best_near, time.perf_counter() - start


def _reconstruct_first_hit(
    prepared: PreparedGraph,
    original_G: CompactDiGraph,
    query: base.QueryBox,
    forward: via.TreeResult,
    backward: via.TreeResult,
    profiles: Sequence[via.ProfileCandidate],
) -> tuple[via.ExactViaCandidate | None, via.ReconstructionCounters, float]:
    start = time.perf_counter()
    counters = via.ReconstructionCounters()
    ordered = sorted(profiles, key=lambda profile: _profile_sort_key(query, profile))
    for profile in ordered:
        counters.reconstructed_count += 1
        attempt = _reconstruct_via_candidate(
            prepared,
            original_G,
            query,
            forward,
            backward,
            profile,
        )
        if attempt.candidate is None:
            counters.rejected_validation_count += 1
            continue
        candidate = attempt.candidate
        if not candidate.elementary:
            counters.rejected_non_elementary_count += 1
            continue
        if not query.is_feasible(candidate.metrics):
            counters.rejected_exact_box_count += 1
            continue
        counters.exact_feasible_count = 1
        counters.time_to_first_feasible_s = time.perf_counter() - start
        return candidate, counters, time.perf_counter() - start
    return None, counters, time.perf_counter() - start


def _row_base(
    item: BenchmarkItem,
    prepared: PreparedGraph,
    spec_name: str,
    method: str,
    *,
    wref: float | None,
    pref: float | None,
) -> dict[str, Any]:
    q = item.query
    row: dict[str, Any] = {
        "query_id": q.name,
        "pair_id": item.pair_id,
        "category": item.category,
        "tightness": item.tightness,
        "tags": "|".join(item.tags),
        "adversarial": item.adversarial,
        "deliberate_adversarial": item.deliberate_adversarial,
        "source": q.source,
        "target": q.target,
        "graph_mode": prepared.mode,
        "graph_edge_count": prepared.context.edge_count,
        "graph_node_count": prepared.context.node_count,
        "graph_storage": prepared.metadata.get("graph_storage"),
        "graph_prep_s": prepared.graph_prep_s,
        "corridor_construction_s": prepared.corridor_construction_s,
        "compaction_s": prepared.compaction_s,
        "scalar_name": spec_name,
        "method": method,
        "Lmin": q.Lmin,
        "Lmax": q.Lmax,
        "Hmin": q.Hmin,
        "Hmax": q.Hmax,
        "Pmin": q.Pmin,
        "Wmax": q.Wmax,
        "Wref_used": wref,
        "Pref_used": pref,
        "rho_H_used": prepared.context.constants.rho_H,
        "witness_route_id": item.witness.route_id,
        "witness_generator": item.witness.generator,
        "witness_scalar_name": item.witness.scalar_name,
        "witness_via_vertex": item.witness.via_vertex,
        **_metric_fields("witness_", item.witness.metrics),
        "shortest_violation_score": item.shortest_violation_score,
        "shortest_violated_constraints": "|".join(
            item.shortest_violated_constraints
        ),
        "shortest_violations": item.shortest_violations,
        **_metric_fields("shortest_", item.physical_shortest.metrics),
    }
    return row


def _run_direct(
    inputs: base.StaticInputs,
    item: BenchmarkItem,
    prepared: PreparedGraph,
    spec: base.ScalarizationSpec,
) -> dict[str, Any]:
    wref, pref = _effective_refs(item.query, spec)
    row = _row_base(
        item,
        prepared,
        spec.name,
        "direct",
        wref=wref,
        pref=pref,
    )
    result = base._dijkstra_scalar_path(prepared.G, item.query, prepared.context, spec)
    original_result = _map_result_to_original(prepared, inputs.G, result)
    validation = base._validate_path(inputs.G, original_result)
    elementary, repeated = base._elementary_status(original_result.path_nodes)
    metrics = original_result.metrics
    score, violations, violated = _violation_details(item.query, metrics)
    feasible = (
        bool(original_result.route_found)
        and metrics is not None
        and validation.passed
        and elementary
        and item.query.is_feasible(metrics)
    )
    row.update(
        {
            "route_found": bool(original_result.route_found),
            "feasible": feasible,
            **_metric_fields("", metrics),
            "elementary": elementary,
            "repeated_vertex_count": repeated,
            "scalar_search_count": 1,
            "forward_tree_s": None,
            "backward_tree_s": None,
            "profile_scan_s": None,
            "first_hit_reconstruction_s": None,
            "exhaustive_reconstruction_s": None,
            "time_to_first_feasible": result.stats.elapsed_s if feasible else None,
            "first_hit_total_s": result.stats.elapsed_s,
            "exhaustive_total_s": result.stats.elapsed_s,
            "candidates_reconstructed_before_first_hit": None,
            "via_vertices_scanned": 0,
            "profile_feasible_count": 0,
            "exact_feasible_count": 1 if feasible else 0,
            "non_elementary_count": 0 if elementary else 1,
            "normalized_violation_score": score,
            "violations": violations,
            "violated_constraints": "|".join(violated),
            "validation_passed": validation.passed,
            "validation": validation.as_dict(),
            "heap_pops": result.stats.heap_pops,
            "expanded_nodes": result.stats.expanded_nodes,
            "edge_scans": result.stats.edge_scans,
            "raw_edge_rows_checked": result.stats.raw_edge_rows_checked,
            "equal_cost_relaxations": None,
            "equal_cost_resource_distinct_relaxations": None,
            "parent_changes_due_to_road_tiebreak": None,
            "nearest_profile": None,
            "nearest_exact_candidate": None
            if metrics is None
            else {
                "normalized_violation_score": score,
                "violations": violations,
                "metrics": metrics.as_dict(),
            },
            "empirical_via_Hmax": None,
            "route_path_nodes": len(original_result.path_nodes),
            "route_edges": len(original_result.edge_ids),
        }
    )
    return row


def _tie_totals(forward: via.TreeResult, backward: via.TreeResult) -> dict[str, Any]:
    return {
        "equal_cost_relaxations": (
            forward.tie_diagnostics.equal_cost_relaxations
            + backward.tie_diagnostics.equal_cost_relaxations
        ),
        "equal_cost_resource_distinct_relaxations": (
            forward.tie_diagnostics.equal_cost_resource_distinct_relaxations
            + backward.tie_diagnostics.equal_cost_resource_distinct_relaxations
        ),
        "parent_changes_due_to_road_tiebreak": (
            forward.tie_diagnostics.parent_changes_due_to_road_tiebreak
            + backward.tie_diagnostics.parent_changes_due_to_road_tiebreak
        ),
        "forward_equal_cost_relaxations": (
            forward.tie_diagnostics.equal_cost_relaxations
        ),
        "backward_equal_cost_relaxations": (
            backward.tie_diagnostics.equal_cost_relaxations
        ),
        "forward_equal_cost_resource_distinct_relaxations": (
            forward.tie_diagnostics.equal_cost_resource_distinct_relaxations
        ),
        "backward_equal_cost_resource_distinct_relaxations": (
            backward.tie_diagnostics.equal_cost_resource_distinct_relaxations
        ),
        "forward_parent_changes_due_to_road_tiebreak": (
            forward.tie_diagnostics.parent_changes_due_to_road_tiebreak
        ),
        "backward_parent_changes_due_to_road_tiebreak": (
            backward.tie_diagnostics.parent_changes_due_to_road_tiebreak
        ),
    }


def _run_via(
    inputs: base.StaticInputs,
    item: BenchmarkItem,
    prepared: PreparedGraph,
    spec: base.ScalarizationSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wref, pref = _effective_refs(item.query, spec)
    row = _row_base(
        item,
        prepared,
        spec.name,
        "via",
        wref=wref,
        pref=pref,
    )
    reverse_start = time.perf_counter()
    reverse_adj = via._build_reverse_edge_adjacency(
        prepared.G,
        prepared.context.edge_mask,
    )
    reverse_adjacency_build_s = time.perf_counter() - reverse_start
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
    (
        profiles,
        via_vertices_scanned,
        empirical_hmax,
        nearest_profile,
        profile_scan_s,
    ) = _scan_profiles_and_empirical_hmax(item.query, forward, backward)
    empirical_hmax = _reconstruct_empirical_hmax(
        prepared,
        inputs.G,
        item.query,
        forward,
        backward,
        empirical_hmax,
    )
    first_hit, first_counters, first_reconstruction_s = _reconstruct_first_hit(
        prepared,
        inputs.G,
        item.query,
        forward,
        backward,
        profiles,
    )
    exact_candidates, exhaustive_counters, best_near, exhaustive_reconstruction_s = (
        _reconstruct_exhaustive(
            prepared,
            inputs.G,
            item.query,
            forward,
            backward,
            profiles,
        )
    )
    tree_and_scan_s = (
        reverse_adjacency_build_s
        + forward.stats.elapsed_s
        + backward.stats.elapsed_s
        + profile_scan_s
    )
    first_hit_total_s = tree_and_scan_s + first_reconstruction_s
    time_to_first = first_hit_total_s if first_hit is not None else None
    exhaustive_total_s = tree_and_scan_s + exhaustive_reconstruction_s
    route = first_hit
    metrics = None if route is None else route.metrics
    score, violations, violated = _violation_details(item.query, metrics)
    feasible = bool(exact_candidates)
    diagnostic_nearest = best_near or nearest_profile
    if not feasible and diagnostic_nearest is not None:
        score = diagnostic_nearest.get("normalized_violation_score")
        violations = diagnostic_nearest.get("violations")
        if isinstance(violations, dict):
            violated = tuple(key for key, value in violations.items() if value > 1e-6)
    tie_totals = _tie_totals(forward, backward)
    row.update(
        {
            "route_found": route is not None,
            "feasible": feasible,
            **_metric_fields("", metrics),
            "elementary": None if route is None else route.elementary,
            "repeated_vertex_count": None
            if route is None
            else route.repeated_vertex_count,
            "scalar_search_count": 2,
            "reverse_adjacency_build_s": reverse_adjacency_build_s,
            "forward_tree_s": forward.stats.elapsed_s,
            "backward_tree_s": backward.stats.elapsed_s,
            "profile_scan_s": profile_scan_s,
            "first_hit_reconstruction_s": first_reconstruction_s,
            "exhaustive_reconstruction_s": exhaustive_reconstruction_s,
            "time_to_first_feasible": time_to_first,
            "first_hit_total_s": first_hit_total_s,
            "exhaustive_total_s": exhaustive_total_s,
            "candidates_reconstructed_before_first_hit": (
                first_counters.reconstructed_count if route is not None else None
            ),
            "candidates_reconstructed_in_first_hit_mode": (
                first_counters.reconstructed_count
            ),
            "via_vertices_scanned": via_vertices_scanned,
            "profile_feasible_count": len(profiles),
            "exact_feasible_count": len(exact_candidates),
            "non_elementary_count": exhaustive_counters.rejected_non_elementary_count,
            "rejected_validation_count": exhaustive_counters.rejected_validation_count,
            "rejected_exact_box_count": exhaustive_counters.rejected_exact_box_count,
            "normalized_violation_score": 0.0 if feasible else score,
            "violations": {} if feasible else violations,
            "violated_constraints": "" if feasible else "|".join(violated),
            "validation_passed": None if route is None else route.validation.passed,
            "validation": None if route is None else route.validation.as_dict(),
            "heap_pops": forward.stats.heap_pops + backward.stats.heap_pops,
            "expanded_nodes": forward.stats.expanded_nodes
            + backward.stats.expanded_nodes,
            "edge_scans": forward.stats.edge_scans + backward.stats.edge_scans,
            "raw_edge_rows_checked": forward.stats.raw_edge_rows_checked
            + backward.stats.raw_edge_rows_checked,
            "forward_heap_pops": forward.stats.heap_pops,
            "forward_expanded_nodes": forward.stats.expanded_nodes,
            "forward_edge_scans": forward.stats.edge_scans,
            "forward_raw_edge_rows_checked": forward.stats.raw_edge_rows_checked,
            "backward_heap_pops": backward.stats.heap_pops,
            "backward_expanded_nodes": backward.stats.expanded_nodes,
            "backward_edge_scans": backward.stats.edge_scans,
            "backward_raw_edge_rows_checked": backward.stats.raw_edge_rows_checked,
            **tie_totals,
            "nearest_profile": nearest_profile,
            "nearest_exact_candidate": best_near,
            "empirical_via_Hmax": empirical_hmax,
            "route_path_nodes": 0 if route is None else len(route.path_nodes),
            "route_edges": 0 if route is None else len(route.edge_ids),
        }
    )
    artifact = {
        "query_id": item.query.name,
        "graph_mode": prepared.mode,
        "scalar_name": spec.name,
        "exact_feasible_candidates": exact_candidates,
        "first_hit_candidate": first_hit,
        "nearest_profile": nearest_profile,
        "nearest_exact_candidate": best_near,
        "row": row,
    }
    return row, artifact


def _make_union_row(
    item: BenchmarkItem,
    prepared: PreparedGraph,
    union_name: str,
    member_names: Sequence[str],
    artifacts_by_scalar: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = _row_base(
        item,
        prepared,
        union_name,
        "via_union",
        wref=None,
        pref=None,
    )
    unique: dict[tuple[int, ...], tuple[str, via.ExactViaCandidate]] = {}
    for scalar_name in member_names:
        artifact = artifacts_by_scalar.get(scalar_name)
        if artifact is None:
            continue
        for candidate in artifact["exact_feasible_candidates"]:
            key = tuple(int(edge_id) for edge_id in candidate.edge_ids)
            if key not in unique:
                unique[key] = (scalar_name, candidate)
    best: tuple[str, via.ExactViaCandidate] | None = None
    if unique:
        best = min(
            unique.values(),
            key=lambda item_: (
                item_[1].box_score,
                item_[1].metrics.length,
                item_[0],
                item_[1].via_vertex,
            ),
        )
    member_rows = [
        artifacts_by_scalar[name]["row"]
        for name in member_names
        if name in artifacts_by_scalar
    ]
    solved_member_times = [
        float(member["time_to_first_feasible"])
        for member in member_rows
        if member.get("time_to_first_feasible") not in (None, "")
    ]
    if best is None:
        nearest_options = [
            member.get("nearest_exact_candidate") or member.get("nearest_profile")
            for member in member_rows
            if member.get("nearest_exact_candidate") or member.get("nearest_profile")
        ]
        nearest = min(
            nearest_options,
            key=lambda entry: float(entry.get("normalized_violation_score", float("inf"))),
        ) if nearest_options else None
        score = None if nearest is None else nearest.get("normalized_violation_score")
        violations = None if nearest is None else nearest.get("violations")
        violated = ()
        if isinstance(violations, dict):
            violated = tuple(key for key, value in violations.items() if value > 1e-6)
        metrics = None
        scalar_source = None
        route = None
    else:
        scalar_source, route = best
        metrics = route.metrics
        score, violations, violated = _violation_details(item.query, metrics)
        nearest = None
    row.update(
        {
            "route_found": best is not None,
            "feasible": best is not None,
            **_metric_fields("", metrics),
            "elementary": None if best is None else route.elementary,
            "repeated_vertex_count": None if best is None else route.repeated_vertex_count,
            "scalar_search_count": 2 * len(member_names),
            "union_members": "|".join(member_names),
            "union_winning_scalar": scalar_source,
            "forward_tree_s": sum(
                float(member.get("forward_tree_s") or 0.0) for member in member_rows
            ),
            "backward_tree_s": sum(
                float(member.get("backward_tree_s") or 0.0) for member in member_rows
            ),
            "profile_scan_s": sum(
                float(member.get("profile_scan_s") or 0.0) for member in member_rows
            ),
            "first_hit_reconstruction_s": None,
            "exhaustive_reconstruction_s": sum(
                float(member.get("exhaustive_reconstruction_s") or 0.0)
                for member in member_rows
            ),
            "time_to_first_feasible": min(solved_member_times)
            if solved_member_times
            else None,
            "first_hit_total_s": min(solved_member_times)
            if solved_member_times
            else sum(float(member.get("first_hit_total_s") or 0.0) for member in member_rows),
            "exhaustive_total_s": sum(
                float(member.get("exhaustive_total_s") or 0.0) for member in member_rows
            ),
            "candidates_reconstructed_before_first_hit": None,
            "via_vertices_scanned": sum(
                int(member.get("via_vertices_scanned") or 0) for member in member_rows
            ),
            "profile_feasible_count": sum(
                int(member.get("profile_feasible_count") or 0) for member in member_rows
            ),
            "exact_feasible_count": len(unique),
            "non_elementary_count": sum(
                int(member.get("non_elementary_count") or 0) for member in member_rows
            ),
            "normalized_violation_score": 0.0 if best is not None else score,
            "violations": {} if best is not None else violations,
            "violated_constraints": "" if best is not None else "|".join(violated),
            "validation_passed": None if best is None else route.validation.passed,
            "validation": None if best is None else route.validation.as_dict(),
            "heap_pops": sum(int(member.get("heap_pops") or 0) for member in member_rows),
            "expanded_nodes": sum(
                int(member.get("expanded_nodes") or 0) for member in member_rows
            ),
            "edge_scans": sum(int(member.get("edge_scans") or 0) for member in member_rows),
            "raw_edge_rows_checked": sum(
                int(member.get("raw_edge_rows_checked") or 0) for member in member_rows
            ),
            "equal_cost_relaxations": sum(
                int(member.get("equal_cost_relaxations") or 0) for member in member_rows
            ),
            "equal_cost_resource_distinct_relaxations": sum(
                int(member.get("equal_cost_resource_distinct_relaxations") or 0)
                for member in member_rows
            ),
            "parent_changes_due_to_road_tiebreak": sum(
                int(member.get("parent_changes_due_to_road_tiebreak") or 0)
                for member in member_rows
            ),
            "nearest_profile": nearest,
            "nearest_exact_candidate": nearest,
            "empirical_via_Hmax": [
                member.get("empirical_via_Hmax")
                for member in member_rows
                if member.get("empirical_via_Hmax") is not None
            ],
            "route_path_nodes": 0 if best is None else len(route.path_nodes),
            "route_edges": 0 if best is None else len(route.edge_ids),
            "union_timing_note": (
                "time_to_first_feasible is the best member first-hit time; "
                "exhaustive_total_s is the sum of member via exhaustive times"
            ),
        }
    )
    return row


def _loose_query_for_pair(
    pair_id: str,
    source: int,
    target: int,
    shortest_metrics: base.RouteMetrics | None = None,
) -> base.QueryBox:
    if shortest_metrics is None:
        Lmax = 80000.0
        Hmax = 1500.0
    else:
        Lmax = max(12000.0, shortest_metrics.length * 1.9 + 4000.0)
        Hmax = max(700.0, shortest_metrics.elevation * 3.0 + 250.0)
    return base.QueryBox(
        name=f"{pair_id}_probe",
        source=source,
        target=target,
        Lmin=0.0,
        Lmax=Lmax,
        Hmin=0.0,
        Hmax=Hmax,
        Pmin=150.0,
        Wmax=15.0,
    )


def _direct_witness_for_pair(
    inputs: base.StaticInputs,
    full_prepared: PreparedGraph,
    pair_id: str,
    source: int,
    target: int,
    spec: base.ScalarizationSpec,
    *,
    route_id_suffix: str,
) -> WitnessRoute | None:
    probe = _loose_query_for_pair(pair_id, source, target)
    result = base._dijkstra_scalar_path(full_prepared.G, probe, full_prepared.context, spec)
    original_result = _map_result_to_original(full_prepared, inputs.G, result)
    return _make_witness_from_result(
        route_id=f"{pair_id}:{route_id_suffix}:{spec.name}",
        pair_id=pair_id,
        generator="direct_scalar_path",
        scalar_name=spec.name,
        via_vertex=None,
        original_G=inputs.G,
        result=original_result,
    )


def _anchor_node_candidates(inputs: base.StaticInputs) -> list[tuple[str, int, int]]:
    G = inputs.G
    out_degree = np.diff(G.offsets) > 0
    in_degree = np.bincount(G.to.astype(np.int64), minlength=G.n_nodes) > 0
    valid = out_degree & in_degree
    valid_nodes = np.flatnonzero(valid)
    xy = inputs.xy_int.astype(np.float64)
    if len(valid_nodes) == 0:
        return []

    def extreme(name: str, values: np.ndarray, want_min: bool) -> tuple[str, int]:
        sliced = values[valid_nodes]
        pos = int(np.argmin(sliced) if want_min else np.argmax(sliced))
        return name, int(valid_nodes[pos])

    anchors = dict(
        [
            extreme("west", xy[:, 0], True),
            extreme("east", xy[:, 0], False),
            extreme("south", xy[:, 1], True),
            extreme("north", xy[:, 1], False),
            extreme("southwest", xy[:, 0] + xy[:, 1], True),
            extreme("northeast", xy[:, 0] + xy[:, 1], False),
            extreme("northwest", -xy[:, 0] + xy[:, 1], False),
            extreme("southeast", xy[:, 0] - xy[:, 1], False),
        ]
    )
    raw = [
        ("anchor_west_east", anchors["west"], anchors["east"]),
        ("anchor_east_west", anchors["east"], anchors["west"]),
        ("anchor_south_north", anchors["south"], anchors["north"]),
        ("anchor_north_south", anchors["north"], anchors["south"]),
        ("anchor_southwest_northeast", anchors["southwest"], anchors["northeast"]),
        ("anchor_northeast_southwest", anchors["northeast"], anchors["southwest"]),
        ("anchor_northwest_southeast", anchors["northwest"], anchors["southeast"]),
        ("anchor_southeast_northwest", anchors["southeast"], anchors["northwest"]),
    ]
    dedup: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for name, s, t in raw:
        if s == t or (s, t) in seen:
            continue
        seen.add((s, t))
        dedup.append((name, s, t))
    return dedup


def _random_far_pair_candidates(
    inputs: base.StaticInputs,
    *,
    seed: int,
    count: int,
) -> list[tuple[str, int, int]]:
    G = inputs.G
    out_degree = np.diff(G.offsets) > 0
    in_degree = np.bincount(G.to.astype(np.int64), minlength=G.n_nodes) > 0
    valid_nodes = [int(node) for node in np.flatnonzero(out_degree & in_degree)]
    rng = random.Random(seed)
    sample = rng.sample(valid_nodes, min(500, len(valid_nodes)))
    xy = inputs.xy_int.astype(np.float64)
    pairs: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx, source in enumerate(sample[: min(150, len(sample))]):
        coords = xy[sample] - xy[source]
        dist2 = np.sum(coords * coords, axis=1)
        far_order = np.argsort(dist2)[::-1][:3]
        for far_pos in far_order:
            target = int(sample[int(far_pos)])
            if source == target or (source, target) in seen:
                continue
            seen.add((source, target))
            pairs.append((f"far_{idx:03d}_{len(pairs):03d}", source, target))
            if len(pairs) >= count:
                return pairs
    return pairs


def _select_pair_specs(
    inputs: base.StaticInputs,
    full_prepared: PreparedGraph,
    *,
    pair_count: int,
    max_candidate_pairs: int,
    seed: int,
    min_shortest_m: float,
    max_shortest_m: float,
) -> list[PairSpec]:
    selection_pool_size = min(max_candidate_pairs, max(pair_count, pair_count * 3))
    candidates: list[tuple[str, int, int]] = [
        ("paris_bures", base.PARIS_BURES_SOURCE, base.PARIS_BURES_TARGET)
    ]
    candidates.extend(_anchor_node_candidates(inputs))
    candidates.extend(
        _random_far_pair_candidates(
            inputs,
            seed=seed,
            count=max(0, max_candidate_pairs - len(candidates)),
        )
    )
    selected: list[PairSpec] = []
    seen_pairs: set[tuple[int, int]] = set()
    physical = _physical_spec()
    for candidate_name, source, target in candidates:
        if (source, target) in seen_pairs:
            continue
        seen_pairs.add((source, target))
        pair_id = candidate_name
        witness = _direct_witness_for_pair(
            inputs,
            full_prepared,
            pair_id,
            source,
            target,
            physical,
            route_id_suffix="ordinary_shortest",
        )
        if witness is None:
            continue
        length = witness.metrics.length
        if pair_id != "paris_bures" and not (min_shortest_m <= length <= max_shortest_m):
            continue
        selected.append(
            PairSpec(
                pair_id=pair_id,
                source=source,
                target=target,
                shortest=witness,
                selection_note=(
                    "fixed Paris-Bures stress pair"
                    if pair_id == "paris_bures"
                    else f"deterministic candidate, shortest_length={length:.1f}"
                ),
            )
        )
        if len(selected) >= selection_pool_size:
            break
    if len(selected) < min(pair_count, 3):
        raise RuntimeError(
            f"only selected {len(selected)} connected pairs; need at least 3"
        )
    return selected


def _collect_pair_witness_pool(
    inputs: base.StaticInputs,
    full_prepared: PreparedGraph,
    pair: PairSpec,
    *,
    max_profile_vertices_per_scalar: int,
) -> tuple[list[WitnessRoute], dict[str, Any]]:
    probe = _loose_query_for_pair(
        pair.pair_id,
        pair.source,
        pair.target,
        pair.shortest.metrics,
    )
    specs = _benchmark_scalarizations(probe)
    witnesses: dict[tuple[int, ...], WitnessRoute] = {
        pair.shortest.edge_ids: pair.shortest
    }
    generation_runs: list[dict[str, Any]] = []

    for spec in specs:
        direct = _direct_witness_for_pair(
            inputs,
            full_prepared,
            pair.pair_id,
            pair.source,
            pair.target,
            spec,
            route_id_suffix="probe_direct",
        )
        if direct is not None:
            witnesses.setdefault(direct.edge_ids, direct)

    for spec in specs:
        start = time.perf_counter()
        reverse_adj = via._build_reverse_edge_adjacency(
            full_prepared.G,
            full_prepared.context.edge_mask,
        )
        forward = via._run_scalar_tree(
            full_prepared.G,
            probe,
            full_prepared.context,
            spec,
            reverse=False,
        )
        backward = via._run_scalar_tree(
            full_prepared.G,
            probe,
            full_prepared.context,
            spec,
            reverse=True,
            reverse_adj=reverse_adj,
        )
        profiles: list[via.ProfileCandidate] = []
        for via_vertex in range(full_prepared.G.n_nodes):
            metrics = via._combined_profile(probe, forward, backward, via_vertex)
            if metrics is None:
                continue
            if metrics.length > probe.Lmax + 1e-6:
                continue
            if metrics.length < max(1.0, 0.70 * pair.shortest.metrics.length):
                continue
            profiles.append(via.ProfileCandidate(via_vertex, metrics))

        selected: dict[int, via.ProfileCandidate] = {}

        def add_profiles(
            chosen: Iterable[via.ProfileCandidate],
            *,
            limit: int,
        ) -> None:
            added = 0
            for profile in chosen:
                if len(selected) >= max_profile_vertices_per_scalar:
                    return
                before = len(selected)
                selected.setdefault(int(profile.via_vertex), profile)
                if len(selected) > before:
                    added += 1
                    if added >= limit:
                        return

        if profiles:
            add_profiles(sorted(profiles, key=lambda p: p.metrics.length), limit=8)
            add_profiles(
                sorted(profiles, key=lambda p: (-p.metrics.elevation, p.metrics.length)),
                limit=12,
            )
            add_profiles(
                sorted(profiles, key=lambda p: (p.metrics.avg_width, p.metrics.length)),
                limit=12,
            )
            add_profiles(
                sorted(
                    profiles,
                    key=lambda p: (-p.metrics.avg_popularity, p.metrics.length),
                ),
                limit=12,
            )
            add_profiles(
                sorted(
                    profiles,
                    key=lambda p: (
                        -p.metrics.elevation,
                        p.metrics.avg_width,
                        -p.metrics.avg_popularity,
                        p.metrics.length,
                    ),
                ),
                limit=12,
            )
            target_len = 1.25 * pair.shortest.metrics.length
            add_profiles(
                sorted(
                    profiles,
                    key=lambda p: (
                        abs(p.metrics.length - target_len),
                        -p.metrics.elevation,
                        p.metrics.avg_width,
                    ),
                ),
                limit=12,
            )
        for special_via in (probe.source, probe.target):
            metrics = via._combined_profile(probe, forward, backward, special_via)
            if metrics is not None:
                selected.setdefault(
                    int(special_via),
                    via.ProfileCandidate(int(special_via), metrics),
                )
        if pair.pair_id == "paris_bures" and spec.name == SCALAR_REFERENCE:
            metrics = via._combined_profile(probe, forward, backward, 13683)
            if metrics is not None:
                selected.setdefault(13683, via.ProfileCandidate(13683, metrics))

        reconstructed = 0
        accepted = 0
        for profile in selected.values():
            reconstructed += 1
            attempt = _reconstruct_via_candidate(
                full_prepared,
                inputs.G,
                probe,
                forward,
                backward,
                profile,
            )
            if attempt.candidate is None or not attempt.candidate.elementary:
                continue
            result = via._make_path_result(
                attempt.candidate.scalar_cost,
                attempt.candidate.path_nodes,
                attempt.candidate.edge_ids,
                attempt.candidate.metrics,
            )
            witness = _make_witness_from_result(
                route_id=(
                    f"{pair.pair_id}:probe_via:{spec.name}:"
                    f"via{attempt.candidate.via_vertex}"
                ),
                pair_id=pair.pair_id,
                generator="same_scalar_via_profile_probe",
                scalar_name=spec.name,
                via_vertex=attempt.candidate.via_vertex,
                original_G=inputs.G,
                result=result,
            )
            if witness is None:
                continue
            if witness.edge_ids not in witnesses:
                accepted += 1
                witnesses[witness.edge_ids] = witness

        generation_runs.append(
            {
                "pair_id": pair.pair_id,
                "scalar_name": spec.name,
                "profiles_considered": len(profiles),
                "profiles_selected_for_reconstruction": len(selected),
                "profiles_reconstructed": reconstructed,
                "witnesses_accepted": accepted,
                "elapsed_s": time.perf_counter() - start,
                **_tie_totals(forward, backward),
            }
        )

    pool = list(witnesses.values())
    pool.sort(
        key=lambda witness: (
            -_route_difference_score(pair.shortest.metrics, witness.metrics),
            witness.metrics.length,
            witness.route_id,
        )
    )
    return pool, {"probe_query": probe.as_dict(), "runs": generation_runs}


def _route_difference_score(
    shortest: base.RouteMetrics,
    candidate: base.RouteMetrics,
) -> float:
    return max(
        abs(candidate.length - shortest.length) / max(shortest.length, 1.0),
        abs(candidate.elevation - shortest.elevation) / max(shortest.elevation, 25.0),
        abs(candidate.avg_popularity - shortest.avg_popularity)
        / max(abs(shortest.avg_popularity), 1.0),
        abs(candidate.avg_width - shortest.avg_width) / max(abs(shortest.avg_width), 1.0),
    )


def _clamped_box(
    witness: base.RouteMetrics,
    *,
    Lmin: float,
    Lmax: float,
    Hmin: float,
    Hmax: float,
    Pmin: float,
    Wmax: float,
) -> tuple[float, float, float, float, float, float]:
    Lmin = min(float(Lmin), witness.length - 1e-3)
    Lmax = max(float(Lmax), witness.length + 1e-3)
    Hmin = min(float(Hmin), witness.elevation - 1e-3)
    Hmax = max(float(Hmax), witness.elevation + 1e-3)
    Pmin = min(max(0.0, float(Pmin)), witness.avg_popularity - 1e-6)
    Wmax = max(float(Wmax), witness.avg_width + 1e-6)
    return max(0.0, Lmin), Lmax, max(0.0, Hmin), Hmax, Pmin, Wmax


def _query_from_template(
    *,
    pair: PairSpec,
    witness: WitnessRoute,
    template: str,
    index: int,
) -> tuple[base.QueryBox, str, str, tuple[str, ...], bool]:
    m = witness.metrics
    if template == "loose":
        Lpad = max(1000.0, 0.14 * m.length)
        Hpad = max(90.0, 0.30 * max(m.elevation, 1.0))
        values = _clamped_box(
            m,
            Lmin=m.length - Lpad,
            Lmax=m.length + Lpad,
            Hmin=m.elevation - Hpad,
            Hmax=m.elevation + Hpad,
            Pmin=m.avg_popularity - 35.0,
            Wmax=m.avg_width + 3.5,
        )
        category, tightness, tags, deliberate = "LOOSE_CONTROL", "loose", ("loose",), False
    elif template == "multi_tight":
        values = _clamped_box(
            m,
            Lmin=m.length - max(180.0, 0.035 * m.length),
            Lmax=m.length + max(220.0, 0.040 * m.length),
            Hmin=m.elevation - max(18.0, 0.075 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(22.0, 0.085 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 6.0,
            Wmax=m.avg_width + 0.75,
        )
        category = "MULTI_TIGHT"
        tightness = "tight"
        tags = ("MULTI_TIGHT", "POP_ACTIVE", "WIDTH_ACTIVE", "tight")
        deliberate = False
    elif template == "l_low_h_low":
        values = _clamped_box(
            m,
            Lmin=m.length - max(90.0, 0.010 * m.length),
            Lmax=m.length + max(1800.0, 0.10 * m.length),
            Hmin=m.elevation - max(4.0, 0.018 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(90.0, 0.24 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 25.0,
            Wmax=m.avg_width + 2.2,
        )
        category = "L_LOW_AND_H_LOW"
        tightness = "tight"
        tags = ("L_LOW_ACTIVE", "H_LOW_ACTIVE", "L_LOW_AND_H_LOW", "tight")
        deliberate = True
    elif template == "width_active":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1500.0, 0.10 * m.length),
            Lmax=m.length + max(3000.0, 0.16 * m.length),
            Hmin=m.elevation - max(80.0, 0.25 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(140.0, 0.35 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 30.0,
            Wmax=m.avg_width + 0.35,
        )
        category = "WIDTH_ACTIVE"
        tightness = "tight"
        tags = ("WIDTH_ACTIVE", "tight")
        deliberate = False
    elif template == "pop_active":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1500.0, 0.10 * m.length),
            Lmax=m.length + max(3000.0, 0.16 * m.length),
            Hmin=m.elevation - max(80.0, 0.25 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(140.0, 0.35 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 3.0,
            Wmax=m.avg_width + 2.2,
        )
        category = "POP_ACTIVE"
        tightness = "tight"
        tags = ("POP_ACTIVE", "tight")
        deliberate = False
    elif template == "quality_conflict":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1200.0, 0.08 * m.length),
            Lmax=m.length + max(2500.0, 0.13 * m.length),
            Hmin=m.elevation - max(70.0, 0.18 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(130.0, 0.28 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 4.0,
            Wmax=m.avg_width + 0.55,
        )
        category = "QUALITY_CONFLICT"
        tightness = "tight"
        tags = ("QUALITY_CONFLICT", "POP_ACTIVE", "WIDTH_ACTIVE", "tight")
        deliberate = False
    elif template == "h_high_active":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1300.0, 0.09 * m.length),
            Lmax=m.length + max(2400.0, 0.13 * m.length),
            Hmin=m.elevation - max(140.0, 0.45 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(8.0, 0.025 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 22.0,
            Wmax=m.avg_width + 2.0,
        )
        category = "H_HIGH_ACTIVE"
        tightness = "tight"
        tags = ("H_HIGH_ACTIVE", "tight")
        deliberate = False
    elif template == "l_high_active":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1800.0, 0.12 * m.length),
            Lmax=m.length + max(90.0, 0.018 * m.length),
            Hmin=m.elevation - max(90.0, 0.25 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(140.0, 0.35 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 24.0,
            Wmax=m.avg_width + 2.3,
        )
        category = "L_HIGH_ACTIVE"
        tightness = "tight"
        tags = ("L_HIGH_ACTIVE", "tight")
        deliberate = False
    else:
        raise ValueError(template)
    Lmin, Lmax, Hmin, Hmax, Pmin, Wmax = values
    query = base.QueryBox(
        name=f"{pair.pair_id}_{index:02d}_{template}",
        source=pair.source,
        target=pair.target,
        Lmin=Lmin,
        Lmax=Lmax,
        Hmin=Hmin,
        Hmax=Hmax,
        Pmin=Pmin,
        Wmax=Wmax,
    )
    return query, category, tightness, tags, deliberate


def _make_item(
    *,
    inputs: base.StaticInputs,
    pair: PairSpec,
    witness: WitnessRoute,
    query: base.QueryBox,
    category: str,
    tightness: str,
    tags: tuple[str, ...],
    deliberate_adversarial: bool,
) -> BenchmarkItem | None:
    dummy = BenchmarkItem(
        query=query,
        pair_id=pair.pair_id,
        category=category,
        tightness=tightness,
        tags=tags,
        witness=witness,
        physical_shortest=pair.shortest,
        shortest_violation_score=0.0,
        shortest_violations={},
        shortest_violated_constraints=(),
        adversarial=False,
        deliberate_adversarial=deliberate_adversarial,
    )
    try:
        return _validate_witness_for_query(inputs, dummy)
    except ValueError:
        return None


def _candidate_items_for_pair(
    inputs: base.StaticInputs,
    pair: PairSpec,
    witnesses: Sequence[WitnessRoute],
) -> list[BenchmarkItem]:
    templates = [
        "l_low_h_low",
        "multi_tight",
        "width_active",
        "pop_active",
        "quality_conflict",
        "h_high_active",
        "l_high_active",
        "loose",
    ]
    candidates: list[BenchmarkItem] = []
    index = 0

    if pair.pair_id == "paris_bures":
        known = next(
            (
                witness
                for witness in witnesses
                if witness.scalar_name == SCALAR_REFERENCE and witness.via_vertex == 13683
            ),
            None,
        )
        if known is not None:
            paris_query = base.QueryBox(
                name="paris_bures_known_feasible",
                source=base.PARIS_BURES_SOURCE,
                target=base.PARIS_BURES_TARGET,
                Lmin=base.PARIS_BURES_LMIN,
                Lmax=base.PARIS_BURES_LMAX,
                Hmin=base.PARIS_BURES_HMIN,
                Hmax=base.PARIS_BURES_HMAX,
                Pmin=base.PARIS_BURES_PMIN,
                Wmax=base.PARIS_BURES_WMAX,
            )
            item = _make_item(
                inputs=inputs,
                pair=pair,
                witness=known,
                query=paris_query,
                category="L_LOW_AND_H_LOW",
                tightness="tight",
                tags=(
                    "L_LOW_ACTIVE",
                    "H_LOW_ACTIVE",
                    "L_LOW_AND_H_LOW",
                    "POP_ACTIVE",
                    "WIDTH_ACTIVE",
                    "QUALITY_CONFLICT",
                    "MULTI_TIGHT",
                    "tight",
                ),
                deliberate_adversarial=True,
            )
            if item is not None:
                candidates.append(item)

    for witness in witnesses:
        if witness.metrics.length <= 0.0:
            continue
        for template in templates:
            index += 1
            query, category, tightness, tags, deliberate = _query_from_template(
                pair=pair,
                witness=witness,
                template=template,
                index=index,
            )
            item = _make_item(
                inputs=inputs,
                pair=pair,
                witness=witness,
                query=query,
                category=category,
                tightness=tightness,
                tags=tags,
                deliberate_adversarial=deliberate,
            )
            if item is not None:
                candidates.append(item)
    return candidates


def _select_items_for_pair(
    candidates: Sequence[BenchmarkItem],
    *,
    limit: int,
) -> list[BenchmarkItem]:
    forced = [
        item for item in candidates if item.query.name == "paris_bures_known_feasible"
    ]
    selected: list[BenchmarkItem] = forced[:limit]
    used_categories: set[str] = set()
    used_names = {item.query.name for item in selected}
    used_categories.update(item.category for item in selected)
    loose = sorted(
        [
            item
            for item in candidates
            if item.tightness == "loose" and item.query.name not in used_names
        ],
        key=lambda item: (-item.shortest_violation_score, item.query.name),
    )
    if loose and len(selected) < limit:
        selected.append(loose[0])
        used_names.add(loose[0].query.name)
        used_categories.add(loose[0].category)
    ordered = sorted(
        [item for item in candidates if item.query.name not in used_names],
        key=lambda item: (
            not item.deliberate_adversarial,
            -item.shortest_violation_score,
            item.category in used_categories,
            item.query.name,
        ),
    )
    for item in ordered:
        if len(selected) >= limit:
            break
        if item.category in used_categories and len(used_categories) < limit:
            continue
        selected.append(item)
        used_names.add(item.query.name)
        used_categories.add(item.category)
    if len(selected) < limit:
        for item in ordered:
            if len(selected) >= limit:
                break
            if item.query.name in used_names:
                continue
            selected.append(item)
            used_names.add(item.query.name)
    return selected


def _auto_generate_benchmark_items(
    inputs: base.StaticInputs,
    global_constants: base.MetricConstants,
    args: argparse.Namespace,
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    full_mask = np.ones(inputs.G.n_edges, dtype=bool)
    full_context = base.GraphContext(
        mode="full",
        edge_mask=full_mask,
        node_count=inputs.G.n_nodes,
        edge_count=inputs.G.n_edges,
        constants=global_constants,
        metadata={"rho_H_policy": "global_full_graph_reused_across_modes"},
    )
    full_prepared = PreparedGraph(
        mode="full",
        G=inputs.G,
        context=full_context,
        edge_id_to_original=np.arange(inputs.G.n_edges, dtype=np.int32),
        graph_prep_s=0.0,
        corridor_construction_s=0.0,
        compaction_s=0.0,
        metadata={"graph_storage": "native_full_csr"},
    )
    pair_specs = _select_pair_specs(
        inputs,
        full_prepared,
        pair_count=args.pair_count,
        max_candidate_pairs=args.max_candidate_pairs,
        seed=args.seed,
        min_shortest_m=args.min_shortest_m,
        max_shortest_m=args.max_shortest_m,
    )
    per_pair_limit = max(4, math.ceil(args.max_boxes / max(args.pair_count, 1)))
    items: list[BenchmarkItem] = []
    pair_metadata: list[dict[str, Any]] = []
    additional_adversarial_found = False

    for pair in pair_specs:
        pool, generation_meta = _collect_pair_witness_pool(
            inputs,
            full_prepared,
            pair,
            max_profile_vertices_per_scalar=args.max_profile_vertices_per_scalar,
        )
        candidates = _candidate_items_for_pair(inputs, pair, pool)
        selected = _select_items_for_pair(candidates, limit=per_pair_limit)
        if pair.pair_id != "paris_bures" and any(
            item.deliberate_adversarial and item.shortest_violation_score > 1e-6
            for item in selected
        ):
            additional_adversarial_found = True
        items.extend(selected)
        pair_metadata.append(
            {
                "pair": {
                    "pair_id": pair.pair_id,
                    "source": pair.source,
                    "target": pair.target,
                    "selection_note": pair.selection_note,
                    "ordinary_shortest": pair.shortest.as_dict(include_paths=False),
                },
                "witness_pool_size": len(pool),
                "candidate_box_count": len(candidates),
                "selected_box_count": len(selected),
                "selected_query_ids": [item.query.name for item in selected],
                "generation": generation_meta,
            }
        )
        if len(items) >= args.max_boxes and additional_adversarial_found:
            break

    if len(items) > args.max_boxes:
        items = items[: args.max_boxes]
    distinct_pairs = {item.pair_id for item in items}
    if len(distinct_pairs) < 3:
        raise RuntimeError(
            f"auto benchmark produced only {len(distinct_pairs)} distinct pairs"
        )
    if not additional_adversarial_found:
        raise RuntimeError(
            "auto benchmark did not find an additional non-Paris adversarial pair"
        )
    validated = [_validate_witness_for_query(inputs, item) for item in items]
    return validated, {
        "mode": "auto_generated_from_validated_scalar_direct_and_same_scalar_via_routes",
        "seed": args.seed,
        "pair_count_requested": args.pair_count,
        "max_boxes_requested": args.max_boxes,
        "boxes_per_pair_limit": per_pair_limit,
        "distinct_pairs": sorted(distinct_pairs),
        "pair_generation": pair_metadata,
        "witness_policy": (
            "candidate routes are reconstructed as directed CSR paths, "
            "metrics are recomputed from base CSR edge rows, and each generated "
            "box is retained only if its witness is elementary and satisfies it"
        ),
    }


def _load_items_from_boxes_json(
    inputs: base.StaticInputs,
    path: str,
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_items = payload["boxes"] if isinstance(payload, dict) and "boxes" in payload else payload
    if not isinstance(raw_items, list):
        raise ValueError("--boxes-json must contain a list or an object with a boxes list")
    items: list[BenchmarkItem] = []
    shortest_by_pair: dict[tuple[str, int, int], WitnessRoute] = {}
    full_prepared = PreparedGraph(
        mode="full",
        G=inputs.G,
        context=base.GraphContext(
            mode="full",
            edge_mask=np.ones(inputs.G.n_edges, dtype=bool),
            node_count=inputs.G.n_nodes,
            edge_count=inputs.G.n_edges,
            constants=base._metric_constants(inputs.G, np.ones(inputs.G.n_edges, dtype=bool)),
            metadata={},
        ),
        edge_id_to_original=np.arange(inputs.G.n_edges, dtype=np.int32),
        graph_prep_s=0.0,
        corridor_construction_s=0.0,
        compaction_s=0.0,
        metadata={"graph_storage": "native_full_csr"},
    )
    for idx, raw in enumerate(raw_items):
        query_raw = raw.get("query", raw)
        query = base.QueryBox(
            name=str(query_raw.get("name", query_raw.get("query_id", f"box_{idx:03d}"))),
            source=int(query_raw["source"]),
            target=int(query_raw["target"]),
            Lmin=float(query_raw["Lmin"]),
            Lmax=float(query_raw["Lmax"]),
            Hmin=float(query_raw["Hmin"]),
            Hmax=float(query_raw["Hmax"]),
            Pmin=float(query_raw["Pmin"]),
            Wmax=float(query_raw["Wmax"]),
            corridor_slack_m=int(
                query_raw.get("corridor_slack_m", base.CORRIDOR_SLACK_M)
            ),
            max_hops_from_boundary=int(
                query_raw.get("max_hops_from_boundary", base.MAX_HOPS_FROM_BOUNDARY)
            ),
        )
        pair_id = str(raw.get("pair_id", f"{query.source}_{query.target}"))
        witness_raw = raw.get("witness", raw)
        edge_ids = witness_raw.get("csr_edge_ids") or witness_raw.get("edge_ids")
        if edge_ids is None:
            raise ValueError(f"{query.name}: boxes-json item has no witness edge path")
        result = _result_from_original_edges(
            inputs.G,
            query.source,
            query.target,
            [int(edge_id) for edge_id in edge_ids],
        )
        if result is None:
            raise ValueError(f"{query.name}: witness edge path is invalid")
        witness = _make_witness_from_result(
            route_id=str(witness_raw.get("route_id", f"{query.name}:witness")),
            pair_id=pair_id,
            generator=str(witness_raw.get("generator", "boxes_json")),
            scalar_name=str(witness_raw.get("scalar_name", "unknown")),
            via_vertex=witness_raw.get("via_vertex"),
            original_G=inputs.G,
            result=result,
        )
        if witness is None:
            raise ValueError(f"{query.name}: witness failed validation")
        shortest_key = (pair_id, query.source, query.target)
        shortest = shortest_by_pair.get(shortest_key)
        if shortest is None:
            shortest = _direct_witness_for_pair(
                inputs,
                full_prepared,
                pair_id,
                query.source,
                query.target,
                _physical_spec(),
                route_id_suffix="ordinary_shortest",
            )
            if shortest is None:
                raise ValueError(f"{query.name}: ordinary shortest path not found")
            shortest_by_pair[shortest_key] = shortest
        item = BenchmarkItem(
            query=query,
            pair_id=pair_id,
            category=str(raw.get("category", "UNSPECIFIED")),
            tightness=str(raw.get("tightness", "unspecified")),
            tags=tuple(str(tag) for tag in raw.get("tags", ())),
            witness=witness,
            physical_shortest=shortest,
            shortest_violation_score=0.0,
            shortest_violations={},
            shortest_violated_constraints=(),
            adversarial=False,
            deliberate_adversarial=bool(raw.get("deliberate_adversarial", False)),
        )
        items.append(_validate_witness_for_query(inputs, item))
    return items, {"mode": "loaded_from_boxes_json", "path": path, "box_count": len(items)}


def _requested_modes(args: argparse.Namespace) -> set[str]:
    modes: set[str] = set()
    for raw in args.graph_modes.split(","):
        item = raw.strip()
        if not item:
            continue
        if item == "certified":
            item = "certified_masked"
        modes.add(item)
    if "full" not in modes:
        modes.add("full")
    allowed = {"full", "certified_masked", "certified_compact"}
    unknown = modes - allowed
    if unknown:
        raise ValueError(f"unknown graph modes: {sorted(unknown)}")
    return modes


def _is_paris_bures_item(item: BenchmarkItem) -> bool:
    return item.query.source == base.PARIS_BURES_SOURCE and item.query.target == base.PARIS_BURES_TARGET


def _run_benchmark(
    inputs: base.StaticInputs,
    items: Sequence[BenchmarkItem],
    global_constants: base.MetricConstants,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modes = _requested_modes(args)
    rows: list[dict[str, Any]] = []
    graph_comparisons: list[dict[str, Any]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        print(
            f"[{index}/{total}] {item.query.name} "
            f"({item.query.source}->{item.query.target})",
            flush=True,
        )
        include_geometric = args.include_geometric_paris and _is_paris_bures_item(item)
        prepared_by_mode, comparison = _build_prepared_graphs(
            inputs,
            item.query,
            global_constants,
            modes,
            include_geometric=include_geometric,
        )
        graph_comparisons.append({"query_id": item.query.name, **comparison})
        run_modes = [mode for mode in ("full", "certified_masked", "certified_compact") if mode in modes]
        if include_geometric:
            run_modes.append("geometric_diagnostic")
        for mode in run_modes:
            prepared = prepared_by_mode[mode]
            artifacts_by_scalar: dict[str, dict[str, Any]] = {}
            for spec in _benchmark_scalarizations(item.query):
                direct_row = _run_direct(inputs, item, prepared, spec)
                rows.append(direct_row)
                via_row, artifact = _run_via(inputs, item, prepared, spec)
                rows.append(via_row)
                artifacts_by_scalar[spec.name] = artifact
            rows.append(
                _make_union_row(
                    item,
                    prepared,
                    UNION_2_NAME,
                    (SCALAR_REFERENCE, SCALAR_SLOPE),
                    artifacts_by_scalar,
                )
            )
            rows.append(
                _make_union_row(
                    item,
                    prepared,
                    UNION_3_NAME,
                    BENCHMARK_SCALAR_NAMES,
                    artifacts_by_scalar,
                )
            )
    return rows, graph_comparisons


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[idx]


def _time_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _method_key(row: dict[str, Any]) -> str:
    if row["method"] == "via_union":
        return str(row["scalar_name"])
    return f"{row['scalar_name']} {row['method']}"


def _success_rates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for graph_mode in sorted({str(row["graph_mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["graph_mode"] == graph_mode]
        mode_out: dict[str, Any] = {}
        for key in sorted({_method_key(row) for row in mode_rows}):
            keyed = [row for row in mode_rows if _method_key(row) == key]
            attempted = len(keyed)
            solved = sum(1 for row in keyed if bool(row["feasible"]))
            mode_out[key] = {
                "solved": solved,
                "attempted": attempted,
                "percent": 100.0 * solved / attempted if attempted else None,
            }
        out[graph_mode] = mode_out
    return out


def _time_to_first_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for graph_mode in sorted({str(row["graph_mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["graph_mode"] == graph_mode]
        mode_out: dict[str, Any] = {}
        for key in sorted({_method_key(row) for row in mode_rows}):
            values = [
                float(row["time_to_first_feasible"])
                for row in mode_rows
                if _method_key(row) == key and row.get("time_to_first_feasible") not in (None, "")
            ]
            mode_out[key] = _time_stats(values)
        out[graph_mode] = mode_out
    return out


def _success_by_category(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    full_rows = [row for row in rows if row["graph_mode"] == "full"]
    tags = sorted(
        {
            tag
            for row in full_rows
            for tag in str(row.get("tags", "")).split("|")
            if tag
        }
    )
    for key in sorted({_method_key(row) for row in full_rows}):
        key_rows = [row for row in full_rows if _method_key(row) == key]
        tag_out: dict[str, Any] = {}
        for tag in tags:
            tagged = [row for row in key_rows if tag in str(row.get("tags", "")).split("|")]
            if not tagged:
                continue
            solved = sum(1 for row in tagged if bool(row["feasible"]))
            tag_out[tag] = {"solved": solved, "attempted": len(tagged)}
        out[key] = tag_out
    return out


def _failure_diagnostics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if row["graph_mode"] != "full" or bool(row["feasible"]):
            continue
        nearest = row.get("nearest_exact_candidate") or row.get("nearest_profile")
        failures.append(
            {
                "query_id": row["query_id"],
                "method": _method_key(row),
                "category": row["category"],
                "tags": row["tags"],
                "best_normalized_violation": row.get("normalized_violation_score"),
                "violated_constraints": row.get("violated_constraints"),
                "profile_feasible_count": row.get("profile_feasible_count"),
                "exact_feasible_count": row.get("exact_feasible_count"),
                "non_elementary_count": row.get("non_elementary_count"),
                "nearest_candidate_or_profile": nearest,
            }
        )
    failures.sort(
        key=lambda item: (
            str(item["query_id"]),
            str(item["method"]),
        )
    )
    return failures


def _tie_summary(rows: Sequence[dict[str, Any]], threshold: int) -> dict[str, Any]:
    via_rows = [row for row in rows if row["method"] == "via"]
    equal_cost = [int(row.get("equal_cost_relaxations") or 0) for row in via_rows]
    resource_distinct = [
        int(row.get("equal_cost_resource_distinct_relaxations") or 0)
        for row in via_rows
    ]
    parent_changes = [
        int(row.get("parent_changes_due_to_road_tiebreak") or 0) for row in via_rows
    ]
    substantial_queries = sorted(
        {
            str(row["query_id"])
            for row in via_rows
            if int(row.get("equal_cost_resource_distinct_relaxations") or 0)
            >= threshold
        }
    )
    return {
        "via_rows": len(via_rows),
        "threshold_for_substantial_resource_distinct_ties": threshold,
        "equal_cost_relaxations": _time_stats([float(v) for v in equal_cost]),
        "equal_cost_resource_distinct_relaxations": _time_stats(
            [float(v) for v in resource_distinct]
        ),
        "parent_changes_due_to_road_tiebreak": _time_stats(
            [float(v) for v in parent_changes]
        ),
        "queries_with_substantial_resource_distinct_ties": substantial_queries,
        "query_count_with_substantial_resource_distinct_ties": len(substantial_queries),
    }


def _certified_slowdown_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paris_rows = [
        row
        for row in rows
        if str(row["query_id"]).startswith("paris_bures")
        and row["method"] == "via"
        and row["scalar_name"] == SCALAR_REFERENCE
        and row["graph_mode"] in {"full", "certified_masked", "certified_compact"}
    ]
    out: dict[str, Any] = {
        "implementation_finding": (
            "certified_masked uses the full CSR with an edge-mask/filter test; "
            "certified_compact materializes kept edges into a smaller CSR over the same node ids"
        ),
        "paris_bures_pop_width_reference_rows": [],
    }
    for row in paris_rows:
        edge_scans = int(row.get("edge_scans") or 0)
        raw_rows = int(row.get("raw_edge_rows_checked") or 0)
        out["paris_bures_pop_width_reference_rows"].append(
            {
                "query_id": row["query_id"],
                "graph_mode": row["graph_mode"],
                "graph_edges": row["graph_edge_count"],
                "forward_raw_edge_rows_checked": row.get("forward_raw_edge_rows_checked"),
                "forward_edge_scans": row.get("forward_edge_scans"),
                "raw_to_kept_scan_ratio": raw_rows / max(edge_scans, 1),
                "exhaustive_total_s": row.get("exhaustive_total_s"),
                "first_hit_total_s": row.get("first_hit_total_s"),
                "corridor_construction_s": row.get("corridor_construction_s"),
                "compaction_s": row.get("compaction_s"),
            }
        )
    return out


def _empirical_hmax_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] != "via":
            continue
        empirical = row.get("empirical_via_Hmax")
        if not isinstance(empirical, dict):
            continue
        profile = empirical.get("profile_metrics") or {}
        out.append(
            {
                "query_id": row["query_id"],
                "graph_mode": row["graph_mode"],
                "scalar_name": row["scalar_name"],
                "via_vertex": empirical.get("via_vertex"),
                "profile_L": profile.get("length"),
                "profile_H": profile.get("elevation"),
                "reconstructed": empirical.get("reconstructed"),
                "elementary": empirical.get("elementary"),
                "exact_H": empirical.get("exact_H"),
            }
        )
    out.sort(
        key=lambda item: (
            str(item["query_id"]),
            str(item["graph_mode"]),
            str(item["scalar_name"]),
        )
    )
    return out


def _build_summary(
    rows: Sequence[dict[str, Any]],
    items: Sequence[BenchmarkItem],
    *,
    tie_threshold: int,
) -> dict[str, Any]:
    return {
        "box_count": len(items),
        "distinct_pair_count": len({item.pair_id for item in items}),
        "distinct_pairs": sorted({item.pair_id for item in items}),
        "deliberately_adversarial_queries": [
            {
                "query_id": item.query.name,
                "pair_id": item.pair_id,
                "shortest_violation_score": item.shortest_violation_score,
                "shortest_violated_constraints": list(
                    item.shortest_violated_constraints
                ),
            }
            for item in items
            if item.deliberate_adversarial and item.shortest_violation_score > 1e-6
        ],
        "success_rate_by_method": _success_rates(rows),
        "time_to_first_feasible": _time_to_first_summary(rows),
        "success_by_constraint_category_full_graph": _success_by_category(rows),
        "failure_diagnostics_full_graph": _failure_diagnostics(rows),
        "tie_ambiguity_summary": _tie_summary(rows, tie_threshold),
        "certified_slowdown_diagnostic": _certified_slowdown_diagnostic(rows),
        "empirical_via_Hmax_rows": _empirical_hmax_summary(rows),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nBenchmark summary", flush=True)
    print(
        f"  boxes={summary['box_count']} "
        f"pairs={summary['distinct_pair_count']} "
        f"pair_ids={', '.join(summary['distinct_pairs'])}",
        flush=True,
    )
    full = summary["success_rate_by_method"].get("full", {})
    for key in [
        f"{SCALAR_PHYSICAL} direct",
        f"{SCALAR_PHYSICAL} via",
        f"{SCALAR_REFERENCE} direct",
        f"{SCALAR_REFERENCE} via",
        f"{SCALAR_SLOPE} direct",
        f"{SCALAR_SLOPE} via",
        UNION_2_NAME,
        UNION_3_NAME,
    ]:
        item = full.get(key)
        if not item:
            continue
        pct = item["percent"]
        print(
            f"  full {key}: {item['solved']}/{item['attempted']} "
            f"({pct:.1f}%)",
            flush=True,
        )
    time_full = summary["time_to_first_feasible"].get("full", {})
    for key in [f"{SCALAR_PHYSICAL} via", f"{SCALAR_REFERENCE} via", f"{SCALAR_SLOPE} via", UNION_2_NAME]:
        stats = time_full.get(key)
        if not stats or stats["count"] == 0:
            continue
        print(
            f"  first-hit {key}: median={stats['median']:.4f}s "
            f"p90={stats['p90']:.4f}s max={stats['max']:.4f}s",
            flush=True,
        )
    failures = [
        item
        for item in summary["failure_diagnostics_full_graph"]
        if item["method"] == UNION_2_NAME
    ]
    print(f"  full {UNION_2_NAME} failures={len(failures)}", flush=True)
    tie = summary["tie_ambiguity_summary"]
    print(
        "  tie resource-distinct median="
        f"{tie['equal_cost_resource_distinct_relaxations']['median']} "
        f"max={tie['equal_cost_resource_distinct_relaxations']['max']} "
        "substantial_queries="
        f"{tie['query_count_with_substantial_resource_distinct_ties']}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct scalar shortest paths against same-scalar via-node "
            "composition on known-feasible four-criterion boxes."
        )
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--boxes-json", default=None)
    parser.add_argument("--boxes-output-json", default=DEFAULT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-boxes", type=int, default=24)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--max-candidate-pairs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--min-shortest-m", type=float, default=6000.0)
    parser.add_argument("--max-shortest-m", type=float, default=52000.0)
    parser.add_argument("--max-profile-vertices-per-scalar", type=int, default=60)
    parser.add_argument(
        "--graph-modes",
        default="full,certified_masked,certified_compact",
        help=(
            "Comma-separated subset of full, certified_masked, certified_compact. "
            "The alias certified means certified_masked."
        ),
    )
    parser.add_argument(
        "--include-geometric-paris",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the old geometric corridor only on Paris-Bures diagnostic boxes.",
    )
    parser.add_argument(
        "--include-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include accepted benchmark run paths in JSON rows. Witness paths are always written in boxes JSON.",
    )
    parser.add_argument("--tie-substantial-threshold", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = _compute_global_metric_constants(inputs)
    if args.boxes_json:
        items, generation = _load_items_from_boxes_json(inputs, args.boxes_json)
    else:
        items, generation = _auto_generate_benchmark_items(
            inputs,
            global_constants,
            args,
        )
    _write_json(
        args.boxes_output_json,
        {
            "boxes": [item.as_dict(include_paths=True) for item in items],
            "generation": generation,
            "rho_H_global": rho_info,
        },
    )
    print(
        f"Witness validation passed for {len(items)} benchmark boxes; "
        f"boxes written to {args.boxes_output_json}",
        flush=True,
    )
    rows, graph_comparisons = _run_benchmark(inputs, items, global_constants, args)
    summary = _build_summary(
        rows,
        items,
        tie_threshold=args.tie_substantial_threshold,
    )
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_path": args.graph_path,
            "scalarizations": list(BENCHMARK_SCALAR_NAMES),
            "union_2": [SCALAR_REFERENCE, SCALAR_SLOPE],
            "union_3": list(BENCHMARK_SCALAR_NAMES),
            "rho_H_global": rho_info,
            "road_id_mode": (
                "normal vertex-state Dijkstra; road_id only breaks exact scalar ties"
            ),
            "via_first_hit_order": (
                "profile-feasible candidates sorted by existing box-centered score, "
                "then reconstructed until first exact elementary feasible route"
            ),
            "empirical_via_Hmax_warning": (
                "The statistic is measured only over the same-scalar via profile "
                "family with L_v <= Lmax; it is not a certified global envelope."
            ),
            "graph_modes_requested": sorted(_requested_modes(args)),
            "include_geometric_paris": args.include_geometric_paris,
        },
        "generation": generation,
        "queries": [item.as_dict(include_paths=False) for item in items],
        "graph_comparisons": graph_comparisons,
        "rows": rows,
        "summary": summary,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    _print_summary(summary)
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
