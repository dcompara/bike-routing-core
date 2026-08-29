from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import logging
import time
from typing import Any, Iterable, Sequence

from brcore.algo import params
import brcore.algo.portal_search as ps
from brcore.algo.portal_search import (
    ArchiveEntry,
    ConstraintBox,
    OverlayEdge,
    RouteAccumulator,
    SparsePortalConfig,
    SparsePortalQuery,
    anytime_sparse_portal_search,
)

import article_paris_bures as article
import article_paris_bures_longrun as longrun


CHECKPOINTS_S = (
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
    1800.0,
    3600.0,
)
DEFAULT_OUTPUT_JSON = "tmp_paris_bures_compression_sweep.json"
DEFAULT_BASELINE_JSON = "tmp_paris_bures_2h_longrun.json"
ORACLE_CELLS = set(longrun.ORACLE_CELLS)
ORACLE_ACTIVE_NODES = longrun.ORACLE_ACTIVE_NODES


@dataclass(frozen=True)
class CompressionVariant:
    name: str
    overrides: dict[str, int]
    combination: bool = False


ONE_FACTOR_VARIANTS: tuple[CompressionVariant, ...] = (
    CompressionVariant("LABEL8", {"max_labels_per_portal": 8}),
    CompressionVariant("LABEL16", {"max_labels_per_portal": 16}),
    CompressionVariant("PORTAL8", {"max_active_portals_per_cell": 8}),
    CompressionVariant("SHORTCUT4", {"max_shortcuts_per_pair": 4}),
    CompressionVariant("BRIDGE4", {"max_bridge_edges_per_cell_pair": 4}),
)
COMBINATION_VARIANTS: tuple[CompressionVariant, ...] = (
    CompressionVariant(
        "LABEL8_SHORTCUT4",
        {"max_labels_per_portal": 8, "max_shortcuts_per_pair": 4},
        combination=True,
    ),
    CompressionVariant(
        "PORTAL8_LABEL8_SHORTCUT4",
        {
            "max_active_portals_per_cell": 8,
            "max_labels_per_portal": 8,
            "max_shortcuts_per_pair": 4,
        },
        combination=True,
    ),
)


@dataclass
class RunTelemetry:
    max_queue_sizes: dict[str, int]
    max_retained_labels: dict[str, int]
    max_representative_portals: dict[str, int]
    edge_attempts: dict[str, int]
    edge_insertions: dict[str, int]
    edge_duplicate_rejections: dict[str, int]
    edge_capacity_rejections: dict[str, int]
    edge_capacity_replacements: dict[str, int]
    first_route_time_s: float | None = None

    @classmethod
    def create(cls) -> "RunTelemetry":
        return cls(
            max_queue_sizes={"fwd": 0, "bwd": 0},
            max_retained_labels={"fwd": 0, "bwd": 0},
            max_representative_portals={"fwd": 0, "bwd": 0},
            edge_attempts={},
            edge_insertions={},
            edge_duplicate_rejections={},
            edge_capacity_rejections={},
            edge_capacity_replacements={},
        )

    def update_state(self, state: Any) -> None:
        for direction in ("fwd", "bwd"):
            self.max_queue_sizes[direction] = max(
                self.max_queue_sizes[direction],
                len(state.frontier[direction]),
            )
            self.max_retained_labels[direction] = max(
                self.max_retained_labels[direction],
                _retained_label_count(state, direction),
            )
            self.max_representative_portals[direction] = max(
                self.max_representative_portals[direction],
                _covered_portal_count(state, direction),
            )

    def record_edge_attempt(
        self,
        state: Any,
        edge: OverlayEdge,
        *,
        inserted: bool,
        duplicate: bool,
        capacity_limited: bool,
        capacity_replaced: bool,
    ) -> None:
        kind = _edge_kind_bucket(edge)
        _increment(self.edge_attempts, kind)
        if inserted:
            _increment(self.edge_insertions, kind)
        elif duplicate:
            _increment(self.edge_duplicate_rejections, kind)
        elif capacity_limited:
            _increment(self.edge_capacity_rejections, kind)
        if capacity_replaced:
            _increment(self.edge_capacity_replacements, kind)


@dataclass
class SweepContext:
    G: Any
    nodes: Any
    kept_cells: list[int]
    partition: dict[int, int]
    boundary_nodes: set[int]
    kept_nodes: set[int]
    constraints: ConstraintBox
    baseline_initial_active_portals: set[int]


def _increment(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def _edge_kind_bucket(edge: OverlayEdge) -> str:
    if edge.bridge_cell_pair is not None:
        return "bridge"
    return edge.kind


def _retained_label_count(state: Any, direction: str) -> int:
    return sum(len(labels) for labels in state.labels[direction].values())


def _covered_portal_count(state: Any, direction: str) -> int:
    return sum(1 for labels in state.labels[direction].values() if labels)


def _covered_portals(state: Any, direction: str) -> set[int]:
    return {
        int(portal)
        for portal, labels in state.labels[direction].items()
        if labels
    }


def _covered_cells(state: Any, direction: str) -> list[int]:
    return sorted({state.trace_cell(portal) for portal in _covered_portals(state, direction)})


def _overlay_edge_count(state: Any) -> int:
    return sum(len(edges) for edges in state.overlay.out_edges.values())


def _config_dict(config: SparsePortalConfig) -> dict[str, Any]:
    return {
        "max_active_portals_per_cell": config.max_active_portals_per_cell,
        "max_labels_per_portal": config.max_labels_per_portal,
        "max_shortcuts_per_pair": config.max_shortcuts_per_pair,
        "local_expand_limit": config.local_expand_limit,
        "advance_round_budget": config.advance_round_budget,
        "max_cell_visits_per_route": config.max_cell_visits_per_route,
        "max_bridge_edges_per_cell_pair": config.max_bridge_edges_per_cell_pair,
        "max_backward_directional_repairs_per_cell": (
            config.max_backward_directional_repairs_per_cell
        ),
        "max_forward_directional_repairs_per_cell": (
            config.max_forward_directional_repairs_per_cell
        ),
        "bridge_refinement_scan_limit": config.bridge_refinement_scan_limit,
        "validate_incremental_bridge_detection": (
            config.validate_incremental_bridge_detection
        ),
    }


def _base_config() -> SparsePortalConfig:
    return article._make_config()


def _variant_config(variant: CompressionVariant) -> SparsePortalConfig:
    return replace(_base_config(), **variant.overrides)


def _make_query(time_budget_s: float, constraints: ConstraintBox) -> SparsePortalQuery:
    return SparsePortalQuery(
        source=article.SOURCE,
        target=article.TARGET,
        constraints=constraints,
        time_budget_s=time_budget_s,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )


def _initial_active_portals(context: SweepContext, config: SparsePortalConfig) -> set[int]:
    return ps._select_active_portals(
        context.G,
        context.boundary_nodes,
        context.partition,
        context.kept_nodes,
        article.SOURCE,
        article.TARGET,
        config.max_active_portals_per_cell,
    )


def _cell_counts(nodes: Iterable[int], partition: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for node in nodes:
        cell = partition.get(int(node), -(int(node) + 1))
        out[cell] = out.get(cell, 0) + 1
    return dict(sorted(out.items()))


def _route_validation_summaries(
    context: SweepContext,
    state: Any,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for entry in state.archive.entries:
        validation = article._validate_route(
            context.G,
            context.partition,
            context.constraints,
            entry,
            state.config.max_cell_visits_per_route,
        )
        summaries.append(longrun._archive_entry_summary(entry, validation))
    return summaries


def _label_summary(label: Any) -> dict[str, Any]:
    x = label.metrics.route_vector()
    return {
        "portal": int(label.portal),
        "priority": float(label.priority),
        "length": float(x[0]),
        "elevation": float(x[1]),
        "avg_pop": float(x[2]),
        "avg_width": float(x[3]),
        "road_changes": int(label.road_changes),
        "visited_cells": sorted(int(cell) for cell in label.visited_cells),
        "revisited_cells": sorted(int(cell) for cell in label.revisited_cells),
    }


def _label_capacity_diagnostics(state: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    portals_using_gt4 = 0
    total_labels_above_4 = 0
    outside_priority_top4_has_lower_width = 0
    outside_priority_top4_has_higher_pop = 0

    for direction in ("fwd", "bwd"):
        for portal, labels in sorted(state.labels[direction].items()):
            if len(labels) <= 4:
                continue
            portals_using_gt4 += 1
            total_labels_above_4 += len(labels) - 4
            sorted_by_priority = sorted(labels, key=lambda label: label.priority)
            top4 = sorted_by_priority[:4]
            extra = sorted_by_priority[4:]
            top4_min_width = min(label.metrics.route_vector()[3] for label in top4)
            top4_max_pop = max(label.metrics.route_vector()[2] for label in top4)
            extra_min_width = min(label.metrics.route_vector()[3] for label in extra)
            extra_max_pop = max(label.metrics.route_vector()[2] for label in extra)
            if extra_min_width < top4_min_width:
                outside_priority_top4_has_lower_width += 1
            if extra_max_pop > top4_max_pop:
                outside_priority_top4_has_higher_pop += 1
            all_vecs = [label.metrics.route_vector() for label in labels]
            rows.append(
                {
                    "direction": direction,
                    "portal": int(portal),
                    "cell": int(state.trace_cell(portal)),
                    "label_count": len(labels),
                    "length_range": [
                        float(min(x[0] for x in all_vecs)),
                        float(max(x[0] for x in all_vecs)),
                    ],
                    "elevation_range": [
                        float(min(x[1] for x in all_vecs)),
                        float(max(x[1] for x in all_vecs)),
                    ],
                    "avg_pop_range": [
                        float(min(x[2] for x in all_vecs)),
                        float(max(x[2] for x in all_vecs)),
                    ],
                    "avg_width_range": [
                        float(min(x[3] for x in all_vecs)),
                        float(max(x[3] for x in all_vecs)),
                    ],
                    "road_changes_range": [
                        int(min(label.road_changes for label in labels)),
                        int(max(label.road_changes for label in labels)),
                    ],
                    "extra_min_width_below_priority_top4": bool(
                        extra_min_width < top4_min_width
                    ),
                    "extra_max_pop_above_priority_top4": bool(
                        extra_max_pop > top4_max_pop
                    ),
                    "labels": [_label_summary(label) for label in sorted_by_priority[:8]],
                }
            )

    rows.sort(key=lambda row: (-int(row["label_count"]), row["direction"], row["portal"]))
    return {
        "portals_using_more_than_4_labels": portals_using_gt4,
        "total_labels_above_4": total_labels_above_4,
        "portals_where_extra_label_is_narrower_than_priority_top4": (
            outside_priority_top4_has_lower_width
        ),
        "portals_where_extra_label_is_more_popular_than_priority_top4": (
            outside_priority_top4_has_higher_pop
        ),
        "examples": rows[:20],
    }


def _edge_metric_summary(edge: OverlayEdge) -> dict[str, Any]:
    x = edge.metrics.route_vector()
    return {
        "src": int(edge.src),
        "dst": int(edge.dst),
        "src_dst": f"{edge.src}->{edge.dst}",
        "kind": _edge_kind_bucket(edge),
        "bridge_cell_pair": (
            None
            if edge.bridge_cell_pair is None
            else list(edge.bridge_cell_pair)
        ),
        "bridge_corridor": (
            None
            if edge.bridge_corridor is None
            else list(edge.bridge_corridor)
        ),
        "length": float(x[0]),
        "elevation": float(x[1]),
        "avg_pop": float(x[2]),
        "avg_width": float(x[3]),
        "road_changes": int(edge.road_changes),
        "path_nodes": len(edge.path_nodes),
    }


def _shortcut_capacity_diagnostics(state: Any) -> dict[str, Any]:
    groups: dict[tuple[int, int, str], list[OverlayEdge]] = {}
    for src, edges in state.overlay.out_edges.items():
        for edge in edges:
            if edge.bridge_cell_pair is not None:
                continue
            groups.setdefault((int(src), int(edge.dst), edge.kind), []).append(edge)

    gt2: list[dict[str, Any]] = []
    at_limit = 0
    for key, edges in groups.items():
        if len(edges) >= state.config.max_shortcuts_per_pair:
            at_limit += 1
        if len(edges) <= 2:
            continue
        gt2.append(
            {
                "src": key[0],
                "dst": key[1],
                "kind": key[2],
                "count": len(edges),
                "edges": [_edge_metric_summary(edge) for edge in sorted(edges, key=ps._overlay_edge_capacity_key)],
            }
        )
    gt2.sort(key=lambda row: (-int(row["count"]), row["src"], row["dst"], row["kind"]))
    return {
        "groups_total": len(groups),
        "groups_at_current_limit": at_limit,
        "groups_retaining_more_than_2_shortcuts": len(gt2),
        "shortcuts_above_2_total": sum(int(row["count"]) - 2 for row in gt2),
        "examples": gt2[:20],
        "local_engine_state_model": "one best_length local state per node",
    }


def _bridge_capacity_diagnostics(state: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell_pair, edges in sorted(state.bridge_representatives_by_cell_pair.items()):
        rows.append(
            {
                "cell_pair": list(cell_pair),
                "count": len(edges),
                "corridors": [
                    None if edge.bridge_corridor is None else list(edge.bridge_corridor)
                    for edge in edges
                ],
                "edges": [_edge_metric_summary(edge) for edge in edges],
            }
        )
    return {
        "cell_pairs_total": len(rows),
        "cell_pairs_retaining_more_than_2_bridges": sum(
            1 for row in rows if int(row["count"]) > 2
        ),
        "bridges_above_2_total": sum(max(0, int(row["count"]) - 2) for row in rows),
        "examples_more_than_2": [row for row in rows if int(row["count"]) > 2][:20],
        "all_cell_pair_counts": {
            f"{row['cell_pair'][0]}->{row['cell_pair'][1]}": row["count"]
            for row in rows
        },
    }


def _portal_capacity_diagnostics(
    context: SweepContext,
    state: Any,
    initial_active: set[int],
) -> dict[str, Any]:
    new_initial = sorted(initial_active - context.baseline_initial_active_portals)
    new_final = sorted(state.active_portals - context.baseline_initial_active_portals)
    initial_cells = _cell_counts(new_initial, context.partition)
    final_cells = _cell_counts(new_final, context.partition)
    oracle_initial = [
        node
        for node in new_initial
        if context.partition.get(node, -(node + 1)) in ORACLE_CELLS
    ]
    oracle_final = [
        node
        for node in new_final
        if context.partition.get(node, -(node + 1)) in ORACLE_CELLS
    ]
    return {
        "baseline_initial_active_portals": len(context.baseline_initial_active_portals),
        "variant_initial_active_portals": len(initial_active),
        "new_initial_active_portals": len(new_initial),
        "new_initial_portals_by_cell": initial_cells,
        "new_initial_oracle_cell_portals": oracle_initial[:100],
        "new_final_active_portals_vs_baseline_initial": len(new_final),
        "new_final_portals_by_cell": final_cells,
        "new_final_oracle_cell_portals": oracle_final[:100],
        "oracle_cell_87_reached": 87 in (_covered_cells(state, "fwd") + _covered_cells(state, "bwd")),
        "oracle_portal_summaries": {
            str(portal): longrun._portal_label_resource_summary(state, portal)
            for portal in ORACLE_ACTIVE_NODES
        },
    }


def _complete_candidate_summary(
    tracker: longrun.CompleteCandidateTracker,
) -> dict[str, Any]:
    data = tracker.as_dict()
    best = data.get("best_near_feasible")
    if isinstance(best, dict):
        key = best.get("key")
        if isinstance(key, list) and key:
            data["best_violation_sum"] = key[0]
            data["best_violation_max"] = key[1] if len(key) > 1 else None
        else:
            data["best_violation_sum"] = None
            data["best_violation_max"] = None
    else:
        data["best_violation_sum"] = None
        data["best_violation_max"] = None
    return data


def _snapshot(
    context: SweepContext,
    state: Any,
    tracker: longrun.CompleteCandidateTracker,
    telemetry: RunTelemetry,
    *,
    checkpoint_target_s: float | None,
    wall_start: float,
    cpu_start: float,
) -> dict[str, Any]:
    telemetry.update_state(state)
    fwd_cells = _covered_cells(state, "fwd")
    bwd_cells = _covered_cells(state, "bwd")
    archive_routes = _route_validation_summaries(context, state)
    return {
        "checkpoint_target_s": checkpoint_target_s,
        "elapsed_wall_s": time.perf_counter() - wall_start,
        "elapsed_cpu_s": time.process_time() - cpu_start,
        "archive_size": len(state.archive.entries),
        "archive_routes": archive_routes,
        "frontier_pops": dict(sorted(state.audit.frontier_pops.items())),
        "queue_sizes": {
            "fwd": len(state.frontier["fwd"]),
            "bwd": len(state.frontier["bwd"]),
        },
        "retained_labels": {
            "fwd": _retained_label_count(state, "fwd"),
            "bwd": _retained_label_count(state, "bwd"),
        },
        "distinct_representative_portals": {
            "fwd": _covered_portal_count(state, "fwd"),
            "bwd": _covered_portal_count(state, "bwd"),
        },
        "covered_cells": {"fwd": fwd_cells, "bwd": bwd_cells},
        "oracle_contact": longrun._oracle_contact(fwd_cells, bwd_cells),
        "oracle_portal_summaries": {
            str(portal): longrun._portal_label_resource_summary(state, portal)
            for portal in ORACLE_ACTIVE_NODES
        },
        "completion_length_lb_rejects": state.audit.rejected_length_completion_lower_bound,
        "bridge_edges_inserted": state.audit.bridge_edges_inserted,
        "complete_join_tests": state.audit.feasibility_checked_on_combined_accumulator,
        "complete_candidate_quality": _complete_candidate_summary(tracker),
        "overlay_edges": _overlay_edge_count(state),
        "active_portals": len(state.active_portals),
        "max_queue_sizes_so_far": dict(telemetry.max_queue_sizes),
        "edge_cap_telemetry_so_far": {
            "attempts": dict(sorted(telemetry.edge_attempts.items())),
            "insertions": dict(sorted(telemetry.edge_insertions.items())),
            "duplicate_rejections": dict(
                sorted(telemetry.edge_duplicate_rejections.items())
            ),
            "capacity_rejections": dict(
                sorted(telemetry.edge_capacity_rejections.items())
            ),
            "capacity_replacements": dict(
                sorted(telemetry.edge_capacity_replacements.items())
            ),
        },
    }


def _print_checkpoint(variant_name: str, snapshot: dict[str, Any]) -> None:
    quality = snapshot["complete_candidate_quality"]
    best = quality.get("best_near_feasible")
    best_text = "none"
    if isinstance(best, dict):
        metrics = best["metrics"]
        violations = best["violations"]
        best_text = (
            f"L={metrics['length']:.1f} H={metrics['elevation']:.1f} "
            f"P={metrics['avg_pop']:.2f} W={metrics['avg_width']:.2f} "
            f"viol={violations}"
        )
    print(
        "SWEEP CHECKPOINT "
        f"variant={variant_name} "
        f"target={snapshot['checkpoint_target_s']} "
        f"elapsed={snapshot['elapsed_wall_s']:.1f}s "
        f"archive={snapshot['archive_size']} "
        f"pops={snapshot['frontier_pops']} "
        f"queues={snapshot['queue_sizes']} "
        f"labels={snapshot['retained_labels']} "
        f"bridges={snapshot['bridge_edges_inserted']} "
        f"tests={snapshot['complete_join_tests']} "
        f"cell87={snapshot['oracle_contact']['cell_87_reached']} "
        f"best={best_text}",
        flush=True,
    )


def _write_output(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_variant(
    context: SweepContext,
    variant: CompressionVariant,
    *,
    max_seconds: float,
    output_payload: dict[str, Any],
    output_json: str,
) -> dict[str, Any]:
    config = _variant_config(variant)
    initial_active = _initial_active_portals(context, config)
    tracker = longrun.CompleteCandidateTracker()
    telemetry = RunTelemetry.create()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checkpoint_targets = tuple(target for target in CHECKPOINTS_S if target <= max_seconds)
    next_checkpoint_index = 0
    snapshots: list[dict[str, Any]] = []
    route_events: list[dict[str, Any]] = []

    original_feasible = ps._is_combined_route_feasible
    original_archive = ps._archive_join_candidate
    original_advance = ps.advance_overlay_and_learn_shortcuts
    original_add_overlay_edge = ps.AnytimeSearchState.add_overlay_edge

    def wrapped_feasible(state: Any, metrics: RouteAccumulator) -> bool:
        feasible = original_feasible(state, metrics)
        tracker.update(state.query.constraints, metrics, feasible=feasible)
        return feasible

    def wrapped_archive(state: Any, **kwargs: Any) -> bool:
        before_paths = {entry.path_nodes for entry in state.archive.entries}
        added = original_archive(state, **kwargs)
        if added:
            new_entries = [
                entry
                for entry in state.archive.entries
                if entry.path_nodes not in before_paths
            ]
            entry = new_entries[0] if new_entries else state.archive.entries[0]
            elapsed = time.perf_counter() - wall_start
            if telemetry.first_route_time_s is None:
                telemetry.first_route_time_s = elapsed
            validation = article._validate_route(
                context.G,
                context.partition,
                context.constraints,
                entry,
                state.config.max_cell_visits_per_route,
            )
            route_events.append(
                {
                    "elapsed_wall_s": elapsed,
                    "elapsed_cpu_s": time.process_time() - cpu_start,
                    "archive_size": len(state.archive.entries),
                    "route": longrun._archive_entry_summary(entry, validation),
                }
            )
        return added

    def wrapped_add_overlay_edge(state: Any, edge: OverlayEdge) -> bool:
        bucket = state.overlay.out_edges.setdefault(edge.src, [])
        same_pair = [
            existing
            for existing in bucket
            if existing.dst == edge.dst and existing.kind == edge.kind
        ]
        duplicate = any(existing.path_nodes == edge.path_nodes for existing in same_pair)
        capacity_limited = False
        capacity_replaced = False
        if not duplicate and len(same_pair) >= state.config.max_shortcuts_per_pair:
            edge_key = ps._overlay_edge_capacity_key(edge)
            worst = max(same_pair, key=ps._overlay_edge_capacity_key)
            worst_key = ps._overlay_edge_capacity_key(worst)
            capacity_limited = edge_key >= worst_key
            capacity_replaced = edge_key < worst_key
        inserted = original_add_overlay_edge(state, edge)
        telemetry.record_edge_attempt(
            state,
            edge,
            inserted=inserted,
            duplicate=duplicate,
            capacity_limited=(capacity_limited and not inserted),
            capacity_replaced=capacity_replaced and inserted,
        )
        return inserted

    def maybe_checkpoint(state: Any, *, force: bool = False) -> None:
        nonlocal next_checkpoint_index
        elapsed = time.perf_counter() - wall_start
        wrote = False
        while (
            next_checkpoint_index < len(checkpoint_targets)
            and elapsed >= checkpoint_targets[next_checkpoint_index]
        ):
            snapshot = _snapshot(
                context,
                state,
                tracker,
                telemetry,
                checkpoint_target_s=checkpoint_targets[next_checkpoint_index],
                wall_start=wall_start,
                cpu_start=cpu_start,
            )
            snapshots.append(snapshot)
            _print_checkpoint(variant.name, snapshot)
            next_checkpoint_index += 1
            wrote = True
        if force:
            snapshot = _snapshot(
                context,
                state,
                tracker,
                telemetry,
                checkpoint_target_s=None,
                wall_start=wall_start,
                cpu_start=cpu_start,
            )
            snapshots.append(snapshot)
            _print_checkpoint(variant.name, snapshot)
            wrote = True
        if wrote:
            output_payload["runs"][variant.name] = {
                "status": "running",
                "variant": variant.name,
                "config": _config_dict(config),
                "snapshots": snapshots,
                "route_events": route_events,
            }
            _write_output(output_json, output_payload)

    def wrapped_advance(state: Any, *fn_args: Any, **fn_kwargs: Any) -> int:
        progressed = original_advance(state, *fn_args, **fn_kwargs)
        telemetry.update_state(state)
        maybe_checkpoint(state)
        if len(state.archive.entries) >= 3:
            validations = [
                article._validate_route(
                    context.G,
                    context.partition,
                    context.constraints,
                    entry,
                    state.config.max_cell_visits_per_route,
                )
                for entry in state.archive.entries
            ]
            if all(validation.passed for validation in validations):
                state.deadline = time.perf_counter()
        return progressed

    ps._is_combined_route_feasible = wrapped_feasible
    ps._archive_join_candidate = wrapped_archive
    ps.advance_overlay_and_learn_shortcuts = wrapped_advance
    ps.AnytimeSearchState.add_overlay_edge = wrapped_add_overlay_edge

    state = None
    status = "complete"
    try:
        state = anytime_sparse_portal_search(
            G=context.G,
            partition=context.partition,
            boundary_nodes=context.boundary_nodes,
            kept_nodes=context.kept_nodes,
            query=_make_query(max_seconds, context.constraints),
            config=config,
        )
        maybe_checkpoint(state, force=True)
    except MemoryError:
        status = "memory_error"
        raise
    except Exception:
        status = "runtime_error"
        raise
    finally:
        ps._is_combined_route_feasible = original_feasible
        ps._archive_join_candidate = original_archive
        ps.advance_overlay_and_learn_shortcuts = original_advance
        ps.AnytimeSearchState.add_overlay_edge = original_add_overlay_edge

    assert state is not None
    elapsed_wall = time.perf_counter() - wall_start
    elapsed_cpu = time.process_time() - cpu_start
    exhausted = not state.frontier["fwd"] and not state.frontier["bwd"]
    final_snapshot = snapshots[-1] if snapshots else None
    archive_validations = [
        article._validate_route(
            context.G,
            context.partition,
            context.constraints,
            entry,
            state.config.max_cell_visits_per_route,
        )
        for entry in state.archive.entries
    ]
    run_result = {
        "status": status,
        "variant": variant.name,
        "combination": variant.combination,
        "config": _config_dict(config),
        "elapsed_wall_s": elapsed_wall,
        "elapsed_cpu_s": elapsed_cpu,
        "cpu_wall_ratio": elapsed_cpu / elapsed_wall if elapsed_wall > 0.0 else None,
        "archive_size": len(state.archive.entries),
        "all_archived_routes_valid": (
            all(validation.passed for validation in archive_validations)
            if archive_validations
            else None
        ),
        "time_to_first_feasible_route_s": telemetry.first_route_time_s,
        "exhausted": exhausted,
        "exhaustion_time_s": elapsed_wall if exhausted else None,
        "frontier_pops": dict(sorted(state.audit.frontier_pops.items())),
        "retained_labels": {
            "fwd": _retained_label_count(state, "fwd"),
            "bwd": _retained_label_count(state, "bwd"),
        },
        "max_queue_sizes": dict(telemetry.max_queue_sizes),
        "max_retained_labels": dict(telemetry.max_retained_labels),
        "max_representative_portals": dict(telemetry.max_representative_portals),
        "portals_reached": {
            "fwd": _covered_portal_count(state, "fwd"),
            "bwd": _covered_portal_count(state, "bwd"),
        },
        "cells_reached": {
            "fwd": _covered_cells(state, "fwd"),
            "bwd": _covered_cells(state, "bwd"),
        },
        "cell_87_reached": 87 in (_covered_cells(state, "fwd") + _covered_cells(state, "bwd")),
        "bridge_edges_inserted": state.audit.bridge_edges_inserted,
        "bridge_fallback_children_accepted": state.audit.bridge_fallback_children_accepted,
        "complete_join_tests": state.audit.feasibility_checked_on_combined_accumulator,
        "completion_length_lb_rejects": state.audit.rejected_length_completion_lower_bound,
        "best_near_feasible": _complete_candidate_summary(tracker),
        "edge_cap_telemetry": {
            "attempts": dict(sorted(telemetry.edge_attempts.items())),
            "insertions": dict(sorted(telemetry.edge_insertions.items())),
            "duplicate_rejections": dict(
                sorted(telemetry.edge_duplicate_rejections.items())
            ),
            "capacity_rejections": dict(
                sorted(telemetry.edge_capacity_rejections.items())
            ),
            "capacity_replacements": dict(
                sorted(telemetry.edge_capacity_replacements.items())
            ),
        },
        "label_capacity_diagnostics": _label_capacity_diagnostics(state),
        "portal_capacity_diagnostics": _portal_capacity_diagnostics(
            context,
            state,
            initial_active,
        ),
        "shortcut_capacity_diagnostics": _shortcut_capacity_diagnostics(state),
        "bridge_capacity_diagnostics": _bridge_capacity_diagnostics(state),
        "oracle_portal_summaries": {
            str(portal): longrun._portal_label_resource_summary(state, portal)
            for portal in ORACLE_ACTIVE_NODES
        },
        "route_events": route_events,
        "archive_routes": _route_validation_summaries(context, state),
        "archive_jaccard": longrun._jaccard_entries(list(state.archive.entries)),
        "snapshots": snapshots,
        "audit_subset": {
            "rejected_length": state.audit.rejected_length,
            "rejected_elevation": state.audit.rejected_elevation,
            "rejected_avg_popularity": state.audit.rejected_avg_popularity,
            "rejected_avg_width": state.audit.rejected_avg_width,
            "rejected_third_cell_visit": state.audit.rejected_third_cell_visit,
            "second_cell_visits_allowed": state.audit.second_cell_visits_allowed,
            "bridge_refinements_attempted": state.audit.bridge_refinements_attempted,
            "bridge_refinement_cell_pairs": dict(
                sorted(state.audit.bridge_refinement_cell_pairs.items())
            ),
            "representative_accept_reason_count": dict(
                sorted(state.audit.representative_accept_reason_count.items())
            ),
        },
    }
    output_payload["runs"][variant.name] = run_result
    _write_output(output_json, output_payload)
    return run_result


def _load_reused_baseline(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return None
    final_snapshot = snapshots[-1]
    quality = final_snapshot.get("complete_candidate_quality", {})
    return {
        "status": payload.get("status"),
        "variant": "BASELINE",
        "source": path,
        "reused": True,
        "archive_size": final_snapshot.get("archive_size"),
        "time_to_first_feasible_route_s": payload.get("first_route_time_s"),
        "exhausted": (
            final_snapshot.get("queue_sizes", {}).get("fwd") == 0
            and final_snapshot.get("queue_sizes", {}).get("bwd") == 0
        ),
        "exhaustion_time_s": final_snapshot.get("elapsed_wall_s"),
        "elapsed_wall_s": final_snapshot.get("elapsed_wall_s"),
        "elapsed_cpu_s": final_snapshot.get("elapsed_cpu_s"),
        "frontier_pops": final_snapshot.get("frontier_pops"),
        "retained_labels": final_snapshot.get("retained_labels"),
        "portals_reached": final_snapshot.get("distinct_representative_portals"),
        "cells_reached": final_snapshot.get("covered_cells"),
        "cell_87_reached": final_snapshot.get("oracle_contact", {}).get(
            "cell_87_reached"
        ),
        "bridge_edges_inserted": final_snapshot.get("bridges", {}).get(
            "edges_inserted"
        ),
        "complete_join_tests": final_snapshot.get("joins", {}).get(
            "complete_join_tests"
        ),
        "completion_length_lb_rejects": final_snapshot.get(
            "completion_length_lb_rejects"
        ),
        "best_near_feasible": quality,
        "snapshots": snapshots,
    }


def _baseline_summary_from_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "archive_size": run.get("archive_size"),
        "best_violation_sum": run.get("best_near_feasible", {}).get(
            "best_violation_sum"
        ),
        "best_near_feasible": run.get("best_near_feasible", {}).get(
            "best_near_feasible"
        ),
        "min_avg_width": run.get("best_near_feasible", {}).get("min_avg_width"),
        "max_avg_pop": run.get("best_near_feasible", {}).get("max_avg_pop"),
        "complete_join_tests": run.get("complete_join_tests"),
        "cell_87_reached": run.get("cell_87_reached"),
    }


def _should_run_combinations(
    baseline: dict[str, Any] | None,
    runs: Sequence[dict[str, Any]],
) -> bool:
    if any(int(run.get("archive_size") or 0) > 0 for run in runs):
        return True
    if baseline is None:
        return True
    base_quality = baseline.get("best_near_feasible", {})
    base_best = base_quality.get("best_near_feasible") if isinstance(base_quality, dict) else None
    base_key = None
    if isinstance(base_best, dict):
        key = base_best.get("key")
        if isinstance(key, list) and key:
            base_key = float(key[0])
    for run in runs:
        quality = run.get("best_near_feasible", {})
        best_sum = quality.get("best_violation_sum") if isinstance(quality, dict) else None
        if base_key is not None and best_sum is not None and float(best_sum) < base_key:
            return True
        if int(run.get("complete_join_tests") or 0) > int(
            baseline.get("complete_join_tests") or 0
        ):
            return True
    return False


def _comparison_rows(runs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, run in runs.items():
        quality = run.get("best_near_feasible", {})
        best = quality.get("best_near_feasible") if isinstance(quality, dict) else None
        metrics = best.get("metrics") if isinstance(best, dict) else None
        violations = best.get("violations") if isinstance(best, dict) else None
        rows.append(
            {
                "variant": name,
                "archive": run.get("archive_size"),
                "elapsed_s": run.get("elapsed_wall_s"),
                "exhausted": run.get("exhausted"),
                "first_route_s": run.get("time_to_first_feasible_route_s"),
                "fwd_pops": (run.get("frontier_pops") or {}).get("fwd"),
                "bwd_pops": (run.get("frontier_pops") or {}).get("bwd"),
                "labels_fwd": (run.get("retained_labels") or {}).get("fwd"),
                "labels_bwd": (run.get("retained_labels") or {}).get("bwd"),
                "portals_fwd": (run.get("portals_reached") or {}).get("fwd"),
                "portals_bwd": (run.get("portals_reached") or {}).get("bwd"),
                "cell87": run.get("cell_87_reached"),
                "bridges": run.get("bridge_edges_inserted"),
                "complete_tests": run.get("complete_join_tests"),
                "lb_rejects": run.get("completion_length_lb_rejects"),
                "best_metrics": metrics,
                "best_violations": violations,
                "min_width_seen": (
                    quality.get("min_avg_width") if isinstance(quality, dict) else None
                ),
                "max_pop_seen": (
                    quality.get("max_avg_pop") if isinstance(quality, dict) else None
                ),
            }
        )
    return rows


def _print_final_table(runs: dict[str, Any]) -> None:
    print("COMPRESSION SWEEP SUMMARY")
    print(
        "variant archive elapsed exhausted fwd_pops bwd_pops labels_f/b "
        "portals_f/b cell87 bridges complete_tests best_L best_H best_pop best_width"
    )
    for row in _comparison_rows(runs):
        metrics = row["best_metrics"] or {}
        print(
            f"{row['variant']:>24} "
            f"{row['archive']!s:>7} "
            f"{float(row['elapsed_s'] or 0.0):>8.1f} "
            f"{str(row['exhausted']):>9} "
            f"{str(row['fwd_pops']):>8} "
            f"{str(row['bwd_pops']):>8} "
            f"{str(row['labels_fwd'])}/{str(row['labels_bwd']):<5} "
            f"{str(row['portals_fwd'])}/{str(row['portals_bwd']):<5} "
            f"{str(row['cell87']):>6} "
            f"{str(row['bridges']):>7} "
            f"{str(row['complete_tests']):>14} "
            f"{float(metrics.get('length', 0.0)):>8.1f} "
            f"{float(metrics.get('elevation', 0.0)):>7.1f} "
            f"{float(metrics.get('avg_pop', 0.0)):>8.2f} "
            f"{float(metrics.get('avg_width', 0.0)):>10.2f}"
        )


def _load_context() -> SweepContext:
    G, nodes, kept_cells, partition, boundary_nodes, kept_nodes, constraints = (
        longrun._load_problem()
    )
    base_active = ps._select_active_portals(
        G,
        boundary_nodes,
        partition,
        kept_nodes,
        article.SOURCE,
        article.TARGET,
        params.MAX_ACTIVE_PORTALS_PER_CELL,
    )
    return SweepContext(
        G=G,
        nodes=nodes,
        kept_cells=kept_cells,
        partition=partition,
        boundary_nodes=boundary_nodes,
        kept_nodes=kept_nodes,
        constraints=constraints,
        baseline_initial_active_portals=base_active,
    )


def _selected_variants(names: str) -> list[CompressionVariant]:
    all_variants = {variant.name: variant for variant in ONE_FACTOR_VARIANTS + COMBINATION_VARIANTS}
    if names == "all":
        return list(ONE_FACTOR_VARIANTS)
    selected: list[CompressionVariant] = []
    for name in names.split(","):
        clean = name.strip()
        if not clean:
            continue
        if clean not in all_variants:
            raise ValueError(f"unknown variant {clean!r}")
        selected.append(all_variants[clean])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled Paris-Bures search-compression feasibility study."
    )
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--baseline-json", default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--rerun-baseline", action="store_true")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants, or 'all' for one-factor variants.",
    )
    parser.add_argument(
        "--include-combinations",
        choices=("auto", "always", "never"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("brcore.algo.portal_search").setLevel(logging.WARNING)
    context = _load_context()
    print(
        "Compression sweep setup "
        f"kept_cells={len(context.kept_cells)} "
        f"kept_nodes={len(context.kept_nodes)} "
        f"baseline_initial_active_portals={len(context.baseline_initial_active_portals)} "
        f"max_seconds={args.max_seconds}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "query": {
            "source": article.SOURCE,
            "target": article.TARGET,
            "constraints": {
                "lower": list(map(float, article.LOWER)),
                "upper": list(map(float, article.UPPER)),
            },
            "corridor_slack_m": article.CORRIDOR_SLACK_M,
            "max_hops_from_boundary": article.MAX_HOPS_FROM_BOUNDARY,
            "oracle_metrics": longrun.ORACLE_METRICS,
            "oracle_cells": list(longrun.ORACLE_CELLS),
            "oracle_active_nodes": list(ORACLE_ACTIVE_NODES),
        },
        "baseline_config": _config_dict(_base_config()),
        "runs": {},
        "comparison_rows": [],
    }

    if args.rerun_baseline:
        baseline_variant = CompressionVariant("BASELINE", {})
        print("Running BASELINE from scratch.", flush=True)
        _run_variant(
            context,
            baseline_variant,
            max_seconds=args.max_seconds,
            output_payload=payload,
            output_json=args.output_json,
        )
    else:
        reused = _load_reused_baseline(args.baseline_json)
        if reused is not None:
            print(f"Reusing baseline JSON: {args.baseline_json}", flush=True)
            payload["runs"]["BASELINE"] = reused
            _write_output(args.output_json, payload)
        else:
            print("No reusable baseline JSON found; running BASELINE.", flush=True)
            _run_variant(
                context,
                CompressionVariant("BASELINE", {}),
                max_seconds=args.max_seconds,
                output_payload=payload,
                output_json=args.output_json,
            )

    one_factor_results: list[dict[str, Any]] = []
    for variant in _selected_variants(args.variants):
        print(f"Running {variant.name}: overrides={variant.overrides}", flush=True)
        result = _run_variant(
            context,
            variant,
            max_seconds=args.max_seconds,
            output_payload=payload,
            output_json=args.output_json,
        )
        one_factor_results.append(result)

    should_run_combos = False
    if args.include_combinations == "always":
        should_run_combos = True
    elif args.include_combinations == "auto":
        should_run_combos = _should_run_combinations(
            payload["runs"].get("BASELINE"),
            one_factor_results,
        )

    if should_run_combos:
        for variant in COMBINATION_VARIANTS:
            print(f"Running {variant.name}: overrides={variant.overrides}", flush=True)
            _run_variant(
                context,
                variant,
                max_seconds=args.max_seconds,
                output_payload=payload,
                output_json=args.output_json,
            )
    else:
        print("Skipping combination variants by policy.", flush=True)

    payload["comparison_rows"] = _comparison_rows(payload["runs"])
    payload["structural_diagnostic"] = {
        "persistent_local_engine_state_model": (
            "LocalCellEngineState keeps best_length: Dict[int, float], "
            "so local discovery is one shortest-length state per node."
        ),
        "shortcut_cap_interpretation": (
            "If SHORTCUT4 retains few or no groups above two shortcuts, increasing "
            "max_shortcuts_per_pair did not expose alternatives because they were not "
            "generated or survived only as one best-length local state per node."
        ),
    }
    _write_output(args.output_json, payload)
    _print_final_table(payload["runs"])
    print(f"Output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
