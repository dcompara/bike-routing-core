from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass, field
import io
import json
import logging
import pstats
import time
from typing import Any, Iterable

import numpy as np

from brcore.algo import params
from brcore.algo.coords import build_local_xy_int
import brcore.algo.portal_search as ps
from brcore.algo.portal_search import (
    ArchiveEntry,
    ConstraintBox,
    RouteAccumulator,
    SparsePortalQuery,
    anytime_sparse_portal_search,
)
from brcore.algo.search_space_reduction import search_space_reduction
from brcore.io.load_plot_xy import load_xy_graph
from brcore.io.loaders import (
    load_boundary_edges,
    load_boundary_nodes,
    load_partition,
    load_seeds,
)

import article_paris_bures as article


CHECKPOINTS_S = (
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
    1800.0,
    2700.0,
    3600.0,
    5400.0,
    7200.0,
)
ORACLE_CELLS = (36, 93, 79, 40, 99, 97, 56, 48, 30, 54, 13, 49, 87, 31, 37, 1)
ORACLE_METRICS = {
    "length": 31818.0,
    "elevation": 405.4,
    "avg_pop": 199.87,
    "avg_width": 14.16,
}
ORACLE_ACTIVE_NODES = (896, 4057)
DEFAULT_OUTPUT_JSON = "tmp_paris_bures_2h_longrun.json"
DEFAULT_ROUTE_NODES_JSON = "tmp_paris_bures_2h_first_route_nodes.json"
DEFAULT_PROFILE_JSON = "tmp_paris_bures_profile.json"


@dataclass
class CompleteCandidateRecord:
    metrics: dict[str, float]
    violations: dict[str, float]
    key: tuple[float, ...]


@dataclass
class CompleteCandidateTracker:
    count: int = 0
    feasible_count: int = 0
    max_avg_pop: float | None = None
    min_avg_width: float | None = None
    min_length: float | None = None
    max_elevation: float | None = None
    min_elevation: float | None = None
    best: CompleteCandidateRecord | None = None

    def update(
        self,
        constraints: ConstraintBox,
        metrics: RouteAccumulator,
        *,
        feasible: bool | None = None,
    ) -> None:
        x = metrics.route_vector()
        length = float(x[0])
        elevation = float(x[1])
        avg_pop = float(x[2])
        avg_width = float(x[3])
        violations = _violation_vector(constraints, metrics)
        key = _violation_key(constraints, violations, x)
        self.count += 1
        self.max_avg_pop = (
            avg_pop if self.max_avg_pop is None else max(self.max_avg_pop, avg_pop)
        )
        self.min_avg_width = (
            avg_width
            if self.min_avg_width is None
            else min(self.min_avg_width, avg_width)
        )
        self.min_length = (
            length if self.min_length is None else min(self.min_length, length)
        )
        self.max_elevation = (
            elevation
            if self.max_elevation is None
            else max(self.max_elevation, elevation)
        )
        self.min_elevation = (
            elevation
            if self.min_elevation is None
            else min(self.min_elevation, elevation)
        )
        if feasible:
            self.feasible_count += 1
        if self.best is None or key < self.best.key:
            self.best = CompleteCandidateRecord(
                metrics={
                    "length": length,
                    "elevation": elevation,
                    "avg_pop": avg_pop,
                    "avg_width": avg_width,
                },
                violations=violations,
                key=key,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete_candidates": self.count,
            "feasible_complete_candidates": self.feasible_count,
            "max_avg_pop": self.max_avg_pop,
            "min_avg_width": self.min_avg_width,
            "min_length": self.min_length,
            "max_elevation": self.max_elevation,
            "min_elevation": self.min_elevation,
            "best_near_feasible": (
                None
                if self.best is None
                else {
                    "metrics": self.best.metrics,
                    "violations": self.best.violations,
                    "key": list(self.best.key),
                }
            ),
        }


@dataclass
class LongRunMonitor:
    constraints: ConstraintBox
    G: Any
    partition: dict[int, int]
    output_json: str
    route_nodes_json: str
    max_cell_visits_per_route: int
    checkpoint_targets: tuple[float, ...] = CHECKPOINTS_S
    wall_start: float = field(default_factory=time.perf_counter)
    cpu_start: float = field(default_factory=time.process_time)
    tracker: CompleteCandidateTracker = field(default_factory=CompleteCandidateTracker)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    route_events: list[dict[str, Any]] = field(default_factory=list)
    next_checkpoint_index: int = 0
    first_route_time_s: float | None = None
    final_state: Any = None

    def elapsed_wall_s(self) -> float:
        return time.perf_counter() - self.wall_start

    def elapsed_cpu_s(self) -> float:
        return time.process_time() - self.cpu_start

    def maybe_checkpoint(self, state: Any, *, force: bool = False) -> None:
        elapsed = self.elapsed_wall_s()
        wrote = False
        while (
            self.next_checkpoint_index < len(self.checkpoint_targets)
            and elapsed >= self.checkpoint_targets[self.next_checkpoint_index]
        ):
            target = self.checkpoint_targets[self.next_checkpoint_index]
            self.snapshots.append(self.snapshot(state, checkpoint_target_s=target))
            self.next_checkpoint_index += 1
            wrote = True
        if force:
            self.snapshots.append(self.snapshot(state, checkpoint_target_s=None))
            wrote = True
        if wrote:
            self.final_state = state
            self.write_json(status="running")
            _print_checkpoint(self.snapshots[-1])

    def record_route(self, state: Any, entry: ArchiveEntry) -> None:
        elapsed = self.elapsed_wall_s()
        if self.first_route_time_s is None:
            self.first_route_time_s = elapsed
        validation = article._validate_route(
            self.G,
            self.partition,
            self.constraints,
            entry,
            self.max_cell_visits_per_route,
        )
        route_vec = entry.metrics.route_vector()
        repeated = _repeated_cells(entry.cell_sequence)
        route_event = {
            "elapsed_wall_s": elapsed,
            "elapsed_cpu_s": self.elapsed_cpu_s(),
            "archive_size": len(state.archive.entries),
            "metrics": _route_vector_dict(route_vec),
            "score": entry.score,
            "road_changes": entry.road_changes,
            "path_nodes_count": len(entry.path_nodes),
            "cell_sequence": list(entry.cell_sequence),
            "repeated_cells": repeated,
            "uses_cell_87": 87 in entry.cell_sequence,
            "oracle_metric_diff": _oracle_metric_diff(route_vec),
            "validation": _validation_dict(validation),
        }
        self.route_events.append(route_event)
        if len(self.route_events) == 1:
            with open(self.route_nodes_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "elapsed_wall_s": elapsed,
                        "path_nodes": list(entry.path_nodes),
                        "cell_sequence": list(entry.cell_sequence),
                    },
                    fh,
                    indent=2,
                )
        self.write_json(status="running")
        print(
            "FIRST ROUTE"
            if len(self.route_events) == 1
            else "ROUTE ARCHIVED",
            json.dumps(route_event, sort_keys=True),
            flush=True,
        )

    def snapshot(
        self,
        state: Any,
        *,
        checkpoint_target_s: float | None,
    ) -> dict[str, Any]:
        fwd_cells = _covered_cells(state, "fwd")
        bwd_cells = _covered_cells(state, "bwd")
        fwd_label_count = _retained_label_count(state, "fwd")
        bwd_label_count = _retained_label_count(state, "bwd")
        fwd_portal_count = _covered_portal_count(state, "fwd")
        bwd_portal_count = _covered_portal_count(state, "bwd")
        archive_validations = [
            article._validate_route(
                self.G,
                self.partition,
                self.constraints,
                entry,
                self.max_cell_visits_per_route,
            )
            for entry in state.archive.entries
        ]
        return {
            "checkpoint_target_s": checkpoint_target_s,
            "elapsed_wall_s": self.elapsed_wall_s(),
            "elapsed_cpu_s": self.elapsed_cpu_s(),
            "cpu_wall_ratio": (
                self.elapsed_cpu_s() / self.elapsed_wall_s()
                if self.elapsed_wall_s() > 0.0
                else None
            ),
            "archive_size": len(state.archive.entries),
            "archive_valid": [validation.passed for validation in archive_validations],
            "archive_routes": [
                _archive_entry_summary(entry, validation)
                for entry, validation in zip(state.archive.entries, archive_validations)
            ],
            "frontier_pops": dict(sorted(state.audit.frontier_pops.items())),
            "queue_sizes": {
                "fwd": len(state.frontier["fwd"]),
                "bwd": len(state.frontier["bwd"]),
            },
            "retained_labels": {
                "fwd": fwd_label_count,
                "bwd": bwd_label_count,
            },
            "distinct_representative_portals": {
                "fwd": fwd_portal_count,
                "bwd": bwd_portal_count,
            },
            "covered_cells": {
                "fwd": fwd_cells,
                "bwd": bwd_cells,
            },
            "oracle_contact": _oracle_contact(fwd_cells, bwd_cells),
            "oracle_portal_summaries": {
                str(portal): _portal_label_resource_summary(state, portal)
                for portal in ORACLE_ACTIVE_NODES
            },
            "completion_length_lb_rejects": (
                state.audit.rejected_length_completion_lower_bound
            ),
            "second_cell_visits_allowed": state.audit.second_cell_visits_allowed,
            "third_cell_visit_rejects": state.audit.rejected_third_cell_visit,
            "bridges": {
                "pairs_attempted": state.audit.bridge_refinements_attempted,
                "edges_inserted": state.audit.bridge_edges_inserted,
                "children_accepted": state.audit.bridge_fallback_children_accepted,
                "repair_children_generated": state.audit.bridge_repair_children_generated,
                "complementary_sets": (
                    state.audit.complementary_connector_sets_considered
                ),
                "complementary_quality_distinct": (
                    state.audit.complementary_quality_candidate_distinct
                ),
                "complementary_quality_inserted": (
                    state.audit.complementary_quality_candidate_inserted
                ),
            },
            "joins": {
                "same_portal_attempts": state.audit.same_portal_join_attempts,
                "same_portal_successes": state.audit.same_portal_join_successes,
                "one_edge_attempts": state.audit.one_edge_join_attempts,
                "one_edge_successes": state.audit.one_edge_join_successes,
                "complete_join_tests": (
                    state.audit.feasibility_checked_on_combined_accumulator
                ),
                "feasible_complete_joins": self.tracker.feasible_count,
            },
            "complete_candidate_quality": self.tracker.as_dict(),
            "overlay_edges": _overlay_edge_count(state),
            "active_portals": len(state.active_portals),
            "local_engines": len(state.local_engines),
        }

    def write_json(self, *, status: str) -> None:
        payload = {
            "status": status,
            "query": {
                "source": article.SOURCE,
                "target": article.TARGET,
                "constraints": {
                    "lower": list(map(float, article.LOWER)),
                    "upper": list(map(float, article.UPPER)),
                },
                "corridor_slack_m": article.CORRIDOR_SLACK_M,
                "max_hops_from_boundary": article.MAX_HOPS_FROM_BOUNDARY,
                "oracle_metrics": ORACLE_METRICS,
                "oracle_cells": list(ORACLE_CELLS),
            },
            "first_route_time_s": self.first_route_time_s,
            "route_events": self.route_events,
            "snapshots": self.snapshots,
        }
        with open(self.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def _violation_vector(
    constraints: ConstraintBox,
    metrics: RouteAccumulator,
) -> dict[str, float]:
    x = metrics.route_vector()
    lower = constraints.lower
    upper = constraints.upper
    return {
        "length_low": max(float(lower[0]) - float(x[0]), 0.0),
        "length_high": max(float(x[0]) - float(upper[0]), 0.0),
        "elevation_low": max(float(lower[1]) - float(x[1]), 0.0),
        "elevation_high": max(float(x[1]) - float(upper[1]), 0.0),
        "popularity_low": max(float(lower[2]) - float(x[2]), 0.0),
        "width_high": max(float(x[3]) - float(upper[3]), 0.0),
    }


def _violation_key(
    constraints: ConstraintBox,
    violations: dict[str, float],
    route_vector: np.ndarray,
) -> tuple[float, ...]:
    lower = constraints.lower
    upper = constraints.upper
    length_range = max(float(upper[0] - lower[0]), 1.0)
    elevation_range = max(float(upper[1] - lower[1]), 1.0)
    popularity_range = max(float(upper[2] - lower[2]), 1.0)
    width_range = max(float(upper[3] - lower[3]), 1.0)
    normalized = (
        violations["length_low"] / length_range,
        violations["length_high"] / length_range,
        violations["elevation_low"] / elevation_range,
        violations["elevation_high"] / elevation_range,
        violations["popularity_low"] / popularity_range,
        violations["width_high"] / width_range,
    )
    positive_count = sum(1 for value in normalized if value > 0.0)
    return (
        sum(normalized),
        max(normalized),
        float(positive_count),
        *normalized,
        float(route_vector[0]),
        float(route_vector[1]),
        -float(route_vector[2]),
        float(route_vector[3]),
    )


def _route_vector_dict(route_vector: np.ndarray) -> dict[str, float]:
    return {
        "length": float(route_vector[0]),
        "elevation": float(route_vector[1]),
        "avg_pop": float(route_vector[2]),
        "avg_width": float(route_vector[3]),
    }


def _validation_dict(validation: Any) -> dict[str, Any]:
    return {
        "passed": validation.passed,
        "missing_edges": validation.missing_edges,
        "ambiguous_edges": validation.ambiguous_edges,
        "length_delta": validation.length_delta,
        "elevation_delta": validation.elevation_delta,
        "avg_popularity_delta": validation.avg_popularity_delta,
        "avg_width_delta": validation.avg_width_delta,
        "road_changes_delta": validation.road_changes_delta,
        "feasible": validation.feasible,
        "cell_visit_limit_ok": validation.cell_visit_limit_ok,
        "max_cell_visit_count": validation.max_cell_visit_count,
        "twice_visited_cells": list(validation.twice_visited_cells),
    }


def _archive_entry_summary(entry: ArchiveEntry, validation: Any) -> dict[str, Any]:
    x = entry.metrics.route_vector()
    return {
        "score": entry.score,
        "metrics": _route_vector_dict(x),
        "road_changes": entry.road_changes,
        "path_nodes": len(entry.path_nodes),
        "cell_sequence": list(entry.cell_sequence),
        "repeated_cells": _repeated_cells(entry.cell_sequence),
        "uses_cell_87": 87 in entry.cell_sequence,
        "bridge_cell_pairs": [list(pair) for pair in entry.bridge_cell_pairs],
        "bridge_corridors": [list(corridor) for corridor in entry.bridge_corridors],
        "validation": _validation_dict(validation),
    }


def _oracle_metric_diff(route_vector: np.ndarray) -> dict[str, float]:
    return {
        "length": float(route_vector[0]) - ORACLE_METRICS["length"],
        "elevation": float(route_vector[1]) - ORACLE_METRICS["elevation"],
        "avg_pop": float(route_vector[2]) - ORACLE_METRICS["avg_pop"],
        "avg_width": float(route_vector[3]) - ORACLE_METRICS["avg_width"],
    }


def _repeated_cells(cell_sequence: Iterable[int]) -> list[int]:
    counts: dict[int, int] = {}
    for cell in cell_sequence:
        counts[int(cell)] = counts.get(int(cell), 0) + 1
    return sorted(cell for cell, count in counts.items() if count > 1)


def _covered_cells(state: Any, direction: str) -> list[int]:
    return sorted(
        {
            state.trace_cell(portal)
            for portal, labels in state.labels[direction].items()
            if labels
        }
    )


def _covered_portal_count(state: Any, direction: str) -> int:
    return sum(1 for labels in state.labels[direction].values() if labels)


def _retained_label_count(state: Any, direction: str) -> int:
    return sum(len(labels) for labels in state.labels[direction].values())


def _overlay_edge_count(state: Any) -> int:
    return sum(len(edges) for edges in state.overlay.out_edges.values())


def _oracle_contact(fwd_cells: list[int], bwd_cells: list[int]) -> dict[str, Any]:
    fwd = set(fwd_cells)
    bwd = set(bwd_cells)
    oracle = set(ORACLE_CELLS)
    return {
        "cell_87_reached": 87 in fwd or 87 in bwd,
        "forward_oracle_cells": sorted(fwd & oracle),
        "backward_oracle_cells": sorted(bwd & oracle),
        "both_oracle_cells": sorted(fwd & bwd & oracle),
        "missing_oracle_cells": sorted(oracle - (fwd | bwd)),
    }


def _portal_label_resource_summary(state: Any, portal: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for direction in ("fwd", "bwd"):
        labels = state.labels[direction].get(portal, [])
        if not labels:
            out[direction] = None
            continue
        low_width = min(labels, key=lambda label: label.metrics.route_vector()[3])
        high_pop = max(labels, key=lambda label: label.metrics.route_vector()[2])
        best_score = min(labels, key=lambda label: label.priority)
        out[direction] = {
            "count": len(labels),
            "low_width": _label_summary(low_width),
            "high_pop": _label_summary(high_pop),
            "best_score": _label_summary(best_score),
        }
    return out


def _label_summary(label: Any) -> dict[str, Any]:
    x = label.metrics.route_vector()
    return {
        "portal": label.portal,
        "priority": label.priority,
        "metrics": _route_vector_dict(x),
        "road_changes": label.road_changes,
        "visited_cells": sorted(label.visited_cells),
        "revisited_cells": sorted(label.revisited_cells),
    }


def _print_checkpoint(snapshot: dict[str, Any]) -> None:
    quality = snapshot["complete_candidate_quality"]
    best = quality["best_near_feasible"]
    best_text = "none"
    if best is not None:
        m = best["metrics"]
        v = best["violations"]
        best_text = (
            f"L={m['length']:.1f} H={m['elevation']:.1f} "
            f"P={m['avg_pop']:.2f} W={m['avg_width']:.2f} "
            f"viol={v}"
        )
    print(
        "CHECKPOINT "
        f"target={snapshot['checkpoint_target_s']} "
        f"elapsed={snapshot['elapsed_wall_s']:.1f}s "
        f"cpu={snapshot['elapsed_cpu_s']:.1f}s "
        f"archive={snapshot['archive_size']} "
        f"pops={snapshot['frontier_pops']} "
        f"queues={snapshot['queue_sizes']} "
        f"labels={snapshot['retained_labels']} "
        f"bridges={snapshot['bridges']} "
        f"joins={snapshot['joins']} "
        f"cell87={snapshot['oracle_contact']['cell_87_reached']} "
        f"best={best_text}",
        flush=True,
    )


def _make_query(time_budget_s: float, constraints: ConstraintBox) -> SparsePortalQuery:
    return SparsePortalQuery(
        source=article.SOURCE,
        target=article.TARGET,
        constraints=constraints,
        time_budget_s=time_budget_s,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )


def _load_problem() -> tuple[Any, Any, list[int], dict[int, int], set[int], set[int], ConstraintBox]:
    xy = load_xy_graph(article.GRAPH_PATH)
    G = xy.G
    nodes = xy.nodes
    xy_int = build_local_xy_int(nodes)
    seeds = load_seeds(article.SEEDS_PATH, id_mode="xy")
    partition = load_partition(article.PARTITION_PATH, id_mode="xy")
    boundary_nodes = load_boundary_nodes(article.BOUNDARY_NODES_PATH, id_mode="xy")
    load_boundary_edges(article.BOUNDARY_EDGES_PATH, id_mode="xy", has_key=True)
    constraints = ConstraintBox.from_bounds(article.LOWER, article.UPPER, params.W)
    kept_cells, kept_nodes = search_space_reduction(
        G=G,
        xy_int=xy_int,
        seeds=seeds,
        partition=partition,
        boundary_nodes=boundary_nodes,
        s=article.SOURCE,
        t=article.TARGET,
        corridor_slack_m=article.CORRIDOR_SLACK_M,
        max_hops_from_boundary=article.MAX_HOPS_FROM_BOUNDARY,
    )
    return G, nodes, kept_cells, partition, boundary_nodes, kept_nodes, constraints


def run_long_search(args: argparse.Namespace) -> dict[str, Any]:
    logging.getLogger("brcore.algo.portal_search").setLevel(logging.WARNING)
    G, _, kept_cells, partition, boundary_nodes, kept_nodes, constraints = _load_problem()
    config = article._make_config()
    monitor = LongRunMonitor(
        constraints=constraints,
        G=G,
        partition=partition,
        output_json=args.output_json,
        route_nodes_json=args.route_nodes_json,
        max_cell_visits_per_route=config.max_cell_visits_per_route,
        checkpoint_targets=tuple(
            checkpoint
            for checkpoint in CHECKPOINTS_S
            if checkpoint <= args.max_seconds
        ),
    )
    print(
        "Long-run setup "
        f"kept_cells={len(kept_cells)} kept_nodes={len(kept_nodes)} "
        f"max_seconds={args.max_seconds}",
        flush=True,
    )

    original_feasible = ps._is_combined_route_feasible
    original_archive = ps._archive_join_candidate
    original_advance = ps.advance_overlay_and_learn_shortcuts

    def wrapped_feasible(state: Any, metrics: RouteAccumulator) -> bool:
        feasible = original_feasible(state, metrics)
        monitor.tracker.update(state.query.constraints, metrics, feasible=feasible)
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
            monitor.record_route(state, entry)
        return added

    def wrapped_advance(state: Any, *fn_args: Any, **fn_kwargs: Any) -> int:
        progressed = original_advance(state, *fn_args, **fn_kwargs)
        monitor.maybe_checkpoint(state)
        if len(state.archive.entries) >= 3:
            validations = [
                article._validate_route(
                    G,
                    partition,
                    constraints,
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
    state = None
    status = "complete"
    try:
        state = anytime_sparse_portal_search(
            G=G,
            partition=partition,
            boundary_nodes=boundary_nodes,
            kept_nodes=kept_nodes,
            query=_make_query(args.max_seconds, constraints),
            config=config,
        )
        monitor.final_state = state
        monitor.maybe_checkpoint(state, force=True)
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
        monitor.write_json(status=status)

    assert state is not None
    return {
        "status": status,
        "state": state,
        "monitor": monitor,
    }


def run_profile(profile_seconds: float, output_json: str) -> dict[str, Any] | None:
    if profile_seconds <= 0.0:
        return None
    G, _, _, partition, boundary_nodes, kept_nodes, constraints = _load_problem()
    config = article._make_config()
    profile = cProfile.Profile()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    state = profile.runcall(
        anytime_sparse_portal_search,
        G=G,
        partition=partition,
        boundary_nodes=boundary_nodes,
        kept_nodes=kept_nodes,
        query=_make_query(profile_seconds, constraints),
        config=config,
    )
    wall_s = time.perf_counter() - start_wall
    cpu_s = time.process_time() - start_cpu
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)
    profile_text = stream.getvalue()
    payload = {
        "profile_seconds_requested": profile_seconds,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "cpu_wall_ratio": cpu_s / wall_s if wall_s > 0.0 else None,
        "archive_size": len(state.archive.entries),
        "frontier_pops": dict(sorted(state.audit.frontier_pops.items())),
        "bridge_edges_inserted": state.audit.bridge_edges_inserted,
        "complete_join_tests": state.audit.feasibility_checked_on_combined_accumulator,
        "pstats_top_cumulative": profile_text,
    }
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("PROFILE")
    print(profile_text)
    return payload


def _jaccard_entries(entries: list[ArchiveEntry]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(entries):
        for j, right in enumerate(entries[i + 1 :], start=i + 1):
            left_nodes = set(left.path_nodes)
            right_nodes = set(right.path_nodes)
            union = left_nodes | right_nodes
            rows.append(
                {
                    "left": i + 1,
                    "right": j + 1,
                    "jaccard": (
                        len(left_nodes & right_nodes) / len(union)
                        if union
                        else 1.0
                    ),
                }
            )
    return rows


def print_final_report(result: dict[str, Any], profile_payload: dict[str, Any] | None) -> None:
    state = result["state"]
    monitor: LongRunMonitor = result["monitor"]
    last_snapshot = monitor.snapshots[-1] if monitor.snapshots else None
    print("FINAL LONG-RUN REPORT")
    print(f"  found_route={bool(state.archive.entries)}")
    print(f"  first_route_time_s={monitor.first_route_time_s}")
    print(f"  archive_size={len(state.archive.entries)}")
    print(f"  output_json={monitor.output_json}")
    print(f"  route_nodes_json={monitor.route_nodes_json if monitor.route_events else None}")
    if last_snapshot is not None:
        print(f"  last_elapsed_wall_s={last_snapshot['elapsed_wall_s']:.1f}")
        print(f"  last_elapsed_cpu_s={last_snapshot['elapsed_cpu_s']:.1f}")
        print(f"  cpu_wall_ratio={last_snapshot['cpu_wall_ratio']:.3f}")
        print(f"  last_best={last_snapshot['complete_candidate_quality']['best_near_feasible']}")
        print(f"  oracle_contact={last_snapshot['oracle_contact']}")
    if state.archive.entries:
        validations = [
            article._validate_route(
                monitor.G,
                monitor.partition,
                monitor.constraints,
                entry,
                state.config.max_cell_visits_per_route,
            )
            for entry in state.archive.entries
        ]
        for idx, (entry, validation) in enumerate(zip(state.archive.entries, validations), 1):
            print(f"  route_{idx}={_archive_entry_summary(entry, validation)}")
        print(f"  archive_jaccard={_jaccard_entries(list(state.archive.entries))}")
    if profile_payload is not None:
        print(
            "  profile_summary="
            f"wall_s={profile_payload['wall_s']:.1f} "
            f"cpu_s={profile_payload['cpu_s']:.1f} "
            f"ratio={profile_payload['cpu_wall_ratio']:.3f} "
            f"output={DEFAULT_PROFILE_JSON}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-run frozen paper_v1 Paris-Bures diagnostic."
    )
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--profile-seconds", type=float, default=90.0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--route-nodes-json", default=DEFAULT_ROUTE_NODES_JSON)
    parser.add_argument("--profile-json", default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--skip-main-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not args.skip_main_regression:
        # Kept as a lightweight import-time guard only; compile/main are run by the shell
        # validation commands around this diagnostic.
        pass
    result = run_long_search(args)
    profile_payload = run_profile(args.profile_seconds, args.profile_json)
    print_final_report(result, profile_payload)


if __name__ == "__main__":
    main()
