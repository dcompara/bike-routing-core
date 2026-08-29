from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import time
from typing import Any, Iterable

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


BUDGETS_S = (10.0, 30.0, 60.0, 300.0)
DEFAULT_OUTPUT_JSON = "tmp_paris_bures_local2_experiment.json"
DEFAULT_ROUTE_NODES_JSON = "tmp_paris_bures_local2_first_route_nodes.json"
ORACLE_ACTIVE_NODES = longrun.ORACLE_ACTIVE_NODES


def _make_config(max_local_states_per_node: int) -> SparsePortalConfig:
    return replace(
        article._make_config(),
        max_local_states_per_node=max_local_states_per_node,
    )


def _make_query(time_budget_s: float, constraints: ConstraintBox) -> SparsePortalQuery:
    return SparsePortalQuery(
        source=article.SOURCE,
        target=article.TARGET,
        constraints=constraints,
        time_budget_s=time_budget_s,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )


def _config_dict(config: SparsePortalConfig) -> dict[str, Any]:
    return {
        "max_active_portals_per_cell": config.max_active_portals_per_cell,
        "max_labels_per_portal": config.max_labels_per_portal,
        "max_shortcuts_per_pair": config.max_shortcuts_per_pair,
        "local_expand_limit": config.local_expand_limit,
        "advance_round_budget": config.advance_round_budget,
        "max_cell_visits_per_route": config.max_cell_visits_per_route,
        "max_bridge_edges_per_cell_pair": config.max_bridge_edges_per_cell_pair,
        "max_local_states_per_node": config.max_local_states_per_node,
    }


def _overlay_edge_count(state: Any) -> int:
    return sum(len(edges) for edges in state.overlay.out_edges.values())


def _retained_label_count(state: Any, direction: str) -> int:
    return sum(len(labels) for labels in state.labels[direction].values())


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


def _avg_pop(metrics: RouteAccumulator) -> float:
    if metrics.length <= 0.0:
        return 0.0
    return metrics.popularity_length / metrics.length


def _avg_width(metrics: RouteAccumulator) -> float:
    if metrics.length <= 0.0:
        return 0.0
    return metrics.street_width_length / metrics.length


def _metric_summary(metrics: RouteAccumulator) -> dict[str, float]:
    x = metrics.route_vector()
    return {
        "length": float(x[0]),
        "elevation": float(x[1]),
        "avg_pop": float(x[2]),
        "avg_width": float(x[3]),
    }


def _state_summary(local_state: Any) -> dict[str, Any]:
    return {
        "node": int(local_state.node),
        "metrics": _metric_summary(local_state.metrics),
        "road_changes": int(local_state.road_changes),
        "path_nodes": len(local_state.path_nodes),
    }


def _delta_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _local_state_diagnostics(state: Any) -> dict[str, Any]:
    delta_length: list[float] = []
    delta_elevation: list[float] = []
    delta_avg_pop: list[float] = []
    delta_avg_width: list[float] = []
    examples: list[dict[str, Any]] = []
    final_nodes_with_two_states = 0
    total_representative_slots = 0
    total_frontier_states = 0

    for engine_key, engine in sorted(state.local_engines.items()):
        total_frontier_states += len(engine.frontier)
        for node, reps in engine.representatives.items():
            total_representative_slots += len(
                {
                    rep.local_state_id
                    for rep in reps.values()
                }
            )
            length_rep = reps.get("length")
            quality_rep = reps.get("quality")
            if (
                length_rep is None
                or quality_rep is None
                or length_rep.local_state_id == quality_rep.local_state_id
            ):
                continue
            final_nodes_with_two_states += 1
            d_len = quality_rep.metrics.length - length_rep.metrics.length
            d_ele = quality_rep.metrics.elevation - length_rep.metrics.elevation
            d_pop = _avg_pop(quality_rep.metrics) - _avg_pop(length_rep.metrics)
            d_width = _avg_width(quality_rep.metrics) - _avg_width(
                length_rep.metrics
            )
            delta_length.append(float(d_len))
            delta_elevation.append(float(d_ele))
            delta_avg_pop.append(float(d_pop))
            delta_avg_width.append(float(d_width))
            if len(examples) < 20:
                cell_id, origin, direction = engine_key
                examples.append(
                    {
                        "cell": int(cell_id),
                        "origin": int(origin),
                        "direction": direction,
                        "node": int(node),
                        "delta": {
                            "length": float(d_len),
                            "elevation": float(d_ele),
                            "avg_pop": float(d_pop),
                            "avg_width": float(d_width),
                        },
                        "length_rep": _state_summary(length_rep),
                        "quality_rep": _state_summary(quality_rep),
                    }
                )

    return {
        "engines": len(state.local_engines),
        "total_expansions": sum(engine.expansions for engine in state.local_engines.values()),
        "max_engine_expansions": max(
            (engine.expansions for engine in state.local_engines.values()),
            default=0,
        ),
        "frontier_states_remaining": total_frontier_states,
        "representative_slots": total_representative_slots,
        "final_nodes_with_two_distinct_states": final_nodes_with_two_states,
        "delta_length": _delta_summary(delta_length),
        "delta_elevation": _delta_summary(delta_elevation),
        "delta_avg_pop": _delta_summary(delta_avg_pop),
        "delta_avg_width": _delta_summary(delta_avg_width),
        "examples": examples,
    }


def _edge_summary(edge: OverlayEdge) -> dict[str, Any]:
    return {
        "src": int(edge.src),
        "dst": int(edge.dst),
        "metrics": _metric_summary(edge.metrics),
        "road_changes": int(edge.road_changes),
        "path_nodes": len(edge.path_nodes),
        "bridge_cell_pair": (
            None if edge.bridge_cell_pair is None else list(edge.bridge_cell_pair)
        ),
        "bridge_corridor": (
            None if edge.bridge_corridor is None else list(edge.bridge_corridor)
        ),
    }


def _local_shortcut_pair_examples(state: Any) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[OverlayEdge]] = {}
    for edges in state.overlay.out_edges.values():
        for edge in edges:
            if edge.kind != "local" or edge.bridge_cell_pair is not None:
                continue
            groups.setdefault((int(edge.src), int(edge.dst)), []).append(edge)

    examples: list[dict[str, Any]] = []
    for (src, dst), edges in sorted(groups.items()):
        distinct_paths = {
            edge.path_nodes
            for edge in edges
        }
        if len(distinct_paths) < 2:
            continue
        shortest = min(edges, key=lambda edge: (edge.metrics.length, edge.metrics.elevation))
        quality = min(
            edges,
            key=lambda edge: (
                _avg_width(edge.metrics),
                -_avg_pop(edge.metrics),
                edge.metrics.length,
                edge.metrics.elevation,
                edge.road_changes,
                edge.path_nodes,
            ),
        )
        if shortest.path_nodes == quality.path_nodes:
            continue
        examples.append(
            {
                "src": src,
                "dst": dst,
                "count": len(edges),
                "shortest": _edge_summary(shortest),
                "quality": _edge_summary(quality),
            }
        )
        if len(examples) >= 20:
            break
    return examples


def _archive_routes(
    G: Any,
    partition: dict[int, int],
    constraints: ConstraintBox,
    state: Any,
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for entry in state.archive.entries:
        validation = article._validate_route(
            G,
            partition,
            constraints,
            entry,
            state.config.max_cell_visits_per_route,
        )
        routes.append(longrun._archive_entry_summary(entry, validation))
    return routes


def _write_first_route_nodes(path: str, entry: ArchiveEntry) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "path_nodes": list(entry.path_nodes),
                "cell_sequence": list(entry.cell_sequence),
            },
            fh,
            indent=2,
        )


def _run_query(
    *,
    G: Any,
    partition: dict[int, int],
    boundary_nodes: set[int],
    kept_nodes: set[int],
    constraints: ConstraintBox,
    max_local_states_per_node: int,
    budget_s: float,
    route_nodes_json: str | None = None,
) -> dict[str, Any]:
    config = _make_config(max_local_states_per_node)
    tracker = longrun.CompleteCandidateTracker()
    original_feasible = ps._is_combined_route_feasible

    def wrapped_feasible(state: Any, metrics: RouteAccumulator) -> bool:
        feasible = original_feasible(state, metrics)
        tracker.update(state.query.constraints, metrics, feasible=feasible)
        return feasible

    ps._is_combined_route_feasible = wrapped_feasible
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    try:
        state = anytime_sparse_portal_search(
            G=G,
            partition=partition,
            boundary_nodes=boundary_nodes,
            kept_nodes=kept_nodes,
            query=_make_query(budget_s, constraints),
            config=config,
        )
    finally:
        ps._is_combined_route_feasible = original_feasible

    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - start_cpu
    archive_routes = _archive_routes(G, partition, constraints, state)
    if state.archive.entries and route_nodes_json is not None:
        _write_first_route_nodes(route_nodes_json, state.archive.entries[0])

    fwd_cells = _covered_cells(state, "fwd")
    bwd_cells = _covered_cells(state, "bwd")
    return {
        "name": f"LOCAL{max_local_states_per_node}",
        "budget_s": budget_s,
        "elapsed_wall_s": elapsed_wall,
        "elapsed_cpu_s": elapsed_cpu,
        "cpu_wall_ratio": elapsed_cpu / elapsed_wall if elapsed_wall > 0.0 else None,
        "archive_size": len(state.archive.entries),
        "archive_routes": archive_routes,
        "archive_jaccard": longrun._jaccard_entries(list(state.archive.entries)),
        "frontier_pops": dict(sorted(state.audit.frontier_pops.items())),
        "queue_sizes": {
            "fwd": len(state.frontier["fwd"]),
            "bwd": len(state.frontier["bwd"]),
        },
        "retained_labels": {
            "fwd": _retained_label_count(state, "fwd"),
            "bwd": _retained_label_count(state, "bwd"),
        },
        "portals_reached": {
            "fwd": _covered_portal_count(state, "fwd"),
            "bwd": _covered_portal_count(state, "bwd"),
        },
        "covered_cells": {"fwd": fwd_cells, "bwd": bwd_cells},
        "oracle_contact": longrun._oracle_contact(fwd_cells, bwd_cells),
        "oracle_portal_summaries": {
            str(portal): longrun._portal_label_resource_summary(state, portal)
            for portal in ORACLE_ACTIVE_NODES
        },
        "overlay_edges": _overlay_edge_count(state),
        "bridge_edges_inserted": state.audit.bridge_edges_inserted,
        "complete_join_tests": state.audit.feasibility_checked_on_combined_accumulator,
        "completion_length_lb_rejects": state.audit.rejected_length_completion_lower_bound,
        "best_complete_candidate": tracker.as_dict(),
        "local_audit": {
            "local_states_generated": state.audit.local_states_generated,
            "local_length_rep_updates": state.audit.local_length_rep_updates,
            "local_quality_rep_updates": state.audit.local_quality_rep_updates,
            "local_nodes_with_two_distinct_states": (
                state.audit.local_nodes_with_two_distinct_states
            ),
            "quality_shortcuts_generated": state.audit.quality_shortcuts_generated,
            "quality_shortcuts_insert_attempted": (
                state.audit.quality_shortcuts_insert_attempted
            ),
            "quality_shortcuts_inserted": state.audit.quality_shortcuts_inserted,
            "quality_shortcuts_overlay_rejected": (
                state.audit.quality_shortcuts_overlay_rejected
            ),
            "quality_shortcut_children_generated": (
                state.audit.quality_shortcut_children_generated
            ),
            "quality_shortcut_children_accepted": (
                state.audit.quality_shortcut_children_accepted
            ),
            "quality_shortcut_examples": state.audit.quality_shortcut_examples,
        },
        "local_state_diagnostics": _local_state_diagnostics(state),
        "local_shortcut_pair_examples": _local_shortcut_pair_examples(state),
        "config": _config_dict(config),
        "audit_subset": {
            "rejected_length": state.audit.rejected_length,
            "rejected_elevation": state.audit.rejected_elevation,
            "rejected_avg_popularity": state.audit.rejected_avg_popularity,
            "rejected_avg_width": state.audit.rejected_avg_width,
            "second_cell_visits_allowed": state.audit.second_cell_visits_allowed,
            "rejected_third_cell_visit": state.audit.rejected_third_cell_visit,
        },
    }


def _best_metrics(run: dict[str, Any]) -> dict[str, float] | None:
    best = run["best_complete_candidate"].get("best_near_feasible")
    if not isinstance(best, dict):
        return None
    metrics = best.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _meaningful_local2_improvement(local1: dict[str, Any], local2: dict[str, Any]) -> bool:
    if int(local2["archive_size"]) > 0:
        return True
    m1 = _best_metrics(local1)
    m2 = _best_metrics(local2)
    if m1 is None or m2 is None:
        return False
    width_gain = float(m1["avg_width"]) - float(m2["avg_width"])
    pop_gain = float(m2["avg_pop"]) - float(m1["avg_pop"])
    if width_gain >= 0.25 or pop_gain >= 5.0:
        return True
    if (
        int(local2["local_audit"]["quality_shortcuts_inserted"]) > 0
        and int(local2["local_state_diagnostics"]["final_nodes_with_two_distinct_states"]) > 0
        and width_gain > 0.0
    ):
        return True
    return False


def _print_run_line(run: dict[str, Any]) -> None:
    metrics = _best_metrics(run) or {}
    violations = {}
    best = run["best_complete_candidate"].get("best_near_feasible")
    if isinstance(best, dict) and isinstance(best.get("violations"), dict):
        violations = best["violations"]
    print(
        f"{run['name']} budget={run['budget_s']:.0f}s "
        f"elapsed={run['elapsed_wall_s']:.1f}s "
        f"archive={run['archive_size']} "
        f"pops={run['frontier_pops']} "
        f"local_states={run['local_audit']['local_states_generated']} "
        f"two_state_nodes={run['local_state_diagnostics']['final_nodes_with_two_distinct_states']} "
        f"quality_shortcuts="
        f"{run['local_audit']['quality_shortcuts_generated']}/"
        f"{run['local_audit']['quality_shortcuts_inserted']} "
        f"bridges={run['bridge_edges_inserted']} "
        f"tests={run['complete_join_tests']} "
        f"best_L={float(metrics.get('length', 0.0)):.1f} "
        f"best_H={float(metrics.get('elevation', 0.0)):.1f} "
        f"best_pop={float(metrics.get('avg_pop', 0.0)):.2f} "
        f"best_width={float(metrics.get('avg_width', 0.0)):.2f} "
        f"viol={violations}",
        flush=True,
    )


def _load_problem() -> tuple[Any, Any, list[int], dict[int, int], set[int], set[int], ConstraintBox]:
    return longrun._load_problem()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paris-Bures bounded multi-resource local-search experiment."
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--route-nodes-json", default=DEFAULT_ROUTE_NODES_JSON)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--force-long", action="store_true")
    parser.add_argument("--long-budget-s", type=float, default=3600.0)
    parser.add_argument(
        "--budgets",
        default=",".join(str(int(budget)) for budget in BUDGETS_S),
        help="Comma-separated independent budget seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("brcore.algo.portal_search").setLevel(logging.WARNING)

    G, _, kept_cells, partition, boundary_nodes, kept_nodes, constraints = _load_problem()
    payload: dict[str, Any] = {
        "query": {
            "source": article.SOURCE,
            "target": article.TARGET,
            "kept_cells": len(kept_cells),
            "kept_nodes": len(kept_nodes),
            "constraints": {
                "lower": list(map(float, article.LOWER)),
                "upper": list(map(float, article.UPPER)),
            },
        },
        "runs": [],
        "long_run_triggered": False,
        "interpretation": None,
    }
    print(
        "LOCAL2 setup "
        f"kept_cells={len(kept_cells)} kept_nodes={len(kept_nodes)}",
        flush=True,
    )

    by_key: dict[tuple[int, float], dict[str, Any]] = {}
    budgets = tuple(
        float(part.strip())
        for part in args.budgets.split(",")
        if part.strip()
    )
    for budget_s in budgets:
        for max_local in (1, 2):
            route_path = args.route_nodes_json if max_local == 2 else None
            run = _run_query(
                G=G,
                partition=partition,
                boundary_nodes=boundary_nodes,
                kept_nodes=kept_nodes,
                constraints=constraints,
                max_local_states_per_node=max_local,
                budget_s=budget_s,
                route_nodes_json=route_path,
            )
            payload["runs"].append(run)
            by_key[(max_local, budget_s)] = run
            _print_run_line(run)
            with open(args.output_json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

    decision_budget = 300.0 if (1, 300.0) in by_key else max(budgets)
    local1_300 = by_key[(1, decision_budget)]
    local2_300 = by_key[(2, decision_budget)]
    run_long = args.force_long or (
        not args.skip_long
        and _meaningful_local2_improvement(local1_300, local2_300)
    )
    payload["long_run_triggered"] = run_long

    if run_long:
        print(
            f"Running LOCAL2 long diagnostic for {args.long_budget_s:.0f}s.",
            flush=True,
        )
        run = _run_query(
            G=G,
            partition=partition,
            boundary_nodes=boundary_nodes,
            kept_nodes=kept_nodes,
            constraints=constraints,
            max_local_states_per_node=2,
            budget_s=args.long_budget_s,
            route_nodes_json=args.route_nodes_json,
        )
        run["name"] = "LOCAL2_LONG"
        payload["runs"].append(run)
        _print_run_line(run)
    else:
        print("Skipping LOCAL2 long run: no meaningful 300s improvement.", flush=True)

    if int(local2_300["archive_size"]) > 0:
        interpretation = "A: LOCAL2 finds a feasible route within the budget sweep."
    elif _meaningful_local2_improvement(local1_300, local2_300):
        interpretation = (
            "B: LOCAL2 materially changes resource quality but still finds no route."
        )
    elif int(local2_300["local_audit"]["quality_shortcuts_generated"]) > 0:
        interpretation = (
            "C/D: LOCAL2 generates quality local states, but no material route-quality "
            "improvement appears at 300s."
        )
    else:
        interpretation = "D: LOCAL2 makes essentially no difference at 300s."
    payload["interpretation"] = interpretation

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Output JSON: {args.output_json}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
