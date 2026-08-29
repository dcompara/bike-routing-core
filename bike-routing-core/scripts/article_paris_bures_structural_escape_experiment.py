from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import time
from typing import Any

from brcore.algo import params
import brcore.algo.portal_search as ps
from brcore.algo.portal_search import (
    ConstraintBox,
    OverlayEdge,
    RouteAccumulator,
    SparsePortalConfig,
    SparsePortalQuery,
    anytime_sparse_portal_search,
)

import article_paris_bures as article
import article_paris_bures_longrun as longrun


DEFAULT_OUTPUT_JSON = "tmp_paris_bures_structural_escape_experiment.json"
SOURCE_CHAIN_PORTALS = (127, 20469, 604, 55, 6297, 896, 249)
ORACLE_CELL_87 = 87


def _make_config(
    *,
    max_local_states_per_node: int,
    use_structural_escape_resume: bool,
) -> SparsePortalConfig:
    return replace(
        article._make_config(),
        max_local_states_per_node=max_local_states_per_node,
        use_structural_escape_resume=use_structural_escape_resume,
    )


def _make_query(time_budget_s: float, constraints: ConstraintBox) -> SparsePortalQuery:
    return SparsePortalQuery(
        source=article.SOURCE,
        target=article.TARGET,
        constraints=constraints,
        time_budget_s=time_budget_s,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )


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


def _metric_summary(metrics: RouteAccumulator) -> dict[str, float]:
    x = metrics.route_vector()
    return {
        "length": float(x[0]),
        "elevation": float(x[1]),
        "avg_pop": float(x[2]),
        "avg_width": float(x[3]),
    }


def _path_edges_are_real(state: Any, path_nodes: tuple[int, ...]) -> bool:
    return all(
        any(int(nxt) == int(v) for nxt in state.G.neighbors(int(u))[0])
        for u, v in zip(path_nodes, path_nodes[1:])
    )


def _edge_summary(state: Any, edge: OverlayEdge) -> dict[str, Any]:
    return {
        "src": int(edge.src),
        "dst": int(edge.dst),
        "src_cell": state.trace_cell(edge.src),
        "dst_cell": state.trace_cell(edge.dst),
        "kind": edge.kind,
        "quality": id(edge) in state.quality_local_shortcut_edge_ids,
        "metrics": _metric_summary(edge.metrics),
        "road_changes": int(edge.road_changes),
        "path_nodes": len(edge.path_nodes),
        "real_directed_path": _path_edges_are_real(state, edge.path_nodes),
    }


def _structural_escape_cells_for_portal(
    state: Any,
    portal: int,
    direction: str,
) -> list[int]:
    current_cell = state.trace_cell(portal)
    cells: set[int] = set()
    if direction == "fwd":
        for edge in state.overlay.out_edges.get(portal, []):
            dst_cell = state.trace_cell(edge.dst)
            if dst_cell != current_cell:
                cells.add(dst_cell)
                continue
            for next_edge in state.overlay.out_edges.get(edge.dst, []):
                next_cell = state.trace_cell(next_edge.dst)
                if next_cell != current_cell:
                    cells.add(next_cell)
        return sorted(cells)

    for edge in state.overlay.in_edges.get(portal, []):
        src_cell = state.trace_cell(edge.src)
        if src_cell != current_cell:
            cells.add(src_cell)
            continue
        for prev_edge in state.overlay.in_edges.get(edge.src, []):
            prev_cell = state.trace_cell(prev_edge.src)
            if prev_cell != current_cell:
                cells.add(prev_cell)
    return sorted(cells)


def _source_chain_summary(state: Any) -> dict[str, Any]:
    portals: dict[str, Any] = {}
    for portal in SOURCE_CHAIN_PORTALS:
        fwd_engine = state.local_engines.get((state.trace_cell(portal), portal, "fwd"))
        portals[str(portal)] = {
            "cell": state.trace_cell(portal),
            "covered_fwd": bool(state.labels["fwd"].get(portal)),
            "fwd_labels": len(state.labels["fwd"].get(portal, [])),
            "raw_out_degree": len(state.overlay.out_edges.get(portal, [])),
            "structural_escape_cells_fwd": _structural_escape_cells_for_portal(
                state,
                portal,
                "fwd",
            ),
            "out_edges": [
                _edge_summary(state, edge)
                for edge in state.overlay.out_edges.get(portal, [])
            ],
            "fwd_engine": (
                None
                if fwd_engine is None
                else {
                    "expansions": fwd_engine.expansions,
                    "frontier": len(fwd_engine.frontier),
                    "exhausted": fwd_engine.exhausted,
                    "discovered_portals": sorted(fwd_engine.discovered_portals),
                }
            ),
        }
    return {
        "portals": portals,
        "has_127_to_55": any(
            edge.dst == 55 for edge in state.overlay.out_edges.get(127, [])
        ),
        "has_55_to_81": any(
            edge.dst == 81 for edge in state.overlay.out_edges.get(55, [])
        ),
        "forced_resume_examples_for_127": [
            example
            for example in state.audit.structural_escape_forced_resume_examples
            if example.get("portal") == 127
        ],
    }


def _adjacent_fwd_bwd_cell_pairs(state: Any) -> list[tuple[int, int]]:
    fwd_cells = set(_covered_cells(state, "fwd"))
    bwd_cells = set(_covered_cells(state, "bwd"))
    pairs: list[tuple[int, int]] = []
    for fwd_cell in sorted(fwd_cells):
        for bwd_cell in sorted(state.retained_cell_neighbors.get(fwd_cell, set())):
            if bwd_cell in bwd_cells:
                pairs.append((fwd_cell, bwd_cell))
    return pairs


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


def _run_query(
    *,
    G: Any,
    partition: dict[int, int],
    boundary_nodes: set[int],
    kept_nodes: set[int],
    constraints: ConstraintBox,
    budget_s: float,
    max_local_states_per_node: int,
    use_structural_escape_resume: bool,
) -> dict[str, Any]:
    config = _make_config(
        max_local_states_per_node=max_local_states_per_node,
        use_structural_escape_resume=use_structural_escape_resume,
    )
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

    fwd_cells = _covered_cells(state, "fwd")
    bwd_cells = _covered_cells(state, "bwd")
    audit = state.audit.as_dict()
    return {
        "name": (
            f"LOCAL{max_local_states_per_node}-"
            f"{'STRUCT' if use_structural_escape_resume else 'OLD'}"
        ),
        "budget_s": budget_s,
        "elapsed_wall_s": elapsed_wall,
        "elapsed_cpu_s": elapsed_cpu,
        "archive_size": len(state.archive.entries),
        "archive_routes": _archive_routes(G, partition, constraints, state),
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
        "adjacent_fwd_bwd_cell_pairs": _adjacent_fwd_bwd_cell_pairs(state),
        "reaches_cell_87_forward": ORACLE_CELL_87 in fwd_cells,
        "overlay_edges": _overlay_edge_count(state),
        "bridges": {
            "pending_pairs_discovered": (
                state.audit.bridge_incremental_pending_pairs_discovered
            ),
            "inserted": state.audit.bridge_edges_inserted,
            "join_successes": state.audit.bridge_join_successes,
        },
        "complete_join_tests": state.audit.feasibility_checked_on_combined_accumulator,
        "same_portal_join_attempts": state.audit.same_portal_join_attempts,
        "one_edge_join_attempts": state.audit.one_edge_join_attempts,
        "best_complete_candidate": tracker.as_dict(),
        "structural_escape_audit": {
            "raw_degree_resume_true": audit["raw_degree_resume_true"],
            "structural_escape_forced_resume": (
                audit["structural_escape_forced_resume"]
            ),
            "structural_escape_cells_found": audit["structural_escape_cells_found"],
            "structural_escape_helper_calls": (
                audit["structural_escape_helper_calls"]
            ),
            "structural_escape_helper_time_s": (
                audit["structural_escape_helper_time_s"]
            ),
            "local_resume_stopped_with_escape": (
                audit["local_resume_stopped_with_escape"]
            ),
            "forced_resume_examples": (
                audit["structural_escape_forced_resume_examples"]
            ),
        },
        "local_audit": {
            "local_states_generated": state.audit.local_states_generated,
            "local_length_rep_updates": state.audit.local_length_rep_updates,
            "local_quality_rep_updates": state.audit.local_quality_rep_updates,
            "local_nodes_with_two_distinct_states": (
                state.audit.local_nodes_with_two_distinct_states
            ),
            "quality_shortcuts_generated": state.audit.quality_shortcuts_generated,
            "quality_shortcuts_inserted": state.audit.quality_shortcuts_inserted,
            "quality_shortcut_children_generated": (
                state.audit.quality_shortcut_children_generated
            ),
            "quality_shortcut_children_accepted": (
                state.audit.quality_shortcut_children_accepted
            ),
        },
        "source_chain": _source_chain_summary(state),
    }


def _print_run_line(run: dict[str, Any]) -> None:
    best = run["best_complete_candidate"].get("best_near_feasible")
    metrics = best.get("metrics") if isinstance(best, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    structural = run["structural_escape_audit"]
    print(
        f"{run['name']} budget={run['budget_s']:.0f}s "
        f"elapsed={run['elapsed_wall_s']:.1f}s "
        f"archive={run['archive_size']} pops={run['frontier_pops']} "
        f"fwd_cells={run['covered_cells']['fwd']} "
        f"adj_pairs={run['adjacent_fwd_bwd_cell_pairs']} "
        f"bridges={run['bridges']['inserted']} "
        f"tests={run['complete_join_tests']} "
        f"forced={structural['structural_escape_forced_resume']} "
        f"stopped_with_escape={structural['local_resume_stopped_with_escape']} "
        f"best_L={float(metrics.get('length', 0.0)):.1f} "
        f"best_H={float(metrics.get('elevation', 0.0)):.1f} "
        f"best_pop={float(metrics.get('avg_pop', 0.0)):.2f} "
        f"best_width={float(metrics.get('avg_width', 0.0)):.2f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paris-Bures structural-escape local-resume experiment."
    )
    parser.add_argument("--budgets", default="10,30,60,300")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("brcore.algo.portal_search").setLevel(logging.WARNING)
    budgets = [
        float(part.strip())
        for part in args.budgets.split(",")
        if part.strip()
    ]

    G, _, kept_cells, partition, boundary_nodes, kept_nodes, constraints = (
        longrun._load_problem()
    )
    payload: dict[str, Any] = {
        "query": {
            "source": article.SOURCE,
            "target": article.TARGET,
            "kept_cells": len(kept_cells),
            "kept_nodes": len(kept_nodes),
        },
        "runs": [],
    }
    print(
        "Structural escape experiment "
        f"kept_cells={len(kept_cells)} kept_nodes={len(kept_nodes)}",
        flush=True,
    )

    for budget_s in budgets:
        for max_local in (1, 2):
            for use_structural in (False, True):
                run = _run_query(
                    G=G,
                    partition=partition,
                    boundary_nodes=boundary_nodes,
                    kept_nodes=kept_nodes,
                    constraints=constraints,
                    budget_s=budget_s,
                    max_local_states_per_node=max_local,
                    use_structural_escape_resume=use_structural,
                )
                payload["runs"].append(run)
                _print_run_line(run)

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
