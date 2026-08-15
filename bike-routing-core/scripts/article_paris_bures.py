from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from brcore.algo import params
from brcore.algo.coords import build_local_xy_int
from brcore.algo.portal_search import (
    ArchiveEntry,
    ConstraintBox,
    RouteAccumulator,
    SparsePortalConfig,
    SparsePortalQuery,
    anytime_sparse_portal_search,
)
from brcore.algo.search_space_reduction import search_space_reduction
from brcore.graph.compact import CompactDiGraph
from brcore.io.load_plot_xy import load_xy_graph
from brcore.io.loaders import (
    load_boundary_edges,
    load_boundary_nodes,
    load_partition,
    load_seeds,
)


GRAPH_PATH = "data/graph_Paris_south_4_objectives.xy"
SEEDS_PATH = "data/seeds.txt"
PARTITION_PATH = "data/paris_voronoi_nodes.txt"
BOUNDARY_NODES_PATH = "data/paris_voronoi_boundary_nodes.txt"
BOUNDARY_EDGES_PATH = "data/paris_voronoi_boundaries.txt"

SOURCE = 127
TARGET = 4433
BUDGETS_S = (0.5, 1.0, 2.0, 5.0, 10.0)
CORRIDOR_SLACK_M = 1500
MAX_HOPS_FROM_BOUNDARY = 1

LOWER = np.array([30000.0, 400.0, 150.0, 5.0], dtype=np.float64)
UPPER = np.array([35000.0, 500.0, 255.0, 15.0], dtype=np.float64)


@dataclass(frozen=True)
class RouteValidation:
    missing_edges: int
    ambiguous_edges: int
    length_delta: float
    elevation_delta: float
    avg_popularity_delta: float
    avg_width_delta: float
    road_changes_delta: int
    feasible: bool
    cell_visit_limit_ok: bool
    max_cell_visit_count: int
    twice_visited_cells: tuple[int, ...]

    @property
    def passed(self) -> bool:
        tolerance = 1e-4
        return (
            self.missing_edges == 0
            and abs(self.length_delta) <= tolerance
            and abs(self.elevation_delta) <= tolerance
            and abs(self.avg_popularity_delta) <= tolerance
            and abs(self.avg_width_delta) <= tolerance
            and self.road_changes_delta == 0
            and self.feasible
            and self.cell_visit_limit_ok
        )


@dataclass(frozen=True)
class BudgetSummary:
    budget_s: float
    elapsed_s: float
    archive_size: int
    fwd_pops: int
    bwd_pops: int
    active_portals: int
    overlay_edges: int
    forward_directional_repairs_attempted: int
    forward_directional_edges_inserted: int
    forward_directional_children_generated: int
    bridge_refinements_attempted: int
    bridge_edges_inserted: int
    bridge_join_successes: int
    completion_lb_rejects: int
    second_cell_visits_allowed: int
    rejected_third_cell_visit: int
    all_valid: bool | None
    max_jaccard: float
    max_forward_local_expansions: int


def _overlay_edge_count(state) -> int:
    return sum(len(edges) for edges in state.overlay.out_edges.values())


def _node_lon_lat(nodes: np.ndarray, node: int) -> tuple[float, float]:
    lon = float(nodes[node, 0]) * 1e-6
    lat = float(nodes[node, 1]) * 1e-6
    return lon, lat


def _edge_indices(G: CompactDiGraph, u: int, v: int) -> list[int]:
    start = int(G.offsets[u])
    end = int(G.offsets[u + 1])
    return [idx for idx in range(start, end) if int(G.to[idx]) == v]


def _recompute_metrics(
    G: CompactDiGraph,
    path_nodes: Iterable[int],
) -> tuple[RouteAccumulator, int, int, int]:
    nodes = tuple(path_nodes)
    missing_edges = 0
    ambiguous_edges = 0
    road_changes = 0
    previous_road_id: int | None = None
    metrics = RouteAccumulator()

    for u, v in zip(nodes, nodes[1:]):
        matches = _edge_indices(G, int(u), int(v))
        if not matches:
            missing_edges += 1
            continue
        if len(matches) > 1:
            ambiguous_edges += 1
        edge_idx = matches[0]
        metrics = metrics.plus(RouteAccumulator.from_edge_weights(G.w[edge_idx]))
        road_id = int(G.road_id[edge_idx])
        if previous_road_id is not None and road_id != previous_road_id:
            road_changes += 1
        previous_road_id = road_id

    return metrics, road_changes, missing_edges, ambiguous_edges


def _compressed_cell_sequence(
    partition: dict[int, int],
    path_nodes: Iterable[int],
) -> tuple[int, ...]:
    sequence: list[int] = []
    previous_cell: int | None = None
    for node in path_nodes:
        cell = partition.get(int(node), -(int(node) + 1))
        if cell != previous_cell:
            sequence.append(cell)
            previous_cell = cell
    return tuple(sequence)


def _cell_visit_counts(cell_sequence: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cell in cell_sequence:
        counts[int(cell)] = counts.get(int(cell), 0) + 1
    return counts


def _validate_route(
    G: CompactDiGraph,
    partition: dict[int, int],
    box: ConstraintBox,
    entry: ArchiveEntry,
    max_cell_visits_per_route: int,
) -> RouteValidation:
    recomputed, road_changes, missing_edges, ambiguous_edges = _recompute_metrics(
        G,
        entry.path_nodes,
    )
    actual_vec = recomputed.route_vector()
    entry_vec = entry.metrics.route_vector()
    cell_sequence = _compressed_cell_sequence(partition, entry.path_nodes)
    cell_visit_counts = _cell_visit_counts(cell_sequence)
    return RouteValidation(
        missing_edges=missing_edges,
        ambiguous_edges=ambiguous_edges,
        length_delta=float(actual_vec[0] - entry_vec[0]),
        elevation_delta=float(actual_vec[1] - entry_vec[1]),
        avg_popularity_delta=float(actual_vec[2] - entry_vec[2]),
        avg_width_delta=float(actual_vec[3] - entry_vec[3]),
        road_changes_delta=road_changes - entry.road_changes,
        feasible=box.is_feasible(recomputed),
        cell_visit_limit_ok=all(
            count <= max_cell_visits_per_route
            for count in cell_visit_counts.values()
        ),
        max_cell_visit_count=max(cell_visit_counts.values(), default=0),
        twice_visited_cells=tuple(
            sorted(cell for cell, count in cell_visit_counts.items() if count == 2)
        ),
    )


def _jaccard(left: ArchiveEntry, right: ArchiveEntry) -> float:
    left_nodes = set(left.path_nodes)
    right_nodes = set(right.path_nodes)
    union = left_nodes | right_nodes
    if not union:
        return 1.0
    return len(left_nodes & right_nodes) / len(union)


def _max_pairwise_jaccard(entries: list[ArchiveEntry]) -> float:
    best = 0.0
    for i, left in enumerate(entries):
        for right in entries[i + 1 :]:
            best = max(best, _jaccard(left, right))
    return best


def _covered_cells(state, direction: str) -> list[int]:
    return sorted(
        {
            state.trace_cell(portal)
            for portal, labels in state.labels[direction].items()
            if labels
        }
    )


def _covered_adjacent_cell_pairs(
    G: CompactDiGraph,
    partition: dict[int, int],
    kept_nodes: set[int],
    forward_cells: list[int],
    backward_cells: list[int],
) -> list[tuple[int, int]]:
    forward_cell_set = set(forward_cells)
    backward_cell_set = set(backward_cells)
    adjacent_pairs: set[tuple[int, int]] = set()
    for u in kept_nodes:
        src_cell = partition.get(int(u))
        if src_cell is None:
            continue
        to, _, _ = G.neighbors(int(u))
        for v in to:
            dst_cell = partition.get(int(v))
            if dst_cell is None or dst_cell == src_cell:
                continue
            if src_cell in forward_cell_set and dst_cell in backward_cell_set:
                adjacent_pairs.add((src_cell, dst_cell))
            if src_cell in backward_cell_set and dst_cell in forward_cell_set:
                adjacent_pairs.add((dst_cell, src_cell))
    return sorted(adjacent_pairs)


def _max_forward_local_expansions(state) -> int:
    expansions = [
        engine.expansions
        for (_, _, direction), engine in state.local_engines.items()
        if direction == "fwd"
    ]
    return max(expansions, default=0)


def _print_node_info(
    nodes: np.ndarray,
    partition: dict[int, int],
    boundary_nodes: set[int],
    seeds: list[int],
) -> None:
    seed_set = set(seeds)
    print("Query endpoints:")
    for name, node in (("source", SOURCE), ("target", TARGET)):
        lon, lat = _node_lon_lat(nodes, node)
        print(
            f"  {name}: node={node} lon={lon:.6f} lat={lat:.6f} "
            f"cell={partition.get(node)} boundary={node in boundary_nodes} "
            f"seed={node in seed_set}"
        )


def _print_no_route_diagnostics(state) -> None:
    audit = state.audit
    print("  no feasible archived route; diagnostics:")
    print(f"    rejected_length={audit.rejected_length}")
    print(f"    rejected_elevation={audit.rejected_elevation}")
    print(f"    rejected_avg_popularity={audit.rejected_avg_popularity}")
    print(f"    rejected_avg_width={audit.rejected_avg_width}")
    print(
        "    rejected_length_completion_lower_bound="
        f"{audit.rejected_length_completion_lower_bound}"
    )
    print(f"    second_cell_visits_allowed={audit.second_cell_visits_allowed}")
    print(f"    rejected_third_cell_visit={audit.rejected_third_cell_visit}")
    print(f"    frontier_pops={dict(sorted(audit.frontier_pops.items()))}")
    print(f"    bridge_refinements_attempted={audit.bridge_refinements_attempted}")
    print(f"    bridge_edges_inserted={audit.bridge_edges_inserted}")
    print(f"    bridge_join_successes={audit.bridge_join_successes}")


def _print_route(
    idx: int,
    entry: ArchiveEntry,
    validation: RouteValidation,
) -> None:
    route_vec = entry.metrics.route_vector()
    print(
        f"  route {idx}: score={entry.score:.4f} "
        f"length={route_vec[0]:.1f} "
        f"elev_m={route_vec[1]:.1f} "
        f"avg_pop={route_vec[2]:.2f} "
        f"avg_width={route_vec[3]:.2f} "
        f"road_changes={entry.road_changes} "
        f"path_nodes={len(entry.path_nodes)}"
    )
    print(f"    cell_sequence={entry.cell_sequence}")
    print(f"    bridge_cell_pairs={entry.bridge_cell_pairs}")
    print(f"    bridge_corridors={entry.bridge_corridors}")
    print(
        "    validation="
        f"passed={validation.passed} "
        f"missing_edges={validation.missing_edges} "
        f"ambiguous_edges={validation.ambiguous_edges} "
        f"length_delta={validation.length_delta:.6g} "
        f"elev_delta={validation.elevation_delta:.6g} "
        f"pop_delta={validation.avg_popularity_delta:.6g} "
        f"width_delta={validation.avg_width_delta:.6g} "
        f"road_changes_delta={validation.road_changes_delta} "
        f"feasible={validation.feasible} "
        f"cell_visit_limit_ok={validation.cell_visit_limit_ok} "
        f"max_cell_visit_count={validation.max_cell_visit_count} "
        f"twice_visited_cells={validation.twice_visited_cells}"
    )


def _print_diversity(entries: list[ArchiveEntry]) -> float:
    if len(entries) < 2:
        print("  diversity: fewer than two archived routes")
        return 0.0
    max_jaccard = _max_pairwise_jaccard(entries)
    cell_sequences = {entry.cell_sequence for entry in entries}
    bridge_pairs = {
        pair
        for entry in entries
        for pair in entry.bridge_cell_pairs
    }
    bridge_corridors = {
        corridor
        for entry in entries
        for corridor in entry.bridge_corridors
    }
    print(f"  diversity: max_pairwise_jaccard={max_jaccard:.3f}")
    print(f"    distinct_cell_sequences={sorted(cell_sequences)}")
    print(f"    distinct_bridge_cell_pairs={sorted(bridge_pairs)}")
    print(f"    distinct_bridge_corridors={sorted(bridge_corridors)}")
    return max_jaccard


def _make_config() -> SparsePortalConfig:
    return SparsePortalConfig(
        max_active_portals_per_cell=params.MAX_ACTIVE_PORTALS_PER_CELL,
        max_labels_per_portal=params.MAX_LABELS_PER_PORTAL,
        max_shortcuts_per_pair=params.MAX_SHORTCUTS_PER_PAIR,
        local_expand_limit=params.LOCAL_EXPAND_LIMIT,
        advance_round_budget=params.ADVANCE_ROUND_BUDGET,
        max_cell_visits_per_route=2,
    )


def main() -> None:
    xy = load_xy_graph(GRAPH_PATH)
    G = xy.G
    nodes = xy.nodes
    xy_int = build_local_xy_int(nodes)
    seeds = load_seeds(SEEDS_PATH, id_mode="xy")
    partition = load_partition(PARTITION_PATH, id_mode="xy")
    boundary_nodes = load_boundary_nodes(BOUNDARY_NODES_PATH, id_mode="xy")
    load_boundary_edges(BOUNDARY_EDGES_PATH, id_mode="xy", has_key=True)

    _print_node_info(nodes, partition, boundary_nodes, seeds)

    constraints = ConstraintBox.from_bounds(
        lower=LOWER,
        upper=UPPER,
        weights=params.W,
    )

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
    print("Stage-1 reduction:")
    print(f"  kept_cells={len(kept_cells)}")
    print(f"  kept_nodes={len(kept_nodes)}")
    print()

    summaries: list[BudgetSummary] = []
    archived_validation_results: list[bool] = []

    for budget_s in BUDGETS_S:
        query = SparsePortalQuery(
            source=SOURCE,
            target=TARGET,
            constraints=constraints,
            time_budget_s=budget_s,
            archive_size=params.DEFAULT_ARCHIVE_SIZE,
        )
        start = time.perf_counter()
        state = anytime_sparse_portal_search(
            G=G,
            partition=partition,
            boundary_nodes=boundary_nodes,
            kept_nodes=kept_nodes,
            query=query,
            config=_make_config(),
        )
        elapsed_s = time.perf_counter() - start
        archive_entries = list(state.archive.entries)
        validations = [
            _validate_route(
                G,
                partition,
                constraints,
                entry,
                max_cell_visits_per_route=state.config.max_cell_visits_per_route,
            )
            for entry in archive_entries
        ]
        all_valid = (
            all(validation.passed for validation in validations)
            if validations
            else None
        )
        archived_validation_results.extend(
            validation.passed for validation in validations
        )
        max_jaccard = _max_pairwise_jaccard(archive_entries)

        print(f"Budget {budget_s:.1f}s:")
        print(f"  elapsed_s={elapsed_s:.3f}")
        print(f"  archive_size={len(archive_entries)}")
        print(
            "  frontier_pops="
            f"fwd={state.audit.frontier_pops.get('fwd', 0)} "
            f"bwd={state.audit.frontier_pops.get('bwd', 0)}"
        )
        print(f"  active_portals={len(state.active_portals)}")
        print(f"  overlay_edges={_overlay_edge_count(state)}")
        forward_cells = _covered_cells(state, "fwd")
        backward_cells = _covered_cells(state, "bwd")
        covered_adjacent_pairs = _covered_adjacent_cell_pairs(
            G,
            partition,
            state.kept_nodes,
            forward_cells,
            backward_cells,
        )
        max_forward_expansions = _max_forward_local_expansions(state)
        print(f"  forward_covered_cells={forward_cells}")
        print(f"  backward_covered_cells={backward_cells}")
        print(f"  covered_adjacent_fwd_bwd_cell_pairs={covered_adjacent_pairs}")
        print(f"  pending_bridge_cell_pairs={sorted(state.pending_bridge_cell_pairs)}")
        print(f"  max_forward_local_engine_expansions={max_forward_expansions}")
        print(
            "  forward_directional_repair="
            f"attempted={state.audit.forward_directional_portal_refinements_attempted} "
            f"inserted={state.audit.forward_directional_portal_refinements_inserted} "
            f"children={state.audit.forward_directional_repair_children_generated}"
        )
        print(
            "  bridge="
            f"attempted={state.audit.bridge_refinements_attempted} "
            f"inserted={state.audit.bridge_edges_inserted} "
            f"join_successes={state.audit.bridge_join_successes}"
        )
        print(
            "  cell_visit_pruning="
            f"max={state.config.max_cell_visits_per_route} "
            f"second_allowed={state.audit.second_cell_visits_allowed} "
            f"third_rejected={state.audit.rejected_third_cell_visit} "
            f"completion_lb_rejected={state.audit.rejected_length_completion_lower_bound}"
        )

        if not archive_entries:
            _print_no_route_diagnostics(state)
        for idx, (entry, validation) in enumerate(
            zip(archive_entries, validations),
            start=1,
        ):
            _print_route(idx, entry, validation)
        _print_diversity(archive_entries)
        print()

        summaries.append(
            BudgetSummary(
                budget_s=budget_s,
                elapsed_s=elapsed_s,
                archive_size=len(archive_entries),
                fwd_pops=state.audit.frontier_pops.get("fwd", 0),
                bwd_pops=state.audit.frontier_pops.get("bwd", 0),
                active_portals=len(state.active_portals),
                overlay_edges=_overlay_edge_count(state),
                forward_directional_repairs_attempted=(
                    state.audit.forward_directional_portal_refinements_attempted
                ),
                forward_directional_edges_inserted=(
                    state.audit.forward_directional_portal_refinements_inserted
                ),
                forward_directional_children_generated=(
                    state.audit.forward_directional_repair_children_generated
                ),
                bridge_refinements_attempted=state.audit.bridge_refinements_attempted,
                bridge_edges_inserted=state.audit.bridge_edges_inserted,
                bridge_join_successes=state.audit.bridge_join_successes,
                completion_lb_rejects=(
                    state.audit.rejected_length_completion_lower_bound
                ),
                second_cell_visits_allowed=state.audit.second_cell_visits_allowed,
                rejected_third_cell_visit=state.audit.rejected_third_cell_visit,
                all_valid=all_valid,
                max_jaccard=max_jaccard,
                max_forward_local_expansions=max_forward_expansions,
            )
        )

    print("Budget summary:")
    print(
        "  budget_s elapsed_s archive fwd_pops bwd_pops active_portals "
        "overlay_edges fwd_repair_attempted fwd_repair_inserted "
        "fwd_repair_children bridge_attempted bridge_inserted bridge_successes "
        "completion_lb_rejects second_visits third_rejects "
        "validation max_jaccard max_fwd_local_exp"
    )
    for summary in summaries:
        print(
            f"  {summary.budget_s:>7.1f} "
            f"{summary.elapsed_s:>8.3f} "
            f"{summary.archive_size:>7} "
            f"{summary.fwd_pops:>8} "
            f"{summary.bwd_pops:>8} "
            f"{summary.active_portals:>14} "
            f"{summary.overlay_edges:>13} "
            f"{summary.forward_directional_repairs_attempted:>22} "
            f"{summary.forward_directional_edges_inserted:>21} "
            f"{summary.forward_directional_children_generated:>19} "
            f"{summary.bridge_refinements_attempted:>16} "
            f"{summary.bridge_edges_inserted:>15} "
            f"{summary.bridge_join_successes:>16} "
            f"{summary.completion_lb_rejects:>21} "
            f"{summary.second_cell_visits_allowed:>13} "
            f"{summary.rejected_third_cell_visit:>13} "
            f"{str(summary.all_valid) if summary.all_valid is not None else 'N/A':>10} "
            f"{summary.max_jaccard:>11.3f} "
            f"{summary.max_forward_local_expansions:>17}"
        )

    budgets_with_routes = [
        summary.budget_s for summary in summaries if summary.archive_size >= 1
    ]
    budgets_with_three_routes = [
        summary.budget_s for summary in summaries if summary.archive_size >= 3
    ]
    smallest_route_budget = min(budgets_with_routes) if budgets_with_routes else None
    smallest_three_route_budget = (
        min(budgets_with_three_routes) if budgets_with_three_routes else None
    )
    last_nonempty = next(
        (
            summary
            for summary in reversed(summaries)
            if summary.archive_size > 0
        ),
        None,
    )
    satisfactory_diversity = (
        last_nonempty is not None
        and last_nonempty.archive_size >= 2
        and last_nonempty.max_jaccard < 0.85
    )
    print("Final assessment:")
    print(f"  feasible_route_found={smallest_route_budget is not None}")
    print(f"  smallest_budget_with_route={smallest_route_budget}")
    print(f"  smallest_budget_with_3_routes={smallest_three_route_budget}")
    validation_report = (
        all(archived_validation_results)
        if archived_validation_results
        else "N/A (no archived route)"
    )
    print(f"  all_archived_routes_passed_csr_reconstruction={validation_report}")
    print(f"  diversity_satisfactory={satisfactory_diversity}")


if __name__ == "__main__":
    main()
