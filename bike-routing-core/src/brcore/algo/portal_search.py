from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import logging
import time
from typing import Dict, Iterable, List, Literal, Sequence, Set, Tuple, TypeVar, TypedDict

import numpy as np
from numpy.typing import ArrayLike

from brcore.algo.params import ChebyshevNorm
from brcore.graph.compact import CompactDiGraph


logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(f"{__name__}.trace")

Direction = Literal["fwd", "bwd"]
CountKey = TypeVar("CountKey", int, str)
CELL93_DIAGNOSTIC_CELL = 93
CELL93_DIAGNOSTIC_PORTAL = 2515
COMPLETION_LENGTH_LB_TOLERANCE_M = 1e-6


class BackwardCrossingExitRow(TypedDict):
    source: Literal["overlay", "reverse_adj"]
    from_node: int
    portal: int
    cell: int
    kind: Literal["inter", "local", "base"]


class Cell93ReachingActivePortalRow(TypedDict, total=False):
    portal: int
    from_node: int
    reason: str
    candidate_length: float
    best_length: float
    path_len: int
    route: str


class Cell93LocalDiag(TypedDict, total=False):
    cell: int
    portal: int
    direction: Direction
    initial_queue_size: int
    local_states_popped: int
    reverse_neighbors_seen: int
    internal_neighbors_considered: int
    skipped_by_cell_boundary: int
    skipped_by_visited_or_local_dominance: int
    reaching_active_portals: List[Cell93ReachingActivePortalRow]
    final_queue_size: int
    engine_exhausted: bool
    inserted_edges_returned: int


@dataclass(frozen=True)
class RouteAccumulator:
    """
    Additive search-state accumulators.

    Notes
    -----
    - length and elevation use the current .xy graph columns directly.
    - popularity and street width are accumulated as length-weighted sums.
    - route-level averages are derived only when needed.
    """

    length: float = 0.0
    elevation: float = 0.0
    popularity_length: float = 0.0
    street_width_length: float = 0.0

    def plus(self, other: "RouteAccumulator") -> "RouteAccumulator":
        return RouteAccumulator(
            length=self.length + other.length,
            elevation=self.elevation + other.elevation,
            popularity_length=self.popularity_length + other.popularity_length,
            street_width_length=self.street_width_length + other.street_width_length,
        )

    def route_vector(self) -> np.ndarray:
        if self.length <= 0.0:
            avg_popularity = 0.0
            avg_street_width = 0.0
        else:
            avg_popularity = self.popularity_length / self.length
            avg_street_width = self.street_width_length / self.length
        return np.array(
            [self.length, self.elevation, avg_popularity, avg_street_width],
            dtype=np.float64,
        )

    def compact_summary(self) -> str:
        x = self.route_vector()
        return (
            f"L={x[0]:.1f} H={x[1]:.1f} "
            f"Pavg={x[2]:.2f} Wavg={x[3]:.2f}"
        )

    def __repr__(self) -> str:
        return f"RouteAccumulator({self.compact_summary()})"

    @staticmethod
    def from_edge_weights(w_row: np.ndarray) -> "RouteAccumulator":
        length = float(w_row[0])
        elevation = float(w_row[1])
        popularity = float(w_row[2])
        street_width = float(w_row[3])
        return RouteAccumulator(
            length=length,
            elevation=elevation,
            popularity_length=length * popularity,
            street_width_length=length * street_width,
        )


BridgeCrossing = Tuple[int, int, RouteAccumulator, int | None]
BridgeFallbackSelectionRole = Literal["primary", "quality", "structural_fill"]


@dataclass(frozen=True)
class ConstraintBox:
    lower: np.ndarray
    upper: np.ndarray
    chebyshev: ChebyshevNorm

    @staticmethod
    def from_bounds(
        lower: ArrayLike,
        upper: ArrayLike,
        weights: ArrayLike,
    ) -> "ConstraintBox":
        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        return ConstraintBox(
            lower=lower_arr,
            upper=upper_arr,
            chebyshev=ChebyshevNorm.from_bounds(lower_arr, upper_arr, weights_arr),
        )

    def is_feasible(self, metrics: RouteAccumulator) -> bool:
        x = metrics.route_vector()
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

    def score(self, metrics: RouteAccumulator) -> float:
        return self.chebyshev.score(metrics.route_vector())


@dataclass(frozen=True)
class SparsePortalQuery:
    source: int
    target: int
    constraints: ConstraintBox
    time_budget_s: float = 0.5
    archive_size: int = 3


@dataclass(frozen=True)
class SparsePortalConfig:
    max_active_portals_per_cell: int = 4
    max_labels_per_portal: int = 4
    max_shortcuts_per_pair: int = 2
    local_expand_limit: int = 64
    advance_round_budget: int = 1
    max_cell_visits_per_route: int = 2
    trace_search: bool = False
    trace_portals: Set[int] | None = None
    trace_cells: Set[int] | None = None
    max_trace_events: int = 500
    max_backward_directional_repairs_per_cell: int = 2
    backward_directional_repair_scan_limit: int = 256
    max_forward_directional_repairs_per_cell: int = 2
    forward_directional_repair_scan_limit: int = 256
    max_bridge_edges_per_cell_pair: int = 2
    bridge_refinement_scan_limit: int = 256
    validate_incremental_bridge_detection: bool = False


@dataclass(frozen=True)
class OverlayEdge:
    src: int
    dst: int
    metrics: RouteAccumulator
    path_nodes: Tuple[int, ...]
    first_road_id: int | None
    last_road_id: int | None
    road_changes: int
    kind: Literal["inter", "local"]
    bridge_cell_pair: Tuple[int, int] | None = None
    bridge_corridor: Tuple[int, int] | None = None

    def compact_summary(
        self,
        *,
        src_cell: int | None = None,
        dst_cell: int | None = None,
    ) -> str:
        src = f"{self.src}@{src_cell}" if src_cell is not None else str(self.src)
        dst = f"{self.dst}@{dst_cell}" if dst_cell is not None else str(self.dst)
        return (
            f"{src}->{dst} kind={self.kind} metrics=({self.metrics.compact_summary()}) "
            f"road_changes={self.road_changes}"
        )

    def __repr__(self) -> str:
        return f"OverlayEdge({self.compact_summary()})"


def _overlay_edge_capacity_key(edge: OverlayEdge) -> Tuple[float, float]:
    return (edge.metrics.length, edge.metrics.elevation)


@dataclass(frozen=True)
class PortalLabel:
    portal: int
    direction: Direction
    metrics: RouteAccumulator
    priority: float
    visited_cells: frozenset[int]
    revisited_cells: frozenset[int]
    last_road_id: int | None
    road_changes: int
    parent: "PortalLabel | None" = None
    parent_edge: OverlayEdge | None = None

    def compact_summary(self, *, cell: int | None = None) -> str:
        portal = f"{self.portal}@{cell}" if cell is not None else str(self.portal)
        return (
            f"portal={portal} dir={self.direction} "
            f"metrics=({self.metrics.compact_summary()}) score={self.priority:.4f} "
            f"visited_cells={len(self.visited_cells)} "
            f"revisited_cells={len(self.revisited_cells)} "
            f"road_changes={self.road_changes}"
        )

    def __repr__(self) -> str:
        return f"PortalLabel({self.compact_summary()})"


@dataclass(frozen=True)
class CellHistoryUpdate:
    visited_cells: frozenset[int]
    revisited_cells: frozenset[int]
    second_visits_added: int


@dataclass
class LocalCellEngineState:
    cell_id: int
    origin_portal: int
    direction: Direction
    frontier: List[tuple] = field(default_factory=list)
    best_length: Dict[int, float] = field(default_factory=dict)
    discovered_portals: Set[int] = field(default_factory=set)
    exhausted: bool = False
    expansions: int = 0


@dataclass(frozen=True)
class BackwardDirectionalRepairCandidate:
    pred: int
    node: int
    pred_cell: int
    local_metrics: RouteAccumulator
    combined_metrics: RouteAccumulator
    path_nodes: Tuple[int, ...]
    first_road_id: int | None
    last_road_id: int | None
    road_changes: int


@dataclass(frozen=True)
class ForwardDirectionalRepairCandidate:
    node: int
    dst: int
    dst_cell: int
    local_metrics: RouteAccumulator
    combined_metrics: RouteAccumulator
    path_nodes: Tuple[int, ...]
    first_road_id: int | None
    last_road_id: int | None
    road_changes: int


@dataclass(frozen=True)
class BasePathSegment:
    metrics: RouteAccumulator
    path_nodes: Tuple[int, ...]
    first_road_id: int | None
    last_road_id: int | None
    road_changes: int


@dataclass(frozen=True)
class BridgeRefinementCandidate:
    edge: OverlayEdge
    forward_portal: int
    backward_portal: int
    crossing_src: int
    crossing_dst: int
    joins_covered_directions: bool
    connector_length: float


@dataclass(frozen=True)
class BridgeJoinSelection:
    candidate: BridgeRefinementCandidate
    fwd_label: PortalLabel
    bwd_label: PortalLabel
    metrics: RouteAccumulator
    score: float
    road_changes: int


@dataclass(frozen=True)
class ArchiveEntry:
    join_portal: int
    metrics: RouteAccumulator
    score: float
    path_nodes: Tuple[int, ...]
    road_changes: int
    node_set: frozenset[int]
    cell_sequence: Tuple[int, ...]
    bridge_cell_pairs: Tuple[Tuple[int, int], ...]
    bridge_corridors: Tuple[Tuple[int, int], ...]

    def compact_summary(self, *, cell: int | None = None) -> str:
        portal = (
            f"{self.join_portal}@{cell}"
            if cell is not None
            else str(self.join_portal)
        )
        return (
            f"join_portal={portal} metrics=({self.metrics.compact_summary()}) "
            f"score={self.score:.4f} road_changes={self.road_changes}"
        )

    def __repr__(self) -> str:
        return f"ArchiveEntry({self.compact_summary()})"


@dataclass
class SolutionArchive:
    max_size: int
    entries: List[ArchiveEntry] = field(default_factory=list)

    def add(
        self,
        entry: ArchiveEntry,
        audit: SearchAuditCounters | None = None,
    ) -> bool:
        exact_conflicts: List[ArchiveEntry] = []
        near_conflicts: List[ArchiveEntry] = []
        for existing in self.entries:
            same_path = existing.path_nodes == entry.path_nodes
            same_set_and_corridor = (
                existing.node_set == entry.node_set
                and existing.bridge_corridors == entry.bridge_corridors
            )
            if same_path or same_set_and_corridor:
                exact_conflicts.append(existing)
                continue

            overlap = _archive_node_jaccard(existing, entry)
            shares_corridor = bool(
                set(existing.bridge_corridors) & set(entry.bridge_corridors)
            )
            if overlap > 0.85 or (shares_corridor and overlap > 0.70):
                near_conflicts.append(existing)

        conflicts = exact_conflicts + near_conflicts
        blocking = [
            existing
            for existing in conflicts
            if _archive_entry_sort_key(existing) <= _archive_entry_sort_key(entry)
        ]
        if blocking:
            if audit is not None:
                if any(existing in exact_conflicts for existing in blocking):
                    audit.archive_rejected_exact_duplicate += 1
                else:
                    audit.archive_rejected_high_overlap += 1
                _refresh_archive_diversity_counters(self, audit)
            return False

        conflict_ids = {id(existing) for existing in conflicts}
        pool = [
            existing
            for existing in self.entries
            if id(existing) not in conflict_ids
        ]
        pool.append(entry)
        selected = _select_diverse_archive_entries(pool, self.max_size)
        added = any(selected_entry is entry for selected_entry in selected)
        if not added:
            if audit is not None:
                _refresh_archive_diversity_counters(self, audit)
            return False

        if audit is not None and conflicts:
            audit.archive_replaced_near_duplicate += len(conflicts)
        self.entries = sorted(selected, key=_archive_entry_sort_key)
        if audit is not None:
            _refresh_archive_diversity_counters(self, audit)
        return True


def _archive_entry_sort_key(entry: ArchiveEntry) -> Tuple[float, int]:
    return (entry.score, entry.road_changes)


def _archive_node_jaccard(left: ArchiveEntry, right: ArchiveEntry) -> float:
    union = left.node_set | right.node_set
    if not union:
        return 1.0
    return len(left.node_set & right.node_set) / len(union)


def _archive_pair_bucket(entry: ArchiveEntry) -> tuple:
    if entry.bridge_cell_pairs:
        return ("bridge_pairs", entry.bridge_cell_pairs)
    return ("cell_sequence", entry.cell_sequence)


def _archive_corridor_bucket(entry: ArchiveEntry) -> tuple:
    if entry.bridge_corridors:
        return ("bridge_corridors", entry.bridge_corridors)
    return ("cell_sequence", entry.cell_sequence)


def _select_diverse_archive_entries(
    entries: Sequence[ArchiveEntry],
    limit: int,
) -> List[ArchiveEntry]:
    if limit <= 0:
        return []
    ranked = sorted(entries, key=_archive_entry_sort_key)
    pair_winners: Dict[tuple, ArchiveEntry] = {}
    for entry in ranked:
        pair_winners.setdefault(_archive_pair_bucket(entry), entry)

    selected = sorted(pair_winners.values(), key=_archive_entry_sort_key)[:limit]
    selected_ids = {id(entry) for entry in selected}
    if len(selected) >= limit:
        return selected

    corridor_winners: Dict[tuple, ArchiveEntry] = {}
    for entry in ranked:
        if id(entry) in selected_ids:
            continue
        corridor_winners.setdefault(_archive_corridor_bucket(entry), entry)
    for entry in sorted(corridor_winners.values(), key=_archive_entry_sort_key):
        if len(selected) >= limit:
            break
        selected.append(entry)
        selected_ids.add(id(entry))

    for entry in ranked:
        if len(selected) >= limit:
            break
        if id(entry) not in selected_ids:
            selected.append(entry)
            selected_ids.add(id(entry))
    return selected


def _refresh_archive_diversity_counters(
    archive: SolutionArchive,
    audit: SearchAuditCounters,
) -> None:
    pair_counts: Dict[str, int] = {}
    corridor_counts: Dict[str, int] = {}
    for entry in archive.entries:
        for pair in entry.bridge_cell_pairs:
            key = f"{pair[0]}->{pair[1]}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        for corridor in entry.bridge_corridors:
            key = f"{corridor[0]}->{corridor[1]}"
            corridor_counts[key] = corridor_counts.get(key, 0) + 1
    audit.archive_routes_by_bridge_pair = pair_counts
    audit.archive_routes_by_bridge_corridor = corridor_counts


@dataclass
class BackwardPopTrace:
    pop_index: int
    portal: int
    cell_id: int
    visited_cells: Tuple[int, ...]
    in_edges_count: int
    out_edges_count: int
    usable_backward_edges: int
    local_triggered: bool = False
    local_cell_id: int | None = None
    local_batches_resumed: int = 0
    local_shortcuts_discovered: int = 0
    local_edges_inserted: int = 0
    children_from_local_shortcuts: int = 0
    children_from_existing_overlay: int = 0
    child_rejections: Dict[str, int] = field(default_factory=dict)


@dataclass
class SearchAuditCounters:
    feasibility_checked_on_combined_accumulator: int = 0
    rejected_length: int = 0
    rejected_elevation: int = 0
    rejected_length_completion_lower_bound: int = 0
    rejected_length_completion_lower_bound_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    rejected_avg_popularity: int = 0
    rejected_avg_width: int = 0
    rejected_by_score_prune: int = 0
    same_portal_join_attempts: int = 0
    same_portal_join_successes: int = 0
    one_edge_join_attempts: int = 0
    one_edge_join_successes: int = 0
    one_edge_join_rejected_cell_conflict: int = 0
    one_edge_join_rejected_infeasible: int = 0
    one_edge_join_rejected_reconstruction: int = 0
    one_edge_join_shared_cells_count_histogram: Dict[int, int] = field(default_factory=dict)
    one_edge_join_shared_only_p_cell: int = 0
    one_edge_join_shared_only_q_cell: int = 0
    one_edge_join_shared_only_edge_cells: int = 0
    one_edge_join_shared_multiple_cells: int = 0
    one_edge_join_unexpected_shared_cells: Dict[int, int] = field(default_factory=dict)
    second_cell_visits_allowed: int = 0
    second_cell_visits_allowed_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0, "join": 0}
    )
    rejected_third_cell_visit: int = 0
    rejected_third_cell_visit_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0, "join": 0}
    )
    representative_accept_count: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    representative_accept_reason_count: Dict[str, int] = field(default_factory=dict)
    representative_child_accept_count: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    representative_visited_cells_size_histogram: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: {"fwd": {}, "bwd": {}}
    )
    representative_cell_path_length_histogram: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: {"fwd": {}, "bwd": {}}
    )
    representative_source_cell_far_from_source: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    representative_target_cell_far_from_target: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    local_shortcuts_discovered_by_cell: Dict[int, int] = field(default_factory=dict)
    local_shortcut_ordering_changed_by_road_continuity: int = 0
    terminal_completion_attempts: int = 0
    terminal_completion_successes: int = 0
    terminal_leak_rejected_after_failed_completion: int = 0
    join_attempts_before_terminal_rejection: int = 0
    join_successes_before_terminal_rejection: int = 0
    generated_child_labels_total: int = 0
    generated_child_labels_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    generated_child_labels_with_same_portal_opposite: int = 0
    generated_child_labels_with_one_edge_opposite: int = 0
    terminal_leak_child_with_same_portal_opposite: int = 0
    terminal_leak_child_with_one_edge_opposite: int = 0
    terminal_leak_labels_generated_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    terminal_leak_labels_rejected_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    terminal_leak_same_portal_opposite_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    terminal_leak_one_edge_opposite_by_direction: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    rejected_backward_enter_source_cell: int = 0
    rejected_forward_enter_target_cell: int = 0
    frontier_pops: Dict[str, int] = field(
        default_factory=lambda: {"fwd": 0, "bwd": 0}
    )
    backward_rejection_reason_count: Dict[str, int] = field(default_factory=dict)
    backward_rejected_length: int = 0
    backward_rejected_elevation: int = 0
    backward_rejected_visited_cells: int = 0
    backward_rejected_terminal_anti_leak: int = 0
    backward_rejected_representative_capacity: int = 0
    backward_pop_traces: List[BackwardPopTrace] = field(default_factory=list)
    cell93_diagnostics: Dict[str, object] = field(default_factory=dict)
    backward_dead_portal_repairs_attempted: int = 0
    backward_dead_portal_repairs_inserted: int = 0
    backward_dead_portal_repair_children_generated: int = 0
    repaired_portals_by_cell: Dict[int, int] = field(default_factory=dict)
    backward_children_entering_dead_cell: int = 0
    backward_children_entering_cell_with_only_return_to_previous_cell: int = 0
    backward_dead_cells_by_id: Dict[int, int] = field(default_factory=dict)
    backward_cell93_entry_diagnostics: List[Dict[str, object]] = field(
        default_factory=list
    )
    backward_culdesac_children_pruned: int = 0
    backward_culdesac_cells_pruned_by_id: Dict[int, int] = field(
        default_factory=dict
    )
    backward_directional_portal_refinements_attempted: int = 0
    backward_directional_portal_refinements_inserted: int = 0
    backward_directional_repair_children_generated: int = 0
    backward_directional_repair_candidates_seen: int = 0
    backward_directional_repair_candidates_unvisited: int = 0
    backward_directional_repair_cells: Dict[int, int] = field(default_factory=dict)
    backward_directional_repair_attempted_cells: Dict[int, int] = field(
        default_factory=dict
    )
    backward_directional_repair_budget_exhausted_by_cell: Dict[int, int] = field(
        default_factory=dict
    )
    backward_directional_repair_ordering_changed_by_road_continuity: int = 0
    forward_directional_portal_refinements_attempted: int = 0
    forward_directional_portal_refinements_inserted: int = 0
    forward_directional_repair_children_generated: int = 0
    forward_directional_repair_candidates_seen: int = 0
    forward_directional_repair_candidates_unvisited: int = 0
    forward_directional_repair_cells: Dict[int, int] = field(default_factory=dict)
    forward_directional_repair_attempted_cells: Dict[int, int] = field(
        default_factory=dict
    )
    forward_directional_repair_budget_exhausted_by_cell: Dict[int, int] = field(
        default_factory=dict
    )
    forward_directional_repair_ordering_changed_by_road_continuity: int = 0
    bridge_refinements_attempted: int = 0
    bridge_candidates_reconstructible: int = 0
    bridge_immediate_feasible_selections: int = 0
    bridge_edges_inserted: int = 0
    bridge_join_attempts: int = 0
    bridge_join_successes: int = 0
    bridge_fallback_edges_attempted: int = 0
    bridge_fallback_edges_inserted: int = 0
    bridge_fallback_children_generated: int = 0
    bridge_fallback_children_accepted: int = 0
    complementary_connector_sets_considered: int = 0
    complementary_quality_candidate_distinct: int = 0
    complementary_quality_candidate_insert_attempted: int = 0
    complementary_quality_candidate_inserted: int = 0
    complementary_quality_candidate_rejected_by_overlay_capacity: int = 0
    complementary_quality_candidate_examples: List[Dict[str, object]] = field(
        default_factory=list
    )
    bridge_immediate_join_successes: int = 0
    later_join_successes_through_bridge: int = 0
    bridge_refinement_cell_pairs: Dict[str, int] = field(default_factory=dict)
    bridge_repair_children_generated: int = 0
    bridge_detection_coverage_checks: int = 0
    bridge_detection_calls: int = 0
    bridge_detection_skipped_unchanged_coverage: int = 0
    bridge_detection_total_time_s: float = 0.0
    bridge_incremental_detection_calls: int = 0
    bridge_incremental_new_fwd_cells: int = 0
    bridge_incremental_new_bwd_cells: int = 0
    bridge_incremental_neighbor_lookups: int = 0
    bridge_incremental_pending_pairs_discovered: int = 0
    bridge_incremental_detection_total_time_s: float = 0.0
    bridge_incremental_crosscheck_mismatches: int = 0
    archive_rejected_exact_duplicate: int = 0
    archive_rejected_high_overlap: int = 0
    archive_replaced_near_duplicate: int = 0
    archive_routes_by_bridge_pair: Dict[str, int] = field(default_factory=dict)
    archive_routes_by_bridge_corridor: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "feasibility_checked_on_combined_accumulator": (
                self.feasibility_checked_on_combined_accumulator
            ),
            "rejected_length": self.rejected_length,
            "rejected_elevation": self.rejected_elevation,
            "rejected_length_completion_lower_bound": (
                self.rejected_length_completion_lower_bound
            ),
            "rejected_length_completion_lower_bound_by_direction": dict(
                sorted(self.rejected_length_completion_lower_bound_by_direction.items())
            ),
            "rejected_avg_popularity": self.rejected_avg_popularity,
            "rejected_avg_width": self.rejected_avg_width,
            "rejected_by_score_prune": self.rejected_by_score_prune,
            "same_portal_join_attempts": self.same_portal_join_attempts,
            "same_portal_join_successes": self.same_portal_join_successes,
            "one_edge_join_attempts": self.one_edge_join_attempts,
            "one_edge_join_successes": self.one_edge_join_successes,
            "one_edge_join_rejected_cell_conflict": (
                self.one_edge_join_rejected_cell_conflict
            ),
            "one_edge_join_rejected_infeasible": (
                self.one_edge_join_rejected_infeasible
            ),
            "one_edge_join_rejected_reconstruction": (
                self.one_edge_join_rejected_reconstruction
            ),
            "one_edge_join_shared_cells_count_histogram": dict(
                sorted(self.one_edge_join_shared_cells_count_histogram.items())
            ),
            "one_edge_join_shared_only_p_cell": self.one_edge_join_shared_only_p_cell,
            "one_edge_join_shared_only_q_cell": self.one_edge_join_shared_only_q_cell,
            "one_edge_join_shared_only_edge_cells": (
                self.one_edge_join_shared_only_edge_cells
            ),
            "one_edge_join_shared_multiple_cells": (
                self.one_edge_join_shared_multiple_cells
            ),
            "one_edge_join_unexpected_shared_cells": dict(
                sorted(self.one_edge_join_unexpected_shared_cells.items())
            ),
            "second_cell_visits_allowed": self.second_cell_visits_allowed,
            "second_cell_visits_allowed_by_direction": dict(
                sorted(self.second_cell_visits_allowed_by_direction.items())
            ),
            "rejected_third_cell_visit": self.rejected_third_cell_visit,
            "rejected_third_cell_visit_by_direction": dict(
                sorted(self.rejected_third_cell_visit_by_direction.items())
            ),
            "representative_accept_count": dict(
                sorted(self.representative_accept_count.items())
            ),
            "representative_accept_reason_count": dict(
                sorted(self.representative_accept_reason_count.items())
            ),
            "representative_child_accept_count": dict(
                sorted(self.representative_child_accept_count.items())
            ),
            "representative_visited_cells_size_histogram": {
                direction: dict(sorted(histogram.items()))
                for direction, histogram
                in sorted(self.representative_visited_cells_size_histogram.items())
            },
            "representative_cell_path_length_histogram": {
                direction: dict(sorted(histogram.items()))
                for direction, histogram
                in sorted(self.representative_cell_path_length_histogram.items())
            },
            "representative_source_cell_far_from_source": dict(
                sorted(self.representative_source_cell_far_from_source.items())
            ),
            "representative_target_cell_far_from_target": dict(
                sorted(self.representative_target_cell_far_from_target.items())
            ),
            "local_shortcuts_discovered_by_cell": dict(
                sorted(self.local_shortcuts_discovered_by_cell.items())
            ),
            "local_shortcut_ordering_changed_by_road_continuity": (
                self.local_shortcut_ordering_changed_by_road_continuity
            ),
            "terminal_completion_attempts": self.terminal_completion_attempts,
            "terminal_completion_successes": self.terminal_completion_successes,
            "terminal_leak_rejected_after_failed_completion": (
                self.terminal_leak_rejected_after_failed_completion
            ),
            "join_attempts_before_terminal_rejection": (
                self.join_attempts_before_terminal_rejection
            ),
            "join_successes_before_terminal_rejection": (
                self.join_successes_before_terminal_rejection
            ),
            "generated_child_labels_total": self.generated_child_labels_total,
            "generated_child_labels_by_direction": dict(
                sorted(self.generated_child_labels_by_direction.items())
            ),
            "generated_child_labels_with_same_portal_opposite": (
                self.generated_child_labels_with_same_portal_opposite
            ),
            "generated_child_labels_with_one_edge_opposite": (
                self.generated_child_labels_with_one_edge_opposite
            ),
            "terminal_leak_child_with_same_portal_opposite": (
                self.terminal_leak_child_with_same_portal_opposite
            ),
            "terminal_leak_child_with_one_edge_opposite": (
                self.terminal_leak_child_with_one_edge_opposite
            ),
            "terminal_leak_labels_generated_by_direction": dict(
                sorted(self.terminal_leak_labels_generated_by_direction.items())
            ),
            "terminal_leak_labels_rejected_by_direction": dict(
                sorted(self.terminal_leak_labels_rejected_by_direction.items())
            ),
            "terminal_leak_same_portal_opposite_by_direction": dict(
                sorted(self.terminal_leak_same_portal_opposite_by_direction.items())
            ),
            "terminal_leak_one_edge_opposite_by_direction": dict(
                sorted(self.terminal_leak_one_edge_opposite_by_direction.items())
            ),
            "rejected_backward_enter_source_cell": (
                self.rejected_backward_enter_source_cell
            ),
            "rejected_forward_enter_target_cell": (
                self.rejected_forward_enter_target_cell
            ),
            "frontier_pops": dict(sorted(self.frontier_pops.items())),
            "backward_rejection_reason_count": dict(
                sorted(self.backward_rejection_reason_count.items())
            ),
            "backward_rejected_length": self.backward_rejected_length,
            "backward_rejected_elevation": self.backward_rejected_elevation,
            "backward_rejected_visited_cells": self.backward_rejected_visited_cells,
            "backward_rejected_terminal_anti_leak": (
                self.backward_rejected_terminal_anti_leak
            ),
            "backward_rejected_representative_capacity": (
                self.backward_rejected_representative_capacity
            ),
            "backward_dead_portal_repairs_attempted": (
                self.backward_dead_portal_repairs_attempted
            ),
            "backward_dead_portal_repairs_inserted": (
                self.backward_dead_portal_repairs_inserted
            ),
            "backward_dead_portal_repair_children_generated": (
                self.backward_dead_portal_repair_children_generated
            ),
            "repaired_portals_by_cell": dict(
                sorted(self.repaired_portals_by_cell.items())
            ),
            "backward_children_entering_dead_cell": (
                self.backward_children_entering_dead_cell
            ),
            "backward_children_entering_cell_with_only_return_to_previous_cell": (
                self.backward_children_entering_cell_with_only_return_to_previous_cell
            ),
            "backward_dead_cells_by_id": dict(
                sorted(self.backward_dead_cells_by_id.items())
            ),
            "backward_cell93_entry_diagnostics": (
                self.backward_cell93_entry_diagnostics
            ),
            "backward_culdesac_children_pruned": (
                self.backward_culdesac_children_pruned
            ),
            "backward_culdesac_cells_pruned_by_id": dict(
                sorted(self.backward_culdesac_cells_pruned_by_id.items())
            ),
            "backward_directional_portal_refinements_attempted": (
                self.backward_directional_portal_refinements_attempted
            ),
            "backward_directional_portal_refinements_inserted": (
                self.backward_directional_portal_refinements_inserted
            ),
            "backward_directional_repair_children_generated": (
                self.backward_directional_repair_children_generated
            ),
            "backward_directional_repair_candidates_seen": (
                self.backward_directional_repair_candidates_seen
            ),
            "backward_directional_repair_candidates_unvisited": (
                self.backward_directional_repair_candidates_unvisited
            ),
            "backward_directional_repair_cells": dict(
                sorted(self.backward_directional_repair_cells.items())
            ),
            "backward_directional_repair_attempted_cells": dict(
                sorted(self.backward_directional_repair_attempted_cells.items())
            ),
            "backward_directional_repair_budget_exhausted_by_cell": dict(
                sorted(
                    self.backward_directional_repair_budget_exhausted_by_cell.items()
                )
            ),
            "backward_directional_repair_ordering_changed_by_road_continuity": (
                self.backward_directional_repair_ordering_changed_by_road_continuity
            ),
            "forward_directional_portal_refinements_attempted": (
                self.forward_directional_portal_refinements_attempted
            ),
            "forward_directional_portal_refinements_inserted": (
                self.forward_directional_portal_refinements_inserted
            ),
            "forward_directional_repair_children_generated": (
                self.forward_directional_repair_children_generated
            ),
            "forward_directional_repair_candidates_seen": (
                self.forward_directional_repair_candidates_seen
            ),
            "forward_directional_repair_candidates_unvisited": (
                self.forward_directional_repair_candidates_unvisited
            ),
            "forward_directional_repair_cells": dict(
                sorted(self.forward_directional_repair_cells.items())
            ),
            "forward_directional_repair_attempted_cells": dict(
                sorted(self.forward_directional_repair_attempted_cells.items())
            ),
            "forward_directional_repair_budget_exhausted_by_cell": dict(
                sorted(
                    self.forward_directional_repair_budget_exhausted_by_cell.items()
                )
            ),
            "forward_directional_repair_ordering_changed_by_road_continuity": (
                self.forward_directional_repair_ordering_changed_by_road_continuity
            ),
            "bridge_refinements_attempted": self.bridge_refinements_attempted,
            "bridge_candidates_reconstructible": (
                self.bridge_candidates_reconstructible
            ),
            "bridge_immediate_feasible_selections": (
                self.bridge_immediate_feasible_selections
            ),
            "bridge_edges_inserted": self.bridge_edges_inserted,
            "bridge_join_attempts": self.bridge_join_attempts,
            "bridge_join_successes": self.bridge_join_successes,
            "bridge_fallback_edges_attempted": self.bridge_fallback_edges_attempted,
            "bridge_fallback_edges_inserted": self.bridge_fallback_edges_inserted,
            "bridge_fallback_children_generated": (
                self.bridge_fallback_children_generated
            ),
            "bridge_fallback_children_accepted": (
                self.bridge_fallback_children_accepted
            ),
            "complementary_connector_sets_considered": (
                self.complementary_connector_sets_considered
            ),
            "complementary_quality_candidate_distinct": (
                self.complementary_quality_candidate_distinct
            ),
            "complementary_quality_candidate_insert_attempted": (
                self.complementary_quality_candidate_insert_attempted
            ),
            "complementary_quality_candidate_inserted": (
                self.complementary_quality_candidate_inserted
            ),
            "complementary_quality_candidate_rejected_by_overlay_capacity": (
                self.complementary_quality_candidate_rejected_by_overlay_capacity
            ),
            "complementary_quality_candidate_examples": (
                self.complementary_quality_candidate_examples
            ),
            "bridge_immediate_join_successes": (
                self.bridge_immediate_join_successes
            ),
            "later_join_successes_through_bridge": (
                self.later_join_successes_through_bridge
            ),
            "bridge_refinement_cell_pairs": dict(
                sorted(self.bridge_refinement_cell_pairs.items())
            ),
            "bridge_repair_children_generated": (
                self.bridge_repair_children_generated
            ),
            "bridge_detection_coverage_checks": self.bridge_detection_coverage_checks,
            "bridge_detection_calls": self.bridge_detection_calls,
            "bridge_detection_skipped_unchanged_coverage": (
                self.bridge_detection_skipped_unchanged_coverage
            ),
            "bridge_detection_total_time_s": self.bridge_detection_total_time_s,
            "bridge_incremental_detection_calls": (
                self.bridge_incremental_detection_calls
            ),
            "bridge_incremental_new_fwd_cells": (
                self.bridge_incremental_new_fwd_cells
            ),
            "bridge_incremental_new_bwd_cells": (
                self.bridge_incremental_new_bwd_cells
            ),
            "bridge_incremental_neighbor_lookups": (
                self.bridge_incremental_neighbor_lookups
            ),
            "bridge_incremental_pending_pairs_discovered": (
                self.bridge_incremental_pending_pairs_discovered
            ),
            "bridge_incremental_detection_total_time_s": (
                self.bridge_incremental_detection_total_time_s
            ),
            "bridge_incremental_crosscheck_mismatches": (
                self.bridge_incremental_crosscheck_mismatches
            ),
            "archive_rejected_exact_duplicate": (
                self.archive_rejected_exact_duplicate
            ),
            "archive_rejected_high_overlap": self.archive_rejected_high_overlap,
            "archive_replaced_near_duplicate": (
                self.archive_replaced_near_duplicate
            ),
            "archive_routes_by_bridge_pair": dict(
                sorted(self.archive_routes_by_bridge_pair.items())
            ),
            "archive_routes_by_bridge_corridor": dict(
                sorted(self.archive_routes_by_bridge_corridor.items())
            ),
        }


@dataclass
class OverlayGraph:
    out_edges: Dict[int, List[OverlayEdge]] = field(default_factory=dict)
    in_edges: Dict[int, List[OverlayEdge]] = field(default_factory=dict)


@dataclass
class AnytimeSearchState:
    G: CompactDiGraph
    query: SparsePortalQuery
    config: SparsePortalConfig
    partition: Dict[int, int]
    kept_nodes: Set[int]
    active_portals: Set[int]
    nodes_by_cell: Dict[int, Set[int]]
    reverse_adj: Dict[int, List[Tuple[int, RouteAccumulator, int | None]]]
    retained_cell_neighbors: Dict[int, Set[int]]
    retained_crossings_by_cell_pair: Dict[Tuple[int, int], List[BridgeCrossing]]
    retained_cell_crossing_index_time_s: float
    retained_directed_inter_cell_crossings: int
    retained_directed_cell_pair_count: int
    retained_undirected_cell_pair_count: int
    overlay: OverlayGraph
    archive: SolutionArchive
    labels: Dict[Direction, Dict[int, List[PortalLabel]]]
    frontier: Dict[Direction, List[tuple]]
    local_engines: Dict[Tuple[int, int, Direction], LocalCellEngineState]
    deadline: float
    min_len_from_source: np.ndarray
    min_len_to_target: np.ndarray
    dijkstra_source_time_s: float
    dijkstra_target_time_s: float
    dijkstra_source_finite_count: int
    dijkstra_target_finite_count: int
    audit: SearchAuditCounters = field(default_factory=SearchAuditCounters)
    turn: int = 0
    push_counter: itertools.count = field(default_factory=itertools.count)
    local_counter: itertools.count = field(default_factory=itertools.count)
    trace_event_count: int = 0
    trace_limit_notified: bool = False
    touched_cells: Dict[int, int] = field(default_factory=dict)
    forward_directional_refinement_attempted_portals: Set[Tuple[int, int]] = field(
        default_factory=set
    )
    backward_directional_refinement_attempted_portals: Set[Tuple[int, int]] = field(
        default_factory=set
    )
    forward_directional_repair_edges_by_cell: Dict[int, int] = field(
        default_factory=dict
    )
    backward_directional_repair_edges_by_cell: Dict[int, int] = field(
        default_factory=dict
    )
    pending_bridge_cell_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    last_bridge_detection_fwd_cells: frozenset[int] | None = None
    last_bridge_detection_bwd_cells: frozenset[int] | None = None
    bridge_seen_fwd_cells: Set[int] = field(default_factory=set)
    bridge_seen_bwd_cells: Set[int] = field(default_factory=set)
    bridge_refinement_attempted_cell_pairs: Set[Tuple[int, int]] = field(
        default_factory=set
    )
    bridge_inserted_crossings_by_cell_pair: Dict[
        Tuple[int, int], Set[Tuple[int, int]]
    ] = field(default_factory=dict)
    bridge_representatives_by_cell_pair: Dict[
        Tuple[int, int], List[OverlayEdge]
    ] = field(default_factory=dict)

    def trace_cell(self, node: int) -> int:
        return self.partition.get(node, -(node + 1))

    def partial_priority(self, metrics: RouteAccumulator) -> float:
        if metrics.length <= 0.0:
            return 0.0
        return self.query.constraints.score(metrics)

    def is_partially_hopeless(
        self,
        metrics: RouteAccumulator,
        direction: Direction | None = None,
    ) -> bool:
        # Safe only for additive route quantities. We do not prune on averages here.
        reasons: List[str] = []
        if metrics.length > float(self.query.constraints.upper[0]):
            self.audit.rejected_length += 1
            if direction == "bwd":
                self.audit.backward_rejected_length += 1
                _record_backward_rejection(self, "length")
            reasons.append("length_upper")
        if metrics.elevation > float(self.query.constraints.upper[1]):
            self.audit.rejected_elevation += 1
            if direction == "bwd":
                self.audit.backward_rejected_elevation += 1
                _record_backward_rejection(self, "elevation")
            reasons.append("elevation_upper")
        if reasons:
            logger.debug(
                "Partial prune reject reasons=%s route=%s counters=%s",
                reasons,
                _fmt_route_vector(metrics),
                self.audit.as_dict(),
            )
        return bool(reasons)

    def completion_length_lower_bound(self, label: PortalLabel) -> float:
        portal = int(label.portal)
        if label.direction == "fwd":
            if portal < 0 or portal >= len(self.min_len_to_target):
                return float("inf")
            return float(self.min_len_to_target[portal])
        if portal < 0 or portal >= len(self.min_len_from_source):
            return float("inf")
        return float(self.min_len_from_source[portal])

    def rejects_completion_length_lower_bound(self, label: PortalLabel) -> bool:
        remaining_lb = self.completion_length_lower_bound(label)
        upper_length = float(self.query.constraints.upper[0])
        total_lb = label.metrics.length + remaining_lb
        if np.isfinite(remaining_lb):
            reject = total_lb > upper_length + COMPLETION_LENGTH_LB_TOLERANCE_M
        else:
            reject = True
        if not reject:
            return False

        self.audit.rejected_length_completion_lower_bound += 1
        _increment_count(
            self.audit.rejected_length_completion_lower_bound_by_direction,
            label.direction,
        )
        if label.direction == "bwd":
            _record_backward_rejection(self, "length_completion_lower_bound")
        logger.debug(
            "Completion length lower-bound reject portal=%s dir=%s "
            "accumulated_length=%.3f remaining_lb=%s total_lb=%s Lmax=%.3f",
            label.portal,
            label.direction,
            label.metrics.length,
            f"{remaining_lb:.3f}" if np.isfinite(remaining_lb) else "inf",
            f"{total_lb:.3f}" if np.isfinite(total_lb) else "inf",
            upper_length,
        )
        return True

    def enqueue(self, label: PortalLabel) -> bool:
        if self.rejects_completion_length_lower_bound(label):
            return False
        heapq.heappush(
            self.frontier[label.direction],
            (label.priority, label.road_changes, next(self.push_counter), label),
        )
        return True

    def add_overlay_edge(self, edge: OverlayEdge) -> bool:
        bucket = self.overlay.out_edges.setdefault(edge.src, [])
        same_pair = [e for e in bucket if e.dst == edge.dst and e.kind == edge.kind]

        if any(existing.path_nodes == edge.path_nodes for existing in same_pair):
            return False

        if len(same_pair) >= self.config.max_shortcuts_per_pair:
            edge_key = _overlay_edge_capacity_key(edge)
            worst = max(same_pair, key=_overlay_edge_capacity_key)
            worst_key = _overlay_edge_capacity_key(worst)
            if edge_key >= worst_key:
                logger.debug(
                    "Overlay shortcut reject src=%s dst=%s kind=%s reason=pair_capacity "
                    "edge_key=%s worst_key=%s rejected_by_score_prune=%s",
                    edge.src,
                    edge.dst,
                    edge.kind,
                    edge_key,
                    worst_key,
                    self.audit.rejected_by_score_prune,
                )
                return False
            logger.debug(
                "Overlay shortcut replace src=%s dst=%s kind=%s reason=pair_capacity "
                "edge_key=%s worst_key=%s rejected_by_score_prune=%s",
                edge.src,
                edge.dst,
                edge.kind,
                edge_key,
                worst_key,
                self.audit.rejected_by_score_prune,
            )
            bucket.remove(worst)
            self.overlay.in_edges[worst.dst].remove(worst)

        bucket.append(edge)
        self.overlay.in_edges.setdefault(edge.dst, []).append(edge)
        return True


def _fmt_route_vector(metrics: RouteAccumulator) -> str:
    x = metrics.route_vector()
    return (
        f"[L={x[0]:.1f}, H={x[1]:.1f}, "
        f"Pavg={x[2]:.2f}, Wavg={x[3]:.2f}]"
    )


def _trace_filter_matches(
    state: AnytimeSearchState,
    *,
    portals: Iterable[int] = (),
    cells: Iterable[int] = (),
) -> bool:
    portal_filter = state.config.trace_portals
    cell_filter = state.config.trace_cells
    if portal_filter is None and cell_filter is None:
        return True

    event_portals = set(portals)
    event_cells = set(cells)
    return bool(
        (portal_filter is not None and event_portals & portal_filter)
        or (cell_filter is not None and event_cells & cell_filter)
    )


def _emit_trace_event(
    state: AnytimeSearchState,
    event: str,
    *,
    portals: Iterable[int] = (),
    cells: Iterable[int] = (),
    always: bool = False,
    **fields: object,
) -> None:
    if not state.config.trace_search:
        return
    if not always and not _trace_filter_matches(
        state,
        portals=portals,
        cells=cells,
    ):
        return

    limit = max(0, state.config.max_trace_events)
    if state.trace_event_count >= limit:
        if not state.trace_limit_notified:
            trace_logger.info("trace limit reached")
            state.trace_limit_notified = True
        return

    state.trace_event_count += 1
    details = " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None
    )
    trace_logger.info(
        "event=%s seq=%s%s",
        event,
        state.trace_event_count,
        f" {details}" if details else "",
    )


def log_search_summary(state: AnytimeSearchState, *, phase: str) -> None:
    if not state.config.trace_search:
        return
    _emit_trace_event(
        state,
        "search_summary",
        always=True,
        phase=phase,
        source=state.query.source,
        target=state.query.target,
        active_portals=len(state.active_portals),
        overlay_edges=_overlay_edge_count(state),
        archive_size=len(state.archive.entries),
        fwd_queue=len(state.frontier["fwd"]),
        bwd_queue=len(state.frontier["bwd"]),
    )


def log_frontier_state(
    state: AnytimeSearchState,
    *,
    action: str,
    direction: Direction | None = None,
) -> None:
    if not state.config.trace_search:
        return
    _emit_trace_event(
        state,
        "frontier",
        always=True,
        action=action,
        direction=direction,
        turn=state.turn,
        fwd_queue=len(state.frontier["fwd"]),
        bwd_queue=len(state.frontier["bwd"]),
    )


def log_label_pop(state: AnytimeSearchState, label: PortalLabel) -> None:
    cell = state.trace_cell(label.portal)
    _increment_count(state.touched_cells, cell)
    if not state.config.trace_search:
        return
    _emit_trace_event(
        state,
        "label_pop",
        portals=(label.portal,),
        cells=(cell,),
        direction=label.direction,
        portal=label.portal,
        cell=cell,
        visited_cells=sorted(label.visited_cells),
        revisited_cells=sorted(label.revisited_cells),
        metrics=label.metrics.compact_summary(),
        score=f"{label.priority:.4f}",
        in_degree=len(state.overlay.in_edges.get(label.portal, [])),
        out_degree=len(state.overlay.out_edges.get(label.portal, [])),
        road_changes=label.road_changes,
    )


def log_local_discovery(
    state: AnytimeSearchState,
    label: PortalLabel,
    *,
    action: str,
    discovered_edges: int | None = None,
) -> None:
    if not state.config.trace_search:
        return
    cell = state.trace_cell(label.portal)
    _emit_trace_event(
        state,
        "local_discovery",
        portals=(label.portal,),
        cells=(cell,),
        action=action,
        direction=label.direction,
        portal=label.portal,
        cell=cell,
        directional_degree=_directional_overlay_degree(state, label),
        discovered_edges=discovered_edges,
    )


def log_child_generation(
    state: AnytimeSearchState,
    parent_label: PortalLabel,
    *,
    child_label: PortalLabel | None = None,
    edge: OverlayEdge | None = None,
    status: str,
    reason: str | None = None,
) -> None:
    if not state.config.trace_search:
        return
    child_portal = child_label.portal if child_label is not None else None
    portals = [parent_label.portal]
    if child_portal is not None:
        portals.append(child_portal)
    cells = [state.trace_cell(portal) for portal in portals]
    _emit_trace_event(
        state,
        "child",
        portals=portals,
        cells=cells,
        status=status,
        reason=reason,
        direction=parent_label.direction,
        parent=parent_label.portal,
        child=child_portal,
        child_cell=(
            state.trace_cell(child_portal) if child_portal is not None else None
        ),
        edge=(
            edge.compact_summary(
                src_cell=state.trace_cell(edge.src),
                dst_cell=state.trace_cell(edge.dst),
            )
            if edge is not None
            else None
        ),
        metrics=(
            child_label.metrics.compact_summary()
            if child_label is not None
            else None
        ),
        visited_cells=(
            sorted(child_label.visited_cells)
            if child_label is not None
            else None
        ),
        revisited_cells=(
            sorted(child_label.revisited_cells)
            if child_label is not None
            else None
        ),
    )


def log_join_attempt(
    state: AnytimeSearchState,
    *,
    kind: str,
    trigger_dir: Direction,
    portals: Iterable[int],
    attempted: bool = True,
    succeeded: bool = False,
    reason: str | None = None,
) -> None:
    if not state.config.trace_search:
        return
    portal_list = list(portals)
    _emit_trace_event(
        state,
        "join",
        portals=portal_list,
        cells=[state.trace_cell(portal) for portal in portal_list],
        kind=kind,
        trigger_dir=trigger_dir,
        attempted=attempted,
        succeeded=succeeded,
        reason=reason,
        portals_involved=portal_list,
    )


def log_representative_decision(
    state: AnytimeSearchState,
    label: PortalLabel,
    *,
    accepted: bool,
    reason: str,
    retained: int | None = None,
) -> None:
    if not state.config.trace_search:
        return
    cell = state.trace_cell(label.portal)
    _emit_trace_event(
        state,
        "representative",
        portals=(label.portal,),
        cells=(cell,),
        direction=label.direction,
        portal=label.portal,
        cell=cell,
        accepted=accepted,
        reason=reason,
        retained=retained,
        metrics=label.metrics.compact_summary(),
        score=f"{label.priority:.4f}",
        visited_cells=len(label.visited_cells),
        road_changes=label.road_changes,
    )


def log_archive_update(
    state: AnytimeSearchState,
    entry: ArchiveEntry,
    *,
    join_kind: str,
    added: bool,
) -> None:
    if not state.config.trace_search:
        return
    cell = state.trace_cell(entry.join_portal)
    _emit_trace_event(
        state,
        "archive",
        portals=(entry.join_portal,),
        cells=(cell,),
        join_kind=join_kind,
        added=added,
        archive_size=len(state.archive.entries),
        entry=entry.compact_summary(cell=cell),
    )


def _csr_edge_index(G: CompactDiGraph, u: int, local_edge_idx: int) -> int | None:
    offsets = getattr(G, "offsets", None)
    if offsets is None:
        return None
    return int(offsets[int(u)]) + int(local_edge_idx)


def _road_id_from_csr_edge(G: CompactDiGraph, edge_idx: int | None) -> int | None:
    if edge_idx is None:
        return None
    road_ids = getattr(G, "road_id", None)
    if road_ids is None:
        return None
    if edge_idx < 0 or edge_idx >= len(road_ids):
        return None
    return int(road_ids[edge_idx])


def _road_id_from_neighbor(
    G: CompactDiGraph,
    u: int,
    local_edge_idx: int,
) -> int | None:
    return _road_id_from_csr_edge(G, _csr_edge_index(G, u, local_edge_idx))


def _road_change_delta(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 0
    return int(left != right)


@dataclass(frozen=True)
class RoadContinuityPreference:
    continues_road: bool
    incremental_road_changes: int


def _valid_road_id(road_id: int | None) -> int | None:
    if road_id is None or road_id < 0:
        return None
    return road_id


def _extension_road_continuity_preference(
    label: PortalLabel,
    first_road_id: int | None,
    last_road_id: int | None,
    road_changes: int,
) -> RoadContinuityPreference:
    parent_road_id = _valid_road_id(label.last_road_id)
    first = _valid_road_id(first_road_id)
    last = _valid_road_id(last_road_id)

    if label.direction == "fwd":
        boundary_delta = _road_change_delta(parent_road_id, first)
        continues_road = parent_road_id is not None and parent_road_id == first
    else:
        boundary_delta = _road_change_delta(last, parent_road_id)
        continues_road = parent_road_id is not None and last == parent_road_id

    return RoadContinuityPreference(
        continues_road=continues_road,
        incremental_road_changes=road_changes + boundary_delta,
    )


def _extension_road_continuity_sort_fields(
    label: PortalLabel,
    first_road_id: int | None,
    last_road_id: int | None,
    road_changes: int,
) -> Tuple[bool, int]:
    preference = _extension_road_continuity_preference(
        label,
        first_road_id,
        last_road_id,
        road_changes,
    )
    return (not preference.continues_road, preference.incremental_road_changes)


def _local_shortcut_candidate_order_key(
    label: PortalLabel,
    edge: OverlayEdge,
) -> Tuple[float, float, bool, int, int, int, int]:
    return (
        edge.metrics.length,
        edge.metrics.elevation,
        *_extension_road_continuity_sort_fields(
            label,
            edge.first_road_id,
            edge.last_road_id,
            edge.road_changes,
        ),
        edge.road_changes,
        edge.src,
        edge.dst,
    )


def _append_road_id(
    first_road_id: int | None,
    last_road_id: int | None,
    road_changes: int,
    next_road_id: int | None,
) -> Tuple[int | None, int | None, int]:
    next_first = first_road_id if first_road_id is not None else next_road_id
    next_changes = road_changes + _road_change_delta(last_road_id, next_road_id)
    next_last = next_road_id if next_road_id is not None else last_road_id
    return next_first, next_last, next_changes


def _prepend_road_id(
    first_road_id: int | None,
    last_road_id: int | None,
    road_changes: int,
    previous_road_id: int | None,
) -> Tuple[int | None, int | None, int]:
    next_first = previous_road_id if previous_road_id is not None else first_road_id
    next_changes = road_changes + _road_change_delta(previous_road_id, first_road_id)
    next_last = last_road_id if last_road_id is not None else previous_road_id
    return next_first, next_last, next_changes


def _extend_label_road_continuity(
    label: PortalLabel,
    edge: OverlayEdge,
) -> Tuple[int | None, int]:
    if label.direction == "fwd":
        road_changes = (
            label.road_changes
            + edge.road_changes
            + _road_change_delta(label.last_road_id, edge.first_road_id)
        )
        last_road_id = (
            edge.last_road_id
            if edge.last_road_id is not None
            else label.last_road_id
        )
        return last_road_id, road_changes

    road_changes = (
        edge.road_changes
        + label.road_changes
        + _road_change_delta(edge.last_road_id, label.last_road_id)
    )
    last_road_id = (
        edge.first_road_id
        if edge.first_road_id is not None
        else label.last_road_id
    )
    return last_road_id, road_changes


def _same_portal_join_road_changes(
    fwd_label: PortalLabel,
    bwd_label: PortalLabel,
) -> int:
    return (
        fwd_label.road_changes
        + bwd_label.road_changes
        + _road_change_delta(fwd_label.last_road_id, bwd_label.last_road_id)
    )


def _one_edge_join_road_changes(
    fwd_label: PortalLabel,
    edge: OverlayEdge,
    bwd_label: PortalLabel,
) -> int:
    return (
        fwd_label.road_changes
        + edge.road_changes
        + bwd_label.road_changes
        + _road_change_delta(fwd_label.last_road_id, edge.first_road_id)
        + _road_change_delta(edge.last_road_id, bwd_label.last_road_id)
    )


def _is_combined_route_feasible(
    state: AnytimeSearchState,
    metrics: RouteAccumulator,
) -> bool:
    state.audit.feasibility_checked_on_combined_accumulator += 1

    length = metrics.length
    elevation = metrics.elevation
    popularity_total = metrics.popularity_length
    width_total = metrics.street_width_length
    if length <= 0.0:
        avg_popularity = 0.0
        avg_width = 0.0
    else:
        avg_popularity = popularity_total / length
        avg_width = width_total / length

    lower = state.query.constraints.lower
    upper = state.query.constraints.upper
    rejected: List[str] = []
    if length < float(lower[0]) or length > float(upper[0]):
        state.audit.rejected_length += 1
        rejected.append("length")
    if elevation < float(lower[1]) or elevation > float(upper[1]):
        state.audit.rejected_elevation += 1
        rejected.append("elevation")
    if avg_popularity < float(lower[2]) or avg_popularity > float(upper[2]):
        state.audit.rejected_avg_popularity += 1
        rejected.append("avg_popularity")
    if avg_width < float(lower[3]) or avg_width > float(upper[3]):
        state.audit.rejected_avg_width += 1
        rejected.append("avg_width")

    feasible = not rejected
    logger.debug(
        "feasibility_checked_on_combined_accumulator=%s feasible=%s rejected=%s "
        "L_total=%.1f H_total=%.1f P_total=%.1f W_total=%.1f "
        "avg_popularity=%.2f avg_width=%.2f counters=%s",
        state.audit.feasibility_checked_on_combined_accumulator,
        feasible,
        rejected,
        length,
        elevation,
        popularity_total,
        width_total,
        avg_popularity,
        avg_width,
        state.audit.as_dict(),
    )
    return feasible


_REPRESENTATIVE_KEYS: Tuple[str, ...] = (
    "centered_score",
    "shortest_length",
    "lowest_elevation",
    "best_popularity",
    "best_width",
    "fewest_road_changes",
)


def _positive_length_average(total: float, length: float) -> float | None:
    if length <= 0.0:
        return None
    return total / length


def _representative_key_value(label: PortalLabel, key: str) -> float:
    metrics = label.metrics
    if key == "centered_score":
        return label.priority
    if key == "shortest_length":
        return metrics.length
    if key == "lowest_elevation":
        return metrics.elevation
    if key == "best_popularity":
        avg = _positive_length_average(metrics.popularity_length, metrics.length)
        return float("inf") if avg is None else -avg
    if key == "best_width":
        avg = _positive_length_average(metrics.street_width_length, metrics.length)
        return float("inf") if avg is None else -avg
    if key == "fewest_road_changes":
        return float(label.road_changes)
    raise ValueError(f"unknown representative key: {key}")


def _representative_sort_key(
    label: PortalLabel,
) -> Tuple[float, float, float, float, float, int]:
    metrics = label.metrics
    avg_popularity = _positive_length_average(metrics.popularity_length, metrics.length)
    avg_width = _positive_length_average(metrics.street_width_length, metrics.length)
    return (
        label.priority,
        metrics.length,
        metrics.elevation,
        -(avg_popularity if avg_popularity is not None else 0.0),
        -(avg_width if avg_width is not None else 0.0),
        label.road_changes,
    )


def _representative_winner_reasons(
    labels: Sequence[PortalLabel],
) -> Dict[int, List[str]]:
    reasons_by_idx: Dict[int, List[str]] = {idx: [] for idx in range(len(labels))}
    for key in _REPRESENTATIVE_KEYS:
        candidates = [
            (idx, _representative_key_value(label, key))
            for idx, label in enumerate(labels)
        ]
        candidates = [
            (idx, value) for idx, value in candidates if value != float("inf")
        ]
        if not candidates:
            continue
        best_idx, _ = min(
            candidates,
            key=lambda item: (
                item[1],
                _representative_sort_key(labels[item[0]]),
                item[0],
            ),
        )
        reasons_by_idx[best_idx].append(key)
    return reasons_by_idx


def _select_representatives(
    labels: Sequence[PortalLabel],
    limit: int,
) -> Tuple[List[PortalLabel], Dict[int, List[str]], Dict[int, List[str]]]:
    reasons_by_idx = _representative_winner_reasons(labels)
    if limit <= 0:
        return [], {}, {id(label): reasons for label, reasons in zip(labels, reasons_by_idx.values())}

    key_order = {key: idx for idx, key in enumerate(_REPRESENTATIVE_KEYS)}
    winner_indices = [idx for idx, reasons in reasons_by_idx.items() if reasons]
    winner_indices.sort(
        key=lambda idx: (
            -len(reasons_by_idx[idx]),
            min(key_order[key] for key in reasons_by_idx[idx]),
            _representative_sort_key(labels[idx]),
            idx,
        )
    )

    selected_indices: List[int] = []
    for idx in winner_indices:
        if len(selected_indices) >= limit:
            break
        selected_indices.append(idx)

    if len(selected_indices) < limit:
        selected_set = set(selected_indices)
        filler_indices = [
            idx for idx in range(len(labels)) if idx not in selected_set
        ]
        filler_indices.sort(
            key=lambda idx: (_representative_sort_key(labels[idx]), idx)
        )
        for idx in filler_indices:
            if len(selected_indices) >= limit:
                break
            selected_indices.append(idx)
            reasons_by_idx[idx].append("centered_score_fill")

    selected_indices.sort(
        key=lambda idx: (_representative_sort_key(labels[idx]), idx)
    )
    selected = [labels[idx] for idx in selected_indices]
    selected_reasons = {
        id(labels[idx]): reasons_by_idx[idx] for idx in selected_indices
    }
    all_reasons = {
        id(label): reasons_by_idx[idx] for idx, label in enumerate(labels)
    }
    return selected, selected_reasons, all_reasons


def _representative_reason_summary(
    labels: Sequence[PortalLabel],
    reasons_by_id: Dict[int, List[str]],
) -> List[str]:
    return [
        f"{'+'.join(reasons_by_id.get(id(label), [])) or 'none'}:{_fmt_route_vector(label.metrics)}"
        for label in labels
    ]


def _bounded_representative_labels(
    state: AnytimeSearchState,
    labels: Sequence[PortalLabel],
) -> List[PortalLabel]:
    return list(labels[: state.config.max_labels_per_portal])


def _overlay_edge_count(state: AnytimeSearchState) -> int:
    return sum(len(edges) for edges in state.overlay.out_edges.values())


def _merge_segments(segments: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    if not segments:
        return ()
    path = list(segments[0])
    for segment in segments[1:]:
        if not segment:
            continue
        path.extend(segment[1:])
    return tuple(path)


def reconstruct_forward_label_path(label: PortalLabel) -> Tuple[int, ...]:
    if label.parent is None or label.parent_edge is None:
        return (label.portal,)

    segments: List[Tuple[int, ...]] = []
    cur = label
    while cur.parent is not None and cur.parent_edge is not None:
        segments.append(cur.parent_edge.path_nodes)
        cur = cur.parent
    segments.reverse()
    return _merge_segments(segments)


def reconstruct_backward_label_path(label: PortalLabel) -> Tuple[int, ...]:
    if label.parent is None or label.parent_edge is None:
        return (label.portal,)

    segments: List[Tuple[int, ...]] = []
    cur = label
    while cur.parent is not None and cur.parent_edge is not None:
        segments.append(cur.parent_edge.path_nodes)
        cur = cur.parent
    return _merge_segments(segments)


def _increment_count(counter: Dict[CountKey, int], key: CountKey) -> None:
    counter[key] = counter.get(key, 0) + 1


def _increment_nested_count(
    counters: Dict[str, Dict[int, int]],
    direction: Direction,
    key: int,
) -> None:
    histogram = counters.setdefault(direction, {})
    _increment_count(histogram, key)


def _increment_direction_count(counter: Dict[str, int], direction: Direction) -> None:
    counter[direction] = counter.get(direction, 0) + 1


def _effective_max_cell_visits(state: AnytimeSearchState) -> int:
    # The label representation tracks first and second contiguous visits.
    return 1 if state.config.max_cell_visits_per_route <= 1 else 2


def _compressed_cell_sequence_from_nodes(
    state: AnytimeSearchState,
    path_nodes: Sequence[int],
) -> Tuple[int, ...]:
    cells: List[int] = []
    previous_cell: int | None = None
    for node in path_nodes:
        cell = state.trace_cell(int(node))
        if cell != previous_cell:
            cells.append(cell)
            previous_cell = cell
    return tuple(cells)


def _cell_visit_counts_from_sequence(
    cell_sequence: Sequence[int],
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for cell in cell_sequence:
        counts[int(cell)] = counts.get(int(cell), 0) + 1
    return counts


def _cell_visit_count(label: PortalLabel, cell: int) -> int:
    if cell in label.revisited_cells:
        return 2
    if cell in label.visited_cells:
        return 1
    return 0


def _apply_cell_visits_to_history(
    state: AnytimeSearchState,
    visited_cells: frozenset[int],
    revisited_cells: frozenset[int],
    cell_visits_to_add: Sequence[int],
) -> CellHistoryUpdate | None:
    max_visits = _effective_max_cell_visits(state)
    visited = set(visited_cells)
    revisited = set(revisited_cells)
    second_visits_added = 0

    for cell in cell_visits_to_add:
        cell_id = int(cell)
        current_count = 2 if cell_id in revisited else 1 if cell_id in visited else 0
        if current_count >= max_visits:
            return None
        if current_count == 0:
            visited.add(cell_id)
            continue
        revisited.add(cell_id)
        second_visits_added += 1

    return CellHistoryUpdate(
        visited_cells=frozenset(visited),
        revisited_cells=frozenset(revisited),
        second_visits_added=second_visits_added,
    )


def _extend_label_cell_history(
    state: AnytimeSearchState,
    label: PortalLabel,
    edge: OverlayEdge,
) -> CellHistoryUpdate | None:
    edge_cells = _compressed_cell_sequence_from_nodes(state, edge.path_nodes)
    current_cell = state.trace_cell(label.portal)
    if not edge_cells:
        additions: Tuple[int, ...] = ()
    elif label.direction == "fwd":
        additions = edge_cells[1:] if edge_cells[0] == current_cell else edge_cells
    else:
        additions = edge_cells[:-1] if edge_cells[-1] == current_cell else edge_cells
    return _apply_cell_visits_to_history(
        state,
        label.visited_cells,
        label.revisited_cells,
        additions,
    )


def _edge_extension_would_exceed_cell_visit_limit(
    state: AnytimeSearchState,
    label: PortalLabel,
    edge: OverlayEdge,
) -> bool:
    return _extend_label_cell_history(state, label, edge) is None


def _record_second_cell_visits_allowed(
    state: AnytimeSearchState,
    context: Direction | Literal["join"],
    count: int,
) -> None:
    if count <= 0:
        return
    state.audit.second_cell_visits_allowed += count
    state.audit.second_cell_visits_allowed_by_direction[context] = (
        state.audit.second_cell_visits_allowed_by_direction.get(context, 0) + count
    )


def _record_rejected_third_cell_visit(
    state: AnytimeSearchState,
    context: Direction | Literal["join"],
) -> None:
    state.audit.rejected_third_cell_visit += 1
    state.audit.rejected_third_cell_visit_by_direction[context] = (
        state.audit.rejected_third_cell_visit_by_direction.get(context, 0) + 1
    )


def _path_cell_visit_limit_result(
    state: AnytimeSearchState,
    path_nodes: Sequence[int],
) -> Tuple[bool, Tuple[int, ...], Dict[int, int]]:
    cell_sequence = _compressed_cell_sequence_from_nodes(state, path_nodes)
    counts = _cell_visit_counts_from_sequence(cell_sequence)
    max_visits = _effective_max_cell_visits(state)
    return (
        all(count <= max_visits for count in counts.values()),
        cell_sequence,
        counts,
    )


def _debug_cell_visit_sequence_result(
    cell_sequence: Sequence[int],
    max_cell_visits_per_route: int,
) -> Dict[str, object]:
    max_visits = 1 if max_cell_visits_per_route <= 1 else 2
    counts = _cell_visit_counts_from_sequence(cell_sequence)
    return {
        "accepted": all(count <= max_visits for count in counts.values()),
        "max_cell_visits_per_route": max_visits,
        "visit_counts": dict(sorted(counts.items())),
        "twice_visited_cells": sorted(
            cell for cell, count in counts.items() if count == 2
        ),
        "over_limit_cells": sorted(
            cell for cell, count in counts.items() if count > max_visits
        ),
    }


def _debug_extend_cell_sequence_result(
    parent_sequence: Sequence[int],
    edge_sequence: Sequence[int],
    direction: Direction,
    max_cell_visits_per_route: int,
) -> Dict[str, object]:
    if direction == "fwd":
        additions = tuple(edge_sequence[1:]) if edge_sequence else ()
        combined = tuple(parent_sequence) + additions
    else:
        prefix = tuple(edge_sequence[:-1]) if edge_sequence else ()
        combined = prefix + tuple(parent_sequence)
    result = _debug_cell_visit_sequence_result(
        combined,
        max_cell_visits_per_route,
    )
    result["parent_sequence"] = tuple(parent_sequence)
    result["edge_sequence"] = tuple(edge_sequence)
    result["combined_sequence"] = combined
    result["direction"] = direction
    return result


def _debug_join_cell_sequence_result(
    fwd_sequence: Sequence[int],
    connector_sequence: Sequence[int],
    bwd_sequence: Sequence[int],
    max_cell_visits_per_route: int,
) -> Dict[str, object]:
    connector_tail = tuple(connector_sequence[1:]) if connector_sequence else ()
    bwd_tail = tuple(bwd_sequence[1:]) if bwd_sequence else ()
    combined = tuple(fwd_sequence) + connector_tail + bwd_tail
    result = _debug_cell_visit_sequence_result(
        combined,
        max_cell_visits_per_route,
    )
    result["fwd_sequence"] = tuple(fwd_sequence)
    result["connector_sequence"] = tuple(connector_sequence)
    result["bwd_sequence"] = tuple(bwd_sequence)
    result["combined_sequence"] = combined
    return result


def debug_cell_visit_helper_examples() -> Dict[str, object]:
    reference = (
        36,
        93,
        79,
        93,
        79,
        40,
        99,
        97,
        56,
        48,
        30,
        54,
        13,
        49,
        87,
        31,
        37,
        1,
    )
    examples = {
        "reference_max1": _debug_cell_visit_sequence_result(reference, 1),
        "reference_max2": _debug_cell_visit_sequence_result(reference, 2),
        "reference_plus_93_max2": _debug_cell_visit_sequence_result(
            reference + (93,),
            2,
        ),
        "reference_plus_79_max2": _debug_cell_visit_sequence_result(
            reference + (79,),
            2,
        ),
        "forward_a": _debug_extend_cell_sequence_result(
            (36, 93),
            (93, 79),
            "fwd",
            2,
        ),
        "forward_b": _debug_extend_cell_sequence_result(
            (36, 93, 79),
            (79, 93),
            "fwd",
            2,
        ),
        "forward_c": _debug_extend_cell_sequence_result(
            (36, 93, 79, 93),
            (93, 79),
            "fwd",
            2,
        ),
        "forward_d": _debug_extend_cell_sequence_result(
            (36, 93, 79, 93, 79),
            (79, 93),
            "fwd",
            2,
        ),
        "backward_equivalent": _debug_extend_cell_sequence_result(
            (93, 79),
            (36, 93),
            "bwd",
            2,
        ),
        "join_connector_overlap_ok": _debug_join_cell_sequence_result(
            (36, 93, 79),
            (79, 93),
            (93, 40),
            2,
        ),
        "join_third_visit_rejected": _debug_join_cell_sequence_result(
            (36, 93, 79, 93, 40),
            (40, 79),
            (79, 93),
            2,
        ),
    }
    return examples


def _record_backward_rejection(
    state: AnytimeSearchState,
    reason: str,
) -> None:
    state.audit.backward_rejection_reason_count[reason] = (
        state.audit.backward_rejection_reason_count.get(reason, 0) + 1
    )


def _record_trace_child_rejection(
    trace: BackwardPopTrace | None,
    reason: str,
) -> None:
    if trace is None:
        return
    trace.child_rejections[reason] = trace.child_rejections.get(reason, 0) + 1


def _top_str_count_items(counts: Dict[str, int], limit: int = 10) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _is_backward_edge_usable_for_label(
    state: AnytimeSearchState,
    label: PortalLabel,
    edge: OverlayEdge,
) -> bool:
    return not _edge_extension_would_exceed_cell_visit_limit(state, label, edge)


def _usable_backward_edge_count(
    state: AnytimeSearchState,
    label: PortalLabel,
    skip_edge_ids: Set[int] | None = None,
) -> int:
    skip_edge_ids = skip_edge_ids or set()
    return sum(
        1
        for edge in state.overlay.in_edges.get(label.portal, [])
        if id(edge) not in skip_edge_ids
        and _is_backward_edge_usable_for_label(state, label, edge)
    )


def _usable_backward_degree_for_cell_entry(
    state: AnytimeSearchState,
    portal: int,
    cell_id: int,
) -> int:
    return sum(
        1
        for edge in state.overlay.in_edges.get(portal, [])
        if not (edge.kind == "inter" and state.trace_cell(edge.src) == cell_id)
    )


def _make_backward_pop_trace(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> BackwardPopTrace:
    trace = BackwardPopTrace(
        pop_index=len(state.audit.backward_pop_traces) + 1,
        portal=label.portal,
        cell_id=state.trace_cell(label.portal),
        visited_cells=tuple(sorted(label.visited_cells)),
        in_edges_count=len(state.overlay.in_edges.get(label.portal, [])),
        out_edges_count=len(state.overlay.out_edges.get(label.portal, [])),
        usable_backward_edges=_usable_backward_edge_count(state, label),
    )
    state.audit.backward_pop_traces.append(trace)
    return trace


def _log_backward_pop_trace(trace: BackwardPopTrace) -> None:
    logger.debug(
        "Backward pop trace pop_index=%s portal=%s cell=%s visited_cells=%s "
        "in_edges=%s out_edges=%s usable_backward_edges=%s "
        "local_triggered=%s local_cell_id=%s local_batches_resumed=%s "
        "local_shortcuts_discovered=%s local_edges_inserted=%s "
        "children_from_local_shortcuts=%s children_from_existing_overlay=%s "
        "child_rejections=%s",
        trace.pop_index,
        trace.portal,
        trace.cell_id,
        list(trace.visited_cells),
        trace.in_edges_count,
        trace.out_edges_count,
        trace.usable_backward_edges,
        trace.local_triggered,
        trace.local_cell_id,
        trace.local_batches_resumed,
        trace.local_shortcuts_discovered,
        trace.local_edges_inserted,
        trace.children_from_local_shortcuts,
        trace.children_from_existing_overlay,
        dict(sorted(trace.child_rejections.items())),
    )


def _label_cell_path_length(state: AnytimeSearchState, label: PortalLabel) -> int:
    if label.direction == "fwd":
        path_nodes = reconstruct_forward_label_path(label)
    else:
        path_nodes = reconstruct_backward_label_path(label)
    cell_count = 0
    previous_cell: int | None = None
    for node in path_nodes:
        cell_id = state.trace_cell(node)
        if previous_cell is None or cell_id != previous_cell:
            cell_count += 1
            previous_cell = cell_id
    return cell_count


def _record_accepted_representative_label(
    state: AnytimeSearchState,
    label: PortalLabel,
    accept_reasons: Sequence[str] | None = None,
) -> None:
    accept_reasons = list(accept_reasons or [])
    current_cell = state.trace_cell(label.portal)
    visited_cells_size = len(label.visited_cells)
    cell_path_length = _label_cell_path_length(state, label)
    source_cell = state.trace_cell(state.query.source)
    target_cell = state.trace_cell(state.query.target)
    has_source_cell_far = (
        source_cell in label.visited_cells
        and current_cell != source_cell
        and cell_path_length > 1
    )
    has_target_cell_far = (
        target_cell in label.visited_cells
        and current_cell != target_cell
        and cell_path_length > 1
    )

    state.audit.representative_accept_count[label.direction] = (
        state.audit.representative_accept_count.get(label.direction, 0) + 1
    )
    if label.parent is not None:
        _increment_direction_count(
            state.audit.representative_child_accept_count,
            label.direction,
        )
    for reason in accept_reasons:
        counter_reason = (
            "best_road_continuity"
            if reason == "fewest_road_changes"
            else reason
        )
        state.audit.representative_accept_reason_count[counter_reason] = (
            state.audit.representative_accept_reason_count.get(counter_reason, 0) + 1
        )
    _increment_nested_count(
        state.audit.representative_visited_cells_size_histogram,
        label.direction,
        visited_cells_size,
    )
    _increment_nested_count(
        state.audit.representative_cell_path_length_histogram,
        label.direction,
        cell_path_length,
    )
    if has_source_cell_far:
        state.audit.representative_source_cell_far_from_source[label.direction] = (
            state.audit.representative_source_cell_far_from_source.get(label.direction, 0)
            + 1
        )
    if has_target_cell_far:
        state.audit.representative_target_cell_far_from_target[label.direction] = (
            state.audit.representative_target_cell_far_from_target.get(label.direction, 0)
            + 1
        )

    logger.debug(
        "Representative audit accept dir=%s portal=%s current_cell=%s "
        "path_length_in_cells=%s visited_cells_size=%s visited_cells=%s "
        "revisited_cells=%s "
        "has_source_cell_far_from_source=%s has_target_cell_far_from_target=%s "
        "road_changes=%s last_road_id=%s accept_reasons=%s",
        label.direction,
        label.portal,
        current_cell,
        cell_path_length,
        visited_cells_size,
        sorted(label.visited_cells),
        sorted(label.revisited_cells),
        has_source_cell_far,
        has_target_cell_far,
        label.road_changes,
        label.last_road_id,
        accept_reasons,
    )
    if "fewest_road_changes" in accept_reasons:
        logger.debug(
            "Representative accepted because best_road_continuity portal=%s dir=%s "
            "road_changes=%s route=%s",
            label.portal,
            label.direction,
            label.road_changes,
            _fmt_route_vector(label.metrics),
        )


def _top_count_items(counts: Dict[int, int], limit: int = 10) -> List[Tuple[int, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _log_overlay_cell_diagnostics(state: AnytimeSearchState) -> None:
    active_portals_by_cell: Dict[int, int] = {}
    low_degree_by_cell: Dict[int, Dict[str, int]] = {}
    current_local_edges_by_cell: Dict[int, int] = {}

    for portal in sorted(state.active_portals):
        cell_id = state.trace_cell(portal)
        _increment_count(active_portals_by_cell, cell_id)
        out_degree = len(state.overlay.out_edges.get(portal, []))
        in_degree = len(state.overlay.in_edges.get(portal, []))
        low_out = out_degree <= 1
        low_in = in_degree <= 1
        if low_out or low_in:
            cell_counts = low_degree_by_cell.setdefault(
                cell_id,
                {"low_out": 0, "low_in": 0, "zero_out": 0, "zero_in": 0},
            )
            if low_out:
                cell_counts["low_out"] += 1
            if low_in:
                cell_counts["low_in"] += 1
            if out_degree == 0:
                cell_counts["zero_out"] += 1
            if in_degree == 0:
                cell_counts["zero_in"] += 1
        logger.debug(
            "Portal overlay degree portal=%s cell=%s out_degree=%s in_degree=%s "
            "low_out=%s low_in=%s",
            portal,
            cell_id,
            out_degree,
            in_degree,
            low_out,
            low_in,
        )

    for edges in state.overlay.out_edges.values():
        for edge in edges:
            if edge.kind != "local":
                continue
            _increment_count(current_local_edges_by_cell, state.trace_cell(edge.src))

    all_cells = (
        set(active_portals_by_cell)
        | set(state.audit.local_shortcuts_discovered_by_cell)
        | set(current_local_edges_by_cell)
        | set(low_degree_by_cell)
    )
    cell_rows = []
    for cell_id in sorted(all_cells):
        low_counts = low_degree_by_cell.get(cell_id, {})
        cell_rows.append(
            {
                "cell": cell_id,
                "active_portals": active_portals_by_cell.get(cell_id, 0),
                "discovered_local_shortcuts": (
                    state.audit.local_shortcuts_discovered_by_cell.get(cell_id, 0)
                ),
                "current_local_edges": current_local_edges_by_cell.get(cell_id, 0),
                "low_out_degree_portals": low_counts.get("low_out", 0),
                "low_in_degree_portals": low_counts.get("low_in", 0),
                "zero_out_degree_portals": low_counts.get("zero_out", 0),
                "zero_in_degree_portals": low_counts.get("zero_in", 0),
            }
        )

    low_degree_rows = [
        row
        for row in cell_rows
        if row["low_out_degree_portals"] or row["low_in_degree_portals"]
    ]
    logger.debug("Cell active portals/local shortcuts diagnostics=%s", cell_rows)
    logger.debug("Low directional overlay degree cells=%s", low_degree_rows)


def _is_opposite_terminal_cell_transition(
    state: AnytimeSearchState,
    label: PortalLabel,
    next_cell: int,
) -> bool:
    current_cell = state.trace_cell(label.portal)
    source_cell = state.trace_cell(state.query.source)
    target_cell = state.trace_cell(state.query.target)
    return (
        label.direction == "bwd"
        and current_cell != source_cell
        and next_cell == source_cell
    ) or (
        label.direction == "fwd"
        and current_cell != target_cell
        and next_cell == target_cell
    )


def _build_nodes_by_cell(
    partition: Dict[int, int],
    kept_nodes: Iterable[int],
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for node in kept_nodes:
        cell_id = partition.get(node)
        if cell_id is None:
            continue
        out.setdefault(cell_id, set()).add(node)
    return out


def _build_reverse_adjacency(
    G: CompactDiGraph,
    kept_nodes: Set[int],
) -> Dict[int, List[Tuple[int, RouteAccumulator, int | None]]]:
    reverse_adj: Dict[int, List[Tuple[int, RouteAccumulator, int | None]]] = {
        u: [] for u in kept_nodes
    }
    for u in kept_nodes:
        to, weights, _ = G.neighbors(u)
        for idx, v in enumerate(to):
            vv = int(v)
            if vv not in kept_nodes:
                continue
            reverse_adj.setdefault(vv, []).append(
                (
                    u,
                    RouteAccumulator.from_edge_weights(weights[idx]),
                    _road_id_from_neighbor(G, u, idx),
                )
            )
    return reverse_adj


def _build_retained_cell_crossing_index(
    G: CompactDiGraph,
    partition: Dict[int, int],
    kept_nodes: Set[int],
) -> Tuple[
    Dict[int, Set[int]],
    Dict[Tuple[int, int], List[BridgeCrossing]],
    int,
    int,
]:
    neighbors: Dict[int, Set[int]] = {}
    crossings_by_pair: Dict[Tuple[int, int], List[BridgeCrossing]] = {}
    directed_crossings = 0
    undirected_pairs: Set[Tuple[int, int]] = set()

    for u in kept_nodes:
        cell_u = partition.get(u)
        if cell_u is None:
            continue
        to, weights, _ = G.neighbors(u)
        for idx, v_raw in enumerate(to):
            v = int(v_raw)
            if v not in kept_nodes:
                continue
            cell_v = partition.get(v)
            if cell_v is None or cell_v == cell_u:
                continue
            neighbors.setdefault(cell_u, set()).add(cell_v)
            neighbors.setdefault(cell_v, set()).add(cell_u)
            crossings_by_pair.setdefault((cell_u, cell_v), []).append(
                (
                    u,
                    v,
                    RouteAccumulator.from_edge_weights(weights[idx]),
                    _road_id_from_neighbor(G, u, idx),
                )
            )
            directed_crossings += 1
            undirected_pairs.add(
                (cell_u, cell_v) if cell_u <= cell_v else (cell_v, cell_u)
            )

    return neighbors, crossings_by_pair, directed_crossings, len(undirected_pairs)


def _dijkstra_forward_retained_length(
    G: CompactDiGraph,
    kept_nodes: Set[int],
    source: int,
) -> np.ndarray:
    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    if source not in kept_nodes:
        return dist

    dist[source] = 0.0
    frontier: List[Tuple[float, int]] = [(0.0, source)]
    while frontier:
        length, node = heapq.heappop(frontier)
        if length != float(dist[node]):
            continue
        to, weights, _ = G.neighbors(node)
        for idx, nxt_raw in enumerate(to):
            nxt = int(nxt_raw)
            if nxt not in kept_nodes:
                continue
            edge_length = float(weights[idx][0])
            next_length = length + edge_length
            if next_length < float(dist[nxt]):
                dist[nxt] = next_length
                heapq.heappush(frontier, (next_length, nxt))
    return dist


def _dijkstra_reverse_retained_length(
    reverse_adj: Dict[int, List[Tuple[int, RouteAccumulator, int | None]]],
    n_nodes: int,
    kept_nodes: Set[int],
    target: int,
) -> np.ndarray:
    dist = np.full(n_nodes, float("inf"), dtype=np.float64)
    if target not in kept_nodes:
        return dist

    dist[target] = 0.0
    frontier: List[Tuple[float, int]] = [(0.0, target)]
    while frontier:
        length, node = heapq.heappop(frontier)
        if length != float(dist[node]):
            continue
        for pred, edge_metrics, _ in reverse_adj.get(node, []):
            if pred not in kept_nodes:
                continue
            next_length = length + edge_metrics.length
            if next_length < float(dist[pred]):
                dist[pred] = next_length
                heapq.heappush(frontier, (next_length, pred))
    return dist


def _cell93_adjacent_cells_for_portal(
    state: AnytimeSearchState,
    portal: int,
) -> Dict[str, List[int]]:
    out_cells: Set[int] = set()
    in_cells: Set[int] = set()
    portal_cell = state.trace_cell(portal)

    to, _, _ = state.G.neighbors(portal)
    for nxt in to:
        cell_id = state.trace_cell(int(nxt))
        if cell_id != portal_cell:
            out_cells.add(cell_id)

    for pred, _, _ in state.reverse_adj.get(portal, []):
        cell_id = state.trace_cell(pred)
        if cell_id != portal_cell:
            in_cells.add(cell_id)

    return {"out": sorted(out_cells), "in": sorted(in_cells)}


def _cell93_forward_path_within_cell(
    state: AnytimeSearchState,
    source: int,
    target: int,
    allowed_nodes: Set[int],
) -> Tuple[bool, List[int]]:
    if source == target:
        return True, [source]
    queue: List[int] = [source]
    parent: Dict[int, int | None] = {source: None}
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        to, _, _ = state.G.neighbors(node)
        for nxt_raw in to:
            nxt = int(nxt_raw)
            if nxt not in allowed_nodes or nxt in parent:
                continue
            parent[nxt] = node
            if nxt == target:
                path = [target]
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return True, path
            queue.append(nxt)
    return False, []


def _record_cell93_static_diagnostics(state: AnytimeSearchState) -> None:
    cell_id = CELL93_DIAGNOSTIC_CELL
    allowed_nodes = state.nodes_by_cell.get(cell_id, set())
    active_portals = sorted(
        portal
        for portal in state.active_portals
        if state.trace_cell(portal) == cell_id
    )

    internal_edges = 0
    for node in allowed_nodes:
        to, _, _ = state.G.neighbors(node)
        internal_edges += sum(1 for nxt in to if int(nxt) in allowed_nodes)

    portal_rows: Dict[int, Dict[str, object]] = {}
    for portal in active_portals:
        to, _, _ = state.G.neighbors(portal)
        outgoing_internal_degree = sum(1 for nxt in to if int(nxt) in allowed_nodes)
        incoming_internal_degree = sum(
            1
            for pred, _, _ in state.reverse_adj.get(portal, [])
            if pred in allowed_nodes
        )
        portal_rows[portal] = {
            "outgoing_internal_degree": outgoing_internal_degree,
            "incoming_internal_degree": incoming_internal_degree,
            "adjacent_cells": _cell93_adjacent_cells_for_portal(state, portal),
        }

    preds_inside = [
        pred
        for pred, _, _ in state.reverse_adj.get(CELL93_DIAGNOSTIC_PORTAL, [])
        if pred in allowed_nodes
    ]
    preds_crossing = [
        {"pred": pred, "cell": state.trace_cell(pred)}
        for pred, _, _ in state.reverse_adj.get(CELL93_DIAGNOSTIC_PORTAL, [])
        if pred not in allowed_nodes
    ]

    directed_paths: Dict[int, Dict[str, object]] = {}
    for portal in active_portals:
        if portal == CELL93_DIAGNOSTIC_PORTAL:
            continue
        reachable, path = _cell93_forward_path_within_cell(
            state,
            portal,
            CELL93_DIAGNOSTIC_PORTAL,
            allowed_nodes,
        )
        directed_paths[portal] = {
            "reachable": reachable,
            "hops": len(path) - 1 if path else None,
            "path": path,
        }

    state.audit.cell93_diagnostics["static"] = {
        "cell": cell_id,
        "internal_nodes": len(allowed_nodes),
        "internal_directed_edges": internal_edges,
        "active_portals": active_portals,
        "portal_degrees": portal_rows,
        "reverse_adj_for_2515": {
            "predecessors_inside_cell_93": sorted(preds_inside),
            "predecessors_crossing_other_cells": sorted(
                preds_crossing,
                key=lambda row: (row["cell"], row["pred"]),
            ),
        },
        "directed_paths_to_2515_inside_cell_93": directed_paths,
    }
    logger.debug("Cell93 static diagnostics %s", state.audit.cell93_diagnostics["static"])


def _score_crossing_candidate(crossing: Tuple[int, int]) -> Tuple[int, int]:
    # Small-node-id scoring is deliberately simple and easy to replace later.
    return crossing


def _graph_xy_int(G: CompactDiGraph) -> np.ndarray | None:
    xy = getattr(G, "xy_int", None)
    if xy is None:
        xy = getattr(G, "nodes", None)
    if xy is None:
        return None
    xy_arr = np.asarray(xy)
    if xy_arr.ndim != 2 or xy_arr.shape[1] < 2:
        return None
    return xy_arr


def _portal_separation_score(
    candidate: int,
    selected: Set[int],
    xy_int: np.ndarray | None,
) -> Tuple[float, int]:
    if not selected:
        return (float("inf"), -candidate)

    if xy_int is not None and candidate < len(xy_int):
        candidate_xy = xy_int[candidate, :2].astype(np.float64)
        nearest_sq = float("inf")
        has_xy_selected = False
        for node in selected:
            if node >= len(xy_int):
                continue
            delta = candidate_xy - xy_int[node, :2].astype(np.float64)
            nearest_sq = min(nearest_sq, float(np.dot(delta, delta)))
            has_xy_selected = True
        if has_xy_selected:
            return (nearest_sq, -candidate)

    return (float(min(abs(candidate - node) for node in selected)), -candidate)


def _select_active_portals(
    G: CompactDiGraph,
    boundary_nodes: Set[int],
    partition: Dict[int, int],
    kept_nodes: Set[int],
    source: int,
    target: int,
    max_per_cell: int,
) -> Set[int]:
    boundary_by_cell: Dict[int, List[int]] = {}
    for node in sorted(boundary_nodes):
        if node not in kept_nodes:
            continue
        cell_id = partition.get(node)
        if cell_id is None:
            continue
        boundary_by_cell.setdefault(cell_id, []).append(node)

    active: Set[int] = set()
    mandatory_by_cell: Dict[int, Set[int]] = {}
    adjacent_cells_by_cell: Dict[int, Set[int]] = {}
    crossings_by_pair: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    # First pass: discover kept inter-cell edges and normalize each cell pair.
    for u in sorted(kept_nodes):
        cell_u = partition.get(u)
        if cell_u is None:
            continue
        to, _, _ = G.neighbors(u)
        for v_raw in to:
            v = int(v_raw)
            if v not in kept_nodes:
                continue
            cell_v = partition.get(v)
            if cell_v is None or cell_v == cell_u:
                continue
            pair = (cell_u, cell_v) if cell_u < cell_v else (cell_v, cell_u)
            crossing = (u, v) if cell_u == pair[0] else (v, u)
            crossings_by_pair.setdefault(pair, []).append(crossing)
            adjacent_cells_by_cell.setdefault(cell_u, set()).add(cell_v)
            adjacent_cells_by_cell.setdefault(cell_v, set()).add(cell_u)

    # Mandatory portals: keep at least one crossing endpoint for every adjacent cell pair.
    for pair in sorted(crossings_by_pair):
        crossing = min(crossings_by_pair[pair], key=_score_crossing_candidate)
        left, right = crossing
        mandatory_by_cell.setdefault(pair[0], set()).add(left)
        mandatory_by_cell.setdefault(pair[1], set()).add(right)
        active.add(left)
        active.add(right)

    mandatory_count = sum(len(nodes) for nodes in mandatory_by_cell.values())
    xy_int = _graph_xy_int(G)
    separation_added = 0

    # Fill remaining per-cell budget with portals far from already selected portals.
    for cell_id in sorted(boundary_by_cell):
        selected = mandatory_by_cell.setdefault(cell_id, set())
        effective_budget = max(max_per_cell, len(adjacent_cells_by_cell.get(cell_id, set())))
        candidates = [node for node in boundary_by_cell[cell_id] if node not in selected]
        while len(selected) < effective_budget and candidates:
            best = max(
                candidates,
                key=lambda node: _portal_separation_score(node, selected, xy_int),
            )
            candidates.remove(best)
            selected.add(best)
            active.add(best)
            separation_added += 1

    active.add(source)
    active.add(target)
    logging.debug("Portal selection adjacent_cell_pairs=%s", len(crossings_by_pair))
    logging.debug(
        "Portal selection mandatory_portals=%s separation_portals=%s",
        mandatory_count,
        separation_added,
    )
    logging.debug(
        "Portal selection portals_per_cell=%s",
        {cell_id: len(nodes) for cell_id, nodes in sorted(mandatory_by_cell.items())},
    )
    logging.debug("Portal selection final_active_portals=%s", len(active))
    return active


def _build_initial_overlay(
    state: AnytimeSearchState,
) -> None:
    for u in state.active_portals:
        if u not in state.kept_nodes:
            continue
        to, weights, _ = state.G.neighbors(u)
        for idx, v in enumerate(to):
            vv = int(v)
            if vv not in state.active_portals or vv not in state.kept_nodes or vv == u:
                continue
            if state.trace_cell(u) == state.trace_cell(vv):
                continue
            road_id = _road_id_from_neighbor(state.G, u, idx)
            edge = OverlayEdge(
                src=u,
                dst=vv,
                metrics=RouteAccumulator.from_edge_weights(weights[idx]),
                path_nodes=(u, vv),
                first_road_id=road_id,
                last_road_id=road_id,
                road_changes=0,
                kind="inter",
            )
            state.add_overlay_edge(edge)


def _make_root_label(state: AnytimeSearchState, node: int, direction: Direction) -> PortalLabel:
    visited = frozenset({state.trace_cell(node)}) if node in state.partition else frozenset()
    label = PortalLabel(
        portal=node,
        direction=direction,
        metrics=RouteAccumulator(),
        priority=0.0,
        visited_cells=visited,
        revisited_cells=frozenset(),
        last_road_id=None,
        road_changes=0,
        parent=None,
        parent_edge=None,
    )
    logger.debug(
        "Root label init portal=%s dir=%s kept=%s active=%s cell=%s visited_cells=%s",
        node,
        direction,
        node in state.kept_nodes,
        node in state.active_portals,
        state.partition.get(node),
        sorted(visited),
    )
    return label


def _keep_representative(state: AnytimeSearchState, label: PortalLabel) -> bool:
    if label.portal not in state.active_portals:
        log_representative_decision(
            state,
            label,
            accepted=False,
            reason="inactive",
            retained=0,
        )
        logger.debug(
            "Representative reject portal=%s dir=%s reason=inactive priority=%.4f route=%s",
            label.portal,
            label.direction,
            label.priority,
            _fmt_route_vector(label.metrics),
        )
        if label.direction == "bwd":
            _record_backward_rejection(state, "inactive")
            logger.debug(
                "Backward representative reject portal=%s reason=inactive active=%s kept=%s",
                label.portal,
                label.portal in state.active_portals,
                label.portal in state.kept_nodes,
            )
        return False
    if state.is_partially_hopeless(label.metrics, label.direction):
        log_representative_decision(
            state,
            label,
            accepted=False,
            reason="metric_upper_bound",
        )
        logger.debug(
            "Representative reject portal=%s dir=%s reason=hopeless priority=%.4f route=%s",
            label.portal,
            label.direction,
            label.priority,
            _fmt_route_vector(label.metrics),
        )
        if label.direction == "bwd":
            logger.debug(
                "Backward representative reject portal=%s reason=hopeless route=%s",
                label.portal,
                _fmt_route_vector(label.metrics),
            )
        return False

    by_portal = state.labels[label.direction].setdefault(label.portal, [])
    limit = state.config.max_labels_per_portal
    if limit <= 0:
        if label.direction == "bwd":
            state.audit.backward_rejected_representative_capacity += 1
            _record_backward_rejection(state, "representative_capacity")
        logger.debug(
            "Representative reject portal=%s dir=%s reason=no_capacity route=%s retained=0/0",
            label.portal,
            label.direction,
            _fmt_route_vector(label.metrics),
        )
        log_representative_decision(
            state,
            label,
            accepted=False,
            reason="no_capacity",
            retained=0,
        )
        return False

    if len(by_portal) < limit:
        by_portal.append(label)
        selected, selected_reasons, all_reasons = _select_representatives(
            by_portal,
            limit,
        )
        by_portal[:] = selected
        candidate_reasons = all_reasons.get(id(label), []) or ["capacity"]
        logger.debug(
            "Representative accept portal=%s dir=%s reason=capacity retained=%s/%s candidate_reasons=%s route=%s retained_reasons=%s",
            label.portal,
            label.direction,
            len(by_portal),
            limit,
            candidate_reasons,
            _fmt_route_vector(label.metrics),
            _representative_reason_summary(by_portal, selected_reasons),
        )
        _record_accepted_representative_label(state, label, candidate_reasons)
        log_representative_decision(
            state,
            label,
            accepted=True,
            reason="+".join(candidate_reasons),
            retained=len(by_portal),
        )
        return True

    candidate_pool = list(by_portal) + [label]
    selected, selected_reasons, all_reasons = _select_representatives(
        candidate_pool,
        limit,
    )
    selected_ids = {id(selected_label) for selected_label in selected}
    candidate_reasons = all_reasons.get(id(label), [])
    if id(label) not in selected_ids and candidate_reasons and selected:
        dropped = selected[-1]
        selected = [
            selected_label
            for selected_label in selected
            if id(selected_label) != id(dropped)
        ]
        selected.append(label)
        selected.sort(key=_representative_sort_key)
        selected_reasons = {
            id(selected_label): all_reasons.get(id(selected_label), [])
            for selected_label in selected
        }
        selected_ids = {id(selected_label) for selected_label in selected}

    if id(label) not in selected_ids:
        logger.debug(
            "Representative reject portal=%s dir=%s reason=not_selected_by_complementary_keys retained=%s/%s candidate_reasons=%s route=%s retained_reasons=%s",
            label.portal,
            label.direction,
            len(by_portal),
            limit,
            candidate_reasons or ["none"],
            _fmt_route_vector(label.metrics),
            _representative_reason_summary(by_portal, selected_reasons),
        )
        if label.direction == "bwd":
            state.audit.backward_rejected_representative_capacity += 1
            _record_backward_rejection(state, "representative_capacity")
            logger.debug(
                "Backward representative reject portal=%s reason=not_selected_by_complementary_keys candidate_reasons=%s",
                label.portal,
                candidate_reasons or ["none"],
            )
        log_representative_decision(
            state,
            label,
            accepted=False,
            reason="not_selected_by_complementary_keys",
            retained=len(by_portal),
        )
        return False

    replaced = [
        existing for existing in by_portal if id(existing) not in selected_ids
    ]
    by_portal[:] = selected
    accepted_reasons = selected_reasons.get(id(label), candidate_reasons) or ["none"]
    logger.debug(
        "Representative accept portal=%s dir=%s reason=complementary retained=%s/%s candidate_reasons=%s replaced=%s route=%s retained_reasons=%s",
        label.portal,
        label.direction,
        len(by_portal),
        limit,
        accepted_reasons,
        [_fmt_route_vector(existing.metrics) for existing in replaced],
        _fmt_route_vector(label.metrics),
        _representative_reason_summary(by_portal, selected_reasons),
    )
    _record_accepted_representative_label(state, label, accepted_reasons)
    log_representative_decision(
        state,
        label,
        accepted=True,
        reason="+".join(accepted_reasons),
        retained=len(by_portal),
    )
    return True


def _path_contains_segment(
    path_nodes: Tuple[int, ...],
    segment: Tuple[int, ...],
) -> bool:
    if not segment or len(segment) > len(path_nodes):
        return False
    width = len(segment)
    return any(
        path_nodes[start : start + width] == segment
        for start in range(len(path_nodes) - width + 1)
    )


def _archive_route_signature(
    state: AnytimeSearchState,
    path_nodes: Tuple[int, ...],
) -> Tuple[
    frozenset[int],
    Tuple[int, ...],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
]:
    cell_sequence: List[int] = []
    for node in path_nodes:
        cell = state.trace_cell(node)
        if not cell_sequence or cell_sequence[-1] != cell:
            cell_sequence.append(cell)

    bridge_pairs: Set[Tuple[int, int]] = set()
    bridge_corridors: Set[Tuple[int, int]] = set()
    for edges in state.bridge_representatives_by_cell_pair.values():
        for edge in edges:
            if not _path_contains_segment(path_nodes, edge.path_nodes):
                continue
            if edge.bridge_cell_pair is not None:
                bridge_pairs.add(edge.bridge_cell_pair)
            if edge.bridge_corridor is not None:
                bridge_corridors.add(edge.bridge_corridor)
    return (
        frozenset(path_nodes),
        tuple(cell_sequence),
        tuple(sorted(bridge_pairs)),
        tuple(sorted(bridge_corridors)),
    )


def _archive_join_candidate(
    state: AnytimeSearchState,
    *,
    join_portal: int,
    metrics: RouteAccumulator,
    path_nodes: Tuple[int, ...],
    join_kind: str,
    road_changes: int,
) -> bool:
    (
        node_set,
        cell_sequence,
        bridge_cell_pairs,
        bridge_corridors,
    ) = _archive_route_signature(state, path_nodes)
    entry = ArchiveEntry(
        join_portal=join_portal,
        metrics=metrics,
        score=state.query.constraints.score(metrics),
        path_nodes=path_nodes,
        road_changes=road_changes,
        node_set=node_set,
        cell_sequence=cell_sequence,
        bridge_cell_pairs=bridge_cell_pairs,
        bridge_corridors=bridge_corridors,
    )
    added = state.archive.add(entry, state.audit)
    log_archive_update(
        state,
        entry,
        join_kind=join_kind,
        added=added,
    )
    if added:
        logger.debug(
            "Archive add kind=%s join_portal=%s score=%.4f route=%s "
            "path_len=%s road_changes=%s archive_size=%s",
            join_kind,
            entry.join_portal,
            entry.score,
            _fmt_route_vector(entry.metrics),
            len(entry.path_nodes),
            entry.road_changes,
            len(state.archive.entries),
        )
    return added


def _record_one_edge_join_shared_cell_diagnostics(
    state: AnytimeSearchState,
    fwd_label: PortalLabel,
    edge: OverlayEdge,
    bwd_label: PortalLabel,
) -> None:
    p_cell = state.trace_cell(edge.src)
    q_cell = state.trace_cell(edge.dst)
    edge_cells = {
        state.trace_cell(node)
        for node in edge.path_nodes
    }
    expected_connector_cells = set(edge_cells)
    shared_cells = set(fwd_label.visited_cells) & set(bwd_label.visited_cells)
    shared_count = len(shared_cells)
    state.audit.one_edge_join_shared_cells_count_histogram[shared_count] = (
        state.audit.one_edge_join_shared_cells_count_histogram.get(shared_count, 0) + 1
    )

    if shared_cells == {p_cell}:
        state.audit.one_edge_join_shared_only_p_cell += 1
    if shared_cells == {q_cell}:
        state.audit.one_edge_join_shared_only_q_cell += 1
    if (
        shared_cells
        and shared_cells <= expected_connector_cells
        and shared_cells not in ({p_cell}, {q_cell})
    ):
        state.audit.one_edge_join_shared_only_edge_cells += 1
    if shared_count > 1:
        state.audit.one_edge_join_shared_multiple_cells += 1

    unexpected_shared_cells = shared_cells - expected_connector_cells
    for cell_id in unexpected_shared_cells:
        _increment_count(state.audit.one_edge_join_unexpected_shared_cells, cell_id)
    logger.debug(
        "One-edge join shared_cells diagnostics src=%s dst=%s p_cell=%s q_cell=%s "
        "edge_cells=%s shared_cells=%s unexpected_shared_cells=%s "
        "source_cell=%s target_cell=%s",
        edge.src,
        edge.dst,
        p_cell,
        q_cell,
        sorted(edge_cells),
        sorted(shared_cells),
        sorted(unexpected_shared_cells),
        state.trace_cell(state.query.source),
        state.trace_cell(state.query.target),
    )


def _complete_path_has_cell_visit_conflict(
    state: AnytimeSearchState,
    path_nodes: Sequence[int],
    *,
    context: Literal["join"],
) -> bool:
    ok, cell_sequence, counts = _path_cell_visit_limit_result(state, path_nodes)
    max_visits = _effective_max_cell_visits(state)
    if ok:
        second_visits = sum(1 for count in counts.values() if count == 2)
        _record_second_cell_visits_allowed(state, context, second_visits)
        logger.debug(
            "Complete path cell-visit check context=%s accepted=True "
            "max_visits=%s cell_sequence=%s counts=%s",
            context,
            max_visits,
            cell_sequence,
            dict(sorted(counts.items())),
        )
        return False

    _record_rejected_third_cell_visit(state, context)
    logger.debug(
        "Complete path cell-visit check context=%s accepted=False "
        "max_visits=%s cell_sequence=%s counts=%s over_limit=%s",
        context,
        max_visits,
        cell_sequence,
        dict(sorted(counts.items())),
        sorted(cell for cell, count in counts.items() if count > max_visits),
    )
    return True


def _reconstruct_one_edge_join_path(
    fwd_label: PortalLabel,
    edge: OverlayEdge,
    bwd_label: PortalLabel,
) -> Tuple[int, ...] | None:
    fwd_path = reconstruct_forward_label_path(fwd_label)
    bwd_path = reconstruct_backward_label_path(bwd_label)
    if not fwd_path or not edge.path_nodes or not bwd_path:
        return None
    if fwd_path[-1] != edge.src:
        return None
    if edge.path_nodes[0] != edge.src or edge.path_nodes[-1] != edge.dst:
        return None
    if bwd_path[0] != edge.dst:
        return None
    return _merge_segments([fwd_path, edge.path_nodes, bwd_path])


def _emit_same_portal_join_candidate(
    state: AnytimeSearchState,
    fwd_label: PortalLabel,
    bwd_label: PortalLabel,
    *,
    trigger_dir: Direction,
) -> bool:
    state.audit.same_portal_join_attempts += 1
    fwd_path = reconstruct_forward_label_path(fwd_label)
    bwd_path = reconstruct_backward_label_path(bwd_label)
    full_path = tuple(list(fwd_path) + list(bwd_path[1:]))
    if _complete_path_has_cell_visit_conflict(
        state,
        full_path,
        context="join",
    ):
        log_join_attempt(
            state,
            kind="same_portal",
            trigger_dir=trigger_dir,
            portals=(fwd_label.portal,),
            succeeded=False,
            reason="third_cell_visit",
        )
        return False

    metrics = fwd_label.metrics.plus(bwd_label.metrics)
    feasible = _is_combined_route_feasible(state, metrics)
    logger.debug(
        "Join candidate kind=same_portal portal=%s trigger_dir=%s feasible=%s route=%s",
        fwd_label.portal,
        trigger_dir,
        feasible,
        _fmt_route_vector(metrics),
    )
    if not feasible:
        log_join_attempt(
            state,
            kind="same_portal",
            trigger_dir=trigger_dir,
            portals=(fwd_label.portal,),
            succeeded=False,
            reason="infeasible",
        )
        return False

    road_changes = _same_portal_join_road_changes(fwd_label, bwd_label)
    added = _archive_join_candidate(
        state,
        join_portal=fwd_label.portal,
        metrics=metrics,
        path_nodes=full_path,
        join_kind="same_portal",
        road_changes=road_changes,
    )
    if added:
        state.audit.same_portal_join_successes += 1
    log_join_attempt(
        state,
        kind="same_portal",
        trigger_dir=trigger_dir,
        portals=(fwd_label.portal,),
        succeeded=added,
        reason=None if added else "archive_not_updated",
    )
    return added


def _emit_one_edge_join_candidate(
    state: AnytimeSearchState,
    fwd_label: PortalLabel,
    edge: OverlayEdge,
    bwd_label: PortalLabel,
    *,
    trigger_dir: Direction,
    bridge_join_context: Literal["normal", "immediate_bridge"] = "normal",
) -> bool:
    state.audit.one_edge_join_attempts += 1
    _record_one_edge_join_shared_cell_diagnostics(state, fwd_label, edge, bwd_label)

    full_path = _reconstruct_one_edge_join_path(fwd_label, edge, bwd_label)
    if full_path is None:
        state.audit.one_edge_join_rejected_reconstruction += 1
        logger.debug(
            "Join candidate kind=one_edge trigger_dir=%s src=%s dst=%s rejected=reconstruction",
            trigger_dir,
            edge.src,
            edge.dst,
        )
        log_join_attempt(
            state,
            kind="one_edge",
            trigger_dir=trigger_dir,
            portals=(edge.src, edge.dst),
            succeeded=False,
            reason="reconstruction",
        )
        return False

    if _complete_path_has_cell_visit_conflict(
        state,
        full_path,
        context="join",
    ):
        state.audit.one_edge_join_rejected_cell_conflict += 1
        logger.debug(
            "Join candidate kind=one_edge trigger_dir=%s src=%s dst=%s rejected=third_cell_visit "
            "fwd_cells=%s bwd_cells=%s",
            trigger_dir,
            edge.src,
            edge.dst,
            sorted(fwd_label.visited_cells),
            sorted(bwd_label.visited_cells),
        )
        log_join_attempt(
            state,
            kind="one_edge",
            trigger_dir=trigger_dir,
            portals=(edge.src, edge.dst),
            succeeded=False,
            reason="third_cell_visit",
        )
        return False

    metrics = fwd_label.metrics.plus(edge.metrics).plus(bwd_label.metrics)
    feasible = _is_combined_route_feasible(state, metrics)
    if not feasible:
        state.audit.one_edge_join_rejected_infeasible += 1
        logger.debug(
            "Join candidate kind=one_edge trigger_dir=%s src=%s dst=%s feasible=False route=%s",
            trigger_dir,
            edge.src,
            edge.dst,
            _fmt_route_vector(metrics),
        )
        log_join_attempt(
            state,
            kind="one_edge",
            trigger_dir=trigger_dir,
            portals=(edge.src, edge.dst),
            succeeded=False,
            reason="infeasible",
        )
        return False

    logger.debug(
        "Join candidate kind=one_edge trigger_dir=%s src=%s dst=%s feasible=True route=%s",
        trigger_dir,
        edge.src,
        edge.dst,
        _fmt_route_vector(metrics),
    )
    road_changes = _one_edge_join_road_changes(fwd_label, edge, bwd_label)
    added = _archive_join_candidate(
        state,
        join_portal=edge.dst,
        metrics=metrics,
        path_nodes=full_path,
        join_kind="one_edge",
        road_changes=road_changes,
    )
    if added:
        state.audit.one_edge_join_successes += 1
        if edge.bridge_cell_pair is not None and bridge_join_context == "normal":
            state.audit.later_join_successes_through_bridge += 1
    log_join_attempt(
        state,
        kind="one_edge",
        trigger_dir=trigger_dir,
        portals=(edge.src, edge.dst),
        succeeded=added,
        reason=None if added else "archive_not_updated",
    )
    return added


def _emit_one_edge_joins_for_forward_label(
    state: AnytimeSearchState,
    fwd_label: PortalLabel,
) -> None:
    for edge in state.overlay.out_edges.get(fwd_label.portal, []):
        bwd_labels = _bounded_representative_labels(
            state,
            state.labels["bwd"].get(edge.dst, []),
        )
        for bwd_label in bwd_labels:
            _emit_one_edge_join_candidate(
                state,
                fwd_label,
                edge,
                bwd_label,
                trigger_dir="fwd",
            )


def _emit_one_edge_joins_for_backward_label(
    state: AnytimeSearchState,
    bwd_label: PortalLabel,
) -> None:
    for edge in state.overlay.in_edges.get(bwd_label.portal, []):
        fwd_labels = _bounded_representative_labels(
            state,
            state.labels["fwd"].get(edge.src, []),
        )
        for fwd_label in fwd_labels:
            _emit_one_edge_join_candidate(
                state,
                fwd_label,
                edge,
                bwd_label,
                trigger_dir="bwd",
            )


def _emit_join(state: AnytimeSearchState, label: PortalLabel) -> None:
    other_dir: Direction = "bwd" if label.direction == "fwd" else "fwd"
    opposite = _bounded_representative_labels(
        state,
        state.labels[other_dir].get(label.portal, []),
    )
    logger.debug(
        "Join attempt portal=%s dir=%s same_portal_opposite_labels=%s",
        label.portal,
        label.direction,
        len(opposite),
    )
    log_join_attempt(
        state,
        kind="scan",
        trigger_dir=label.direction,
        portals=(label.portal,),
        attempted=bool(opposite),
        succeeded=False,
        reason=None if opposite else "no_same_portal_opposite",
    )

    if label.direction == "fwd":
        for bwd_label in opposite:
            _emit_same_portal_join_candidate(
                state,
                label,
                bwd_label,
                trigger_dir="fwd",
            )
        _emit_one_edge_joins_for_forward_label(state, label)
        return

    for fwd_label in opposite:
        _emit_same_portal_join_candidate(
            state,
            fwd_label,
            label,
            trigger_dir="bwd",
        )
    _emit_one_edge_joins_for_backward_label(state, label)


def _synthetic_terminal_root_label(
    state: AnytimeSearchState,
    direction: Direction,
) -> PortalLabel:
    node = state.query.source if direction == "fwd" else state.query.target
    visited = frozenset({state.trace_cell(node)}) if node in state.partition else frozenset()
    return PortalLabel(
        portal=node,
        direction=direction,
        metrics=RouteAccumulator(),
        priority=0.0,
        visited_cells=visited,
        revisited_cells=frozenset(),
        last_road_id=None,
        road_changes=0,
        parent=None,
        parent_edge=None,
    )


def _terminal_side_labels(
    state: AnytimeSearchState,
    direction: Direction,
    portal: int,
) -> List[PortalLabel]:
    labels = list(state.labels[direction].get(portal, []))
    terminal = state.query.source if direction == "fwd" else state.query.target
    if portal == terminal and not labels:
        labels.append(_synthetic_terminal_root_label(state, direction))
    return _bounded_representative_labels(state, labels)


def _try_terminal_completion(
    state: AnytimeSearchState,
    terminal_label: PortalLabel,
) -> bool:
    archived = False
    if terminal_label.direction == "bwd":
        fwd_labels = _terminal_side_labels(state, "fwd", terminal_label.portal)
        for fwd_label in fwd_labels:
            archived = (
                _emit_same_portal_join_candidate(
                    state,
                    fwd_label,
                    terminal_label,
                    trigger_dir="bwd",
                )
                or archived
            )

        source_labels = _terminal_side_labels(state, "fwd", state.query.source)
        for edge in state.overlay.out_edges.get(state.query.source, []):
            if edge.dst != terminal_label.portal:
                continue
            for fwd_label in source_labels:
                archived = (
                    _emit_one_edge_join_candidate(
                        state,
                        fwd_label,
                        edge,
                        terminal_label,
                        trigger_dir="bwd",
                    )
                    or archived
                )
        return archived

    bwd_labels = _terminal_side_labels(state, "bwd", terminal_label.portal)
    for bwd_label in bwd_labels:
        archived = (
            _emit_same_portal_join_candidate(
                state,
                terminal_label,
                bwd_label,
                trigger_dir="fwd",
            )
            or archived
        )

    target_labels = _terminal_side_labels(state, "bwd", state.query.target)
    for edge in state.overlay.out_edges.get(terminal_label.portal, []):
        if edge.dst != state.query.target:
            continue
        for bwd_label in target_labels:
            archived = (
                _emit_one_edge_join_candidate(
                    state,
                    terminal_label,
                    edge,
                    bwd_label,
                    trigger_dir="fwd",
                )
                or archived
            )
    return archived


def _one_edge_neighbor_label_diagnostics(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    other_dir: Direction = "bwd" if label.direction == "fwd" else "fwd"
    if label.direction == "fwd":
        edges = state.overlay.out_edges.get(label.portal, [])
        neighbor_portals = [edge.dst for edge in edges]
    else:
        edges = state.overlay.in_edges.get(label.portal, [])
        neighbor_portals = [edge.src for edge in edges]

    neighbor_rows: List[Tuple[int, int]] = []
    one_edge_opposite_label_total = 0
    for portal in neighbor_portals:
        opposite_count = len(state.labels[other_dir].get(portal, []))
        neighbor_rows.append((portal, opposite_count))
        one_edge_opposite_label_total += opposite_count
    return len(edges), one_edge_opposite_label_total, neighbor_rows


def _record_generated_child_join_diagnostics(
    state: AnytimeSearchState,
    parent_label: PortalLabel,
    child_label: PortalLabel,
    *,
    edge: OverlayEdge,
    context: str,
    terminal_rejection_candidate: bool,
) -> Tuple[int, int, int, List[Tuple[int, int]]]:
    log_child_generation(
        state,
        parent_label,
        child_label=child_label,
        edge=edge,
        status="generated",
    )
    other_dir: Direction = "bwd" if child_label.direction == "fwd" else "fwd"
    child_cell = state.trace_cell(child_label.portal)
    opposite_at_child_count = len(state.labels[other_dir].get(child_label.portal, []))
    (
        directional_edge_count,
        one_edge_opposite_label_total,
        neighbor_rows,
    ) = _one_edge_neighbor_label_diagnostics(state, child_label)

    state.audit.generated_child_labels_total += 1
    _increment_direction_count(
        state.audit.generated_child_labels_by_direction,
        child_label.direction,
    )
    if opposite_at_child_count > 0:
        state.audit.generated_child_labels_with_same_portal_opposite += 1
    if one_edge_opposite_label_total > 0:
        state.audit.generated_child_labels_with_one_edge_opposite += 1
    if terminal_rejection_candidate:
        _increment_direction_count(
            state.audit.terminal_leak_labels_generated_by_direction,
            child_label.direction,
        )
    if terminal_rejection_candidate and opposite_at_child_count > 0:
        state.audit.terminal_leak_child_with_same_portal_opposite += 1
        _increment_direction_count(
            state.audit.terminal_leak_same_portal_opposite_by_direction,
            child_label.direction,
        )
    if terminal_rejection_candidate and one_edge_opposite_label_total > 0:
        state.audit.terminal_leak_child_with_one_edge_opposite += 1
        _increment_direction_count(
            state.audit.terminal_leak_one_edge_opposite_by_direction,
            child_label.direction,
        )

    logger.debug(
        "Generated child join diagnostics context=%s dir=%s parent=%s child=%s "
        "child_cell=%s same_portal_opposite_exists=%s "
        "same_portal_opposite_count=%s directional_one_edge_edges=%s "
        "one_edge_opposite_label_total=%s one_edge_candidate_portals=%s",
        context,
        child_label.direction,
        parent_label.portal,
        child_label.portal,
        child_cell,
        opposite_at_child_count > 0,
        opposite_at_child_count,
        directional_edge_count,
        one_edge_opposite_label_total,
        neighbor_rows,
    )

    if terminal_rejection_candidate:
        logger.debug(
            "Terminal leak child join diagnostics context=%s dir=%s parent=%s "
            "child=%s parent_cell=%s child_cell=%s visited_cells=%s "
            "opposite_labels_at_child=%s one_edge_candidate_portals=%s",
            context,
            child_label.direction,
            parent_label.portal,
            child_label.portal,
            state.trace_cell(parent_label.portal),
            child_cell,
            sorted(child_label.visited_cells),
            opposite_at_child_count,
            neighbor_rows,
        )

    return (
        opposite_at_child_count,
        directional_edge_count,
        one_edge_opposite_label_total,
        neighbor_rows,
    )


def _attempt_child_joins_before_terminal_gate(
    state: AnytimeSearchState,
    parent_label: PortalLabel,
    child_label: PortalLabel,
    *,
    edge: OverlayEdge,
    context: str,
) -> bool:
    child_cell = state.trace_cell(child_label.portal)
    terminal_portal = (
        state.query.source if child_label.direction == "bwd" else state.query.target
    )
    terminal_rejection_candidate = (
        _is_opposite_terminal_cell_transition(state, parent_label, child_cell)
        and child_label.portal != terminal_portal
    )
    _record_generated_child_join_diagnostics(
        state,
        parent_label,
        child_label,
        edge=edge,
        context=context,
        terminal_rejection_candidate=terminal_rejection_candidate,
    )
    if terminal_rejection_candidate:
        state.audit.join_attempts_before_terminal_rejection += 1

    before_successes = (
        state.audit.same_portal_join_successes
        + state.audit.one_edge_join_successes
    )
    _emit_join(state, child_label)
    success_delta = (
        state.audit.same_portal_join_successes
        + state.audit.one_edge_join_successes
        - before_successes
    )
    if terminal_rejection_candidate and success_delta > 0:
        state.audit.join_successes_before_terminal_rejection += success_delta

    logger.debug(
        "Child pre-terminal join context=%s parent=%s child=%s dir=%s "
        "edge_src=%s edge_dst=%s terminal_rejection_candidate=%s success_delta=%s",
        context,
        parent_label.portal,
        child_label.portal,
        child_label.direction,
        edge.src,
        edge.dst,
        terminal_rejection_candidate,
        success_delta,
    )
    return success_delta > 0


def _backward_reachable_nodes_in_cell(
    state: AnytimeSearchState,
    portal: int,
) -> Set[int]:
    portal_cell = state.trace_cell(portal)
    reachable: Set[int] = {portal}
    stack = [portal]
    while stack:
        node = stack.pop()
        for pred, _, _ in state.reverse_adj.get(node, []):
            if state.trace_cell(pred) != portal_cell or pred in reachable:
                continue
            reachable.add(pred)
            stack.append(pred)
    return reachable


def _backward_crossing_exit_rows(
    state: AnytimeSearchState,
    portal: int,
) -> List[BackwardCrossingExitRow]:
    portal_cell = state.trace_cell(portal)
    rows: List[BackwardCrossingExitRow] = []
    seen: Set[Tuple[str, int, int, int]] = set()
    reachable_nodes = _backward_reachable_nodes_in_cell(state, portal)

    for node in sorted(reachable_nodes):
        for edge in state.overlay.in_edges.get(node, []):
            exit_portal = edge.src
            exit_cell = state.trace_cell(exit_portal)
            if exit_cell == portal_cell:
                continue
            key = ("overlay", node, exit_portal, exit_cell)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": "overlay",
                    "from_node": node,
                    "portal": exit_portal,
                    "cell": exit_cell,
                    "kind": edge.kind,
                }
            )

        for pred, _, _ in state.reverse_adj.get(node, []):
            pred_cell = state.trace_cell(pred)
            if pred_cell == portal_cell:
                continue
            key = ("reverse_adj", node, pred, pred_cell)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": "reverse_adj",
                    "from_node": node,
                    "portal": pred,
                    "cell": pred_cell,
                    "kind": "base",
                }
            )

    return rows


def _record_backward_child_entering_new_cell(
    state: AnytimeSearchState,
    parent_label: PortalLabel,
    child_label: PortalLabel,
    *,
    edge: OverlayEdge,
    context: str,
) -> bool:
    parent_cell = state.trace_cell(parent_label.portal)
    child_cell = state.trace_cell(child_label.portal)
    if child_label.direction != "bwd" or child_cell == parent_cell:
        return False

    exit_rows = _backward_crossing_exit_rows(state, child_label.portal)
    reachable_nodes = sorted(_backward_reachable_nodes_in_cell(state, child_label.portal))
    exit_cells = sorted({int(row["cell"]) for row in exit_rows})
    usable_exit_cells = [
        cell
        for cell in exit_cells
        if _cell_visit_count(child_label, cell) < _effective_max_cell_visits(state)
    ]
    all_exit_cells_already_at_visit_limit = bool(exit_cells) and not usable_exit_cells
    only_return_to_previous_cell = bool(exit_cells) and set(exit_cells) == {
        parent_cell
    }
    dead_cell_entry = not usable_exit_cells

    if dead_cell_entry:
        state.audit.backward_children_entering_dead_cell += 1
        _increment_count(state.audit.backward_dead_cells_by_id, child_cell)
    if only_return_to_previous_cell:
        state.audit.backward_children_entering_cell_with_only_return_to_previous_cell += 1

    diagnostic = {
        "context": context,
        "parent_portal": parent_label.portal,
        "parent_cell": parent_cell,
        "child_portal": child_label.portal,
        "child_cell": child_cell,
        "edge_src": edge.src,
        "edge_dst": edge.dst,
        "visited_cells": sorted(child_label.visited_cells),
        "revisited_cells": sorted(child_label.revisited_cells),
        "backward_reachable_nodes_inside_child_cell": reachable_nodes,
        "possible_backward_exits_after_entry": exit_rows,
        "exit_cells": exit_cells,
        "usable_exit_cells_below_visit_limit": usable_exit_cells,
        "all_exit_cells_already_at_visit_limit": all_exit_cells_already_at_visit_limit,
        "only_return_to_previous_cell": only_return_to_previous_cell,
        "dead_cell_entry": dead_cell_entry,
    }

    logger.debug(
        "Backward cul-de-sac entry context=%s parent=%s parent_cell=%s "
        "child=%s child_cell=%s visited_cells=%s exit_cells=%s "
        "reachable_nodes_inside_child_cell=%s usable_exit_cells=%s "
        "all_exit_cells_already_at_visit_limit=%s "
        "only_return_to_previous_cell=%s dead_cell_entry=%s "
        "possible_backward_exits=%s",
        context,
        parent_label.portal,
        parent_cell,
        child_label.portal,
        child_cell,
        sorted(child_label.visited_cells),
        exit_cells,
        reachable_nodes,
        usable_exit_cells,
        all_exit_cells_already_at_visit_limit,
        only_return_to_previous_cell,
        dead_cell_entry,
        exit_rows,
    )

    if child_cell == CELL93_DIAGNOSTIC_CELL:
        state.audit.backward_cell93_entry_diagnostics.append(diagnostic)
        logger.debug("Cell 93 backward entry diagnostic %s", diagnostic)
    return dead_cell_entry


def _prune_backward_culdesac_child(
    state: AnytimeSearchState,
    child_label: PortalLabel,
    *,
    dead_cell_entry: bool,
    context: str,
    trace: BackwardPopTrace | None = None,
) -> bool:
    if not dead_cell_entry:
        return False

    child_cell = state.trace_cell(child_label.portal)
    state.audit.backward_culdesac_children_pruned += 1
    _increment_count(state.audit.backward_culdesac_cells_pruned_by_id, child_cell)
    _record_backward_rejection(state, "culdesac_cell")
    _record_trace_child_rejection(trace, "culdesac_cell")
    logger.debug(
        "Backward cul-de-sac child pruned context=%s child=%s child_cell=%s "
        "visited_cells=%s",
        context,
        child_label.portal,
        child_cell,
        sorted(child_label.visited_cells),
    )
    if child_label.parent is not None:
        log_child_generation(
            state,
            child_label.parent,
            child_label=child_label,
            edge=child_label.parent_edge,
            status="rejected",
            reason="culdesac_cell",
        )
    return True


def _handle_terminal_cell_child(
    state: AnytimeSearchState,
    parent_label: PortalLabel,
    child_label: PortalLabel,
    *,
    edge: OverlayEdge,
    context: str,
    prejoin_archived: bool = False,
    trace: BackwardPopTrace | None = None,
) -> bool:
    child_cell = state.trace_cell(child_label.portal)
    if not _is_opposite_terminal_cell_transition(state, parent_label, child_cell):
        return False

    terminal_portal = (
        state.query.source if child_label.direction == "bwd" else state.query.target
    )
    state.audit.terminal_completion_attempts += 1
    archived = prejoin_archived or _try_terminal_completion(state, child_label)
    if archived:
        state.audit.terminal_completion_successes += 1

    logger.debug(
        "Terminal completion attempt context=%s dir=%s parent=%s child=%s "
        "edge_src=%s edge_dst=%s child_cell=%s terminal_portal=%s archived=%s",
        context,
        child_label.direction,
        parent_label.portal,
        child_label.portal,
        edge.src,
        edge.dst,
        child_cell,
        terminal_portal,
        archived,
    )

    if child_label.portal == terminal_portal:
        logger.debug(
            "Terminal-cell child kept context=%s dir=%s child=%s reason=actual_terminal_portal",
            context,
            child_label.direction,
            child_label.portal,
        )
        return False

    if not archived:
        state.audit.terminal_leak_rejected_after_failed_completion += 1
        _increment_direction_count(
            state.audit.terminal_leak_labels_rejected_by_direction,
            child_label.direction,
        )
        if child_label.direction == "bwd":
            state.audit.rejected_backward_enter_source_cell += 1
            state.audit.backward_rejected_terminal_anti_leak += 1
            _record_backward_rejection(state, "terminal_anti_leak")
            _record_trace_child_rejection(trace, "terminal_anti_leak")
        else:
            state.audit.rejected_forward_enter_target_cell += 1
        logger.debug(
            "Terminal-cell leak rejected context=%s dir=%s child=%s route=%s",
            context,
            child_label.direction,
            child_label.portal,
            _fmt_route_vector(child_label.metrics),
        )
        log_child_generation(
            state,
            parent_label,
            child_label=child_label,
            edge=edge,
            status="rejected",
            reason="terminal_cell_leak",
        )
        return True

    logger.debug(
        "Terminal-cell child archived_without_storage context=%s dir=%s child=%s route=%s",
        context,
        child_label.direction,
        child_label.portal,
        _fmt_route_vector(child_label.metrics),
    )
    return True


def grow_portal_shortcuts_in_cell(
    state: AnytimeSearchState,
    portal: int,
    direction: Direction,
    trace: BackwardPopTrace | None = None,
) -> List[OverlayEdge]:
    inserted_edges: List[OverlayEdge] = []
    cell_id = state.partition.get(portal)
    if cell_id is None:
        return inserted_edges

    key = (cell_id, portal, direction)
    engine = state.local_engines.get(key)
    if engine is None:
        engine = LocalCellEngineState(cell_id=cell_id, origin_portal=portal, direction=direction)
        engine.best_length[portal] = 0.0
        heapq.heappush(
            engine.frontier,
            (
                0.0,
                next(state.local_counter),
                portal,
                RouteAccumulator(),
                (portal,),
                None,
                None,
                0,
            ),
        )
        state.local_engines[key] = engine
        logger.debug(
            "Local engine init cell=%s portal=%s dir=%s",
            cell_id,
            portal,
            direction,
        )

    if engine.exhausted:
        return inserted_edges

    allowed_nodes = state.nodes_by_cell.get(cell_id, set())
    if not allowed_nodes:
        engine.exhausted = True
        logger.debug(
            "Local engine exhausted cell=%s portal=%s dir=%s reason=no_nodes",
            cell_id,
            portal,
            direction,
        )
        return inserted_edges

    cell93_run = (
        cell_id == CELL93_DIAGNOSTIC_CELL
        and portal == CELL93_DIAGNOSTIC_PORTAL
        and direction == "bwd"
    )
    cell93_local_diag: Cell93LocalDiag | None = None
    if cell93_run:
        cell93_local_diag = {
            "cell": cell_id,
            "portal": portal,
            "direction": direction,
            "initial_queue_size": len(engine.frontier),
            "local_states_popped": 0,
            "reverse_neighbors_seen": 0,
            "internal_neighbors_considered": 0,
            "skipped_by_cell_boundary": 0,
            "skipped_by_visited_or_local_dominance": 0,
            "reaching_active_portals": [],
        }
        logger.debug(
            "Cell93 local engine start diagnostics %s",
            cell93_local_diag,
        )

    expanded = 0
    while engine.frontier and expanded < state.config.local_expand_limit:
        (
            _,
            _,
            node,
            metrics,
            path_nodes,
            first_road_id,
            last_road_id,
            road_changes,
        ) = heapq.heappop(engine.frontier)
        expanded += 1
        engine.expansions += 1
        if cell93_local_diag is not None:
            cell93_local_diag["local_states_popped"] = int(
                cell93_local_diag["local_states_popped"]
            ) + 1

        if direction == "fwd":
            to, weights, _ = state.G.neighbors(node)
            for idx, nxt in enumerate(to):
                vv = int(nxt)
                if vv not in allowed_nodes:
                    continue
                edge_metrics = RouteAccumulator.from_edge_weights(weights[idx])
                new_metrics = metrics.plus(edge_metrics)
                best_len = engine.best_length.get(vv)
                if best_len is not None and new_metrics.length >= best_len:
                    continue
                engine.best_length[vv] = new_metrics.length
                new_path = tuple(list(path_nodes) + [vv])
                edge_road_id = _road_id_from_neighbor(state.G, node, idx)
                (
                    new_first_road_id,
                    new_last_road_id,
                    new_road_changes,
                ) = _append_road_id(
                    first_road_id,
                    last_road_id,
                    road_changes,
                    edge_road_id,
                )
                if vv in state.active_portals and vv != portal:
                    engine.discovered_portals.add(vv)
                    if direction == "bwd":
                        trace_for_count = trace
                    else:
                        trace_for_count = None
                    if trace_for_count is not None:
                        trace_for_count.local_shortcuts_discovered += 1
                    edge = OverlayEdge(
                        src=portal,
                        dst=vv,
                        metrics=new_metrics,
                        path_nodes=new_path,
                        first_road_id=new_first_road_id,
                        last_road_id=new_last_road_id,
                        road_changes=new_road_changes,
                        kind="local",
                    )
                    if state.add_overlay_edge(edge):
                        inserted_edges.append(edge)
                        if trace_for_count is not None:
                            trace_for_count.local_edges_inserted += 1
                        _increment_count(
                            state.audit.local_shortcuts_discovered_by_cell,
                            cell_id,
                        )
                        logger.debug(
                            "Local shortcut discovered cell=%s dir=%s src=%s dst=%s route=%s path_len=%s",
                            cell_id,
                            direction,
                            portal,
                            vv,
                            _fmt_route_vector(new_metrics),
                            len(new_path),
                        )
                    else:
                        _record_trace_child_rejection(
                            trace_for_count,
                            "duplicate_or_capacity",
                        )
                heapq.heappush(
                    engine.frontier,
                    (
                        new_metrics.length,
                        next(state.local_counter),
                        vv,
                        new_metrics,
                        new_path,
                        new_first_road_id,
                        new_last_road_id,
                        new_road_changes,
                    ),
                )
            continue

        for pred, edge_metrics, edge_road_id in state.reverse_adj.get(node, []):
            if cell93_local_diag is not None:
                cell93_local_diag["reverse_neighbors_seen"] = int(
                    cell93_local_diag["reverse_neighbors_seen"]
                ) + 1
            if pred not in allowed_nodes:
                if cell93_local_diag is not None:
                    cell93_local_diag["skipped_by_cell_boundary"] = int(
                        cell93_local_diag["skipped_by_cell_boundary"]
                    ) + 1
                continue
            if cell93_local_diag is not None:
                cell93_local_diag["internal_neighbors_considered"] = int(
                    cell93_local_diag["internal_neighbors_considered"]
                ) + 1
            new_metrics = edge_metrics.plus(metrics)
            best_len = engine.best_length.get(pred)
            if best_len is not None and new_metrics.length >= best_len:
                if direction == "bwd":
                    _record_trace_child_rejection(trace, "already_seen")
                if cell93_local_diag is not None:
                    cell93_local_diag[
                        "skipped_by_visited_or_local_dominance"
                    ] = int(
                        cell93_local_diag[
                            "skipped_by_visited_or_local_dominance"
                        ]
                    ) + 1
                    if pred in state.active_portals and pred != portal:
                        reaching = cell93_local_diag["reaching_active_portals"]
                        assert isinstance(reaching, list)
                        reaching.append(
                            {
                                "portal": pred,
                                "from_node": node,
                                "reason": "not_inserted_local_dominance",
                                "candidate_length": new_metrics.length,
                                "best_length": best_len,
                            }
                        )
                continue
            engine.best_length[pred] = new_metrics.length
            new_path = tuple([pred] + list(path_nodes))
            (
                new_first_road_id,
                new_last_road_id,
                new_road_changes,
            ) = _prepend_road_id(
                first_road_id,
                last_road_id,
                road_changes,
                edge_road_id,
            )
            if pred in state.active_portals and pred != portal:
                engine.discovered_portals.add(pred)
                if direction == "bwd" and trace is not None:
                    trace.local_shortcuts_discovered += 1
                if cell93_local_diag is not None:
                    reaching = cell93_local_diag["reaching_active_portals"]
                    assert isinstance(reaching, list)
                    reaching.append(
                        {
                            "portal": pred,
                            "from_node": node,
                            "path_len": len(new_path),
                            "route": _fmt_route_vector(new_metrics),
                            "reason": "candidate",
                        }
                    )
                edge = OverlayEdge(
                    src=pred,
                    dst=portal,
                    metrics=new_metrics,
                    path_nodes=new_path,
                    first_road_id=new_first_road_id,
                    last_road_id=new_last_road_id,
                    road_changes=new_road_changes,
                    kind="local",
                )
                if state.add_overlay_edge(edge):
                    inserted_edges.append(edge)
                    if direction == "bwd" and trace is not None:
                        trace.local_edges_inserted += 1
                    if cell93_local_diag is not None:
                        reaching = cell93_local_diag["reaching_active_portals"]
                        assert isinstance(reaching, list)
                        reaching[-1]["reason"] = "inserted"
                    _increment_count(
                        state.audit.local_shortcuts_discovered_by_cell,
                        cell_id,
                    )
                    logger.debug(
                        "Local shortcut discovered cell=%s dir=%s src=%s dst=%s route=%s path_len=%s",
                        cell_id,
                        direction,
                        pred,
                        portal,
                        _fmt_route_vector(new_metrics),
                        len(new_path),
                    )
                else:
                    _record_trace_child_rejection(trace, "duplicate_or_capacity")
                    if cell93_local_diag is not None:
                        reaching = cell93_local_diag["reaching_active_portals"]
                        assert isinstance(reaching, list)
                        reaching[-1]["reason"] = "not_inserted_duplicate_or_capacity"
                    logger.debug(
                        "Backward local shortcut discovered cell=%s src=%s dst=%s route=%s path_len=%s",
                        cell_id,
                        pred,
                        portal,
                        _fmt_route_vector(new_metrics),
                        len(new_path),
                    )
            heapq.heappush(
                engine.frontier,
                (
                    new_metrics.length,
                    next(state.local_counter),
                    pred,
                    new_metrics,
                    new_path,
                    new_first_road_id,
                    new_last_road_id,
                    new_road_changes,
                ),
            )

    if not engine.frontier:
        engine.exhausted = True
        logger.debug(
            "Local engine exhausted cell=%s portal=%s dir=%s expansions=%s shortcuts=%s",
            cell_id,
            portal,
            direction,
            engine.expansions,
            len(engine.discovered_portals),
        )
    if cell93_local_diag is not None:
        cell93_local_diag["final_queue_size"] = len(engine.frontier)
        cell93_local_diag["engine_exhausted"] = engine.exhausted
        cell93_local_diag["inserted_edges_returned"] = len(inserted_edges)
        runs = state.audit.cell93_diagnostics.setdefault(
            "local_engine_2515_bwd_runs",
            [],
        )
        assert isinstance(runs, list)
        runs.append(cell93_local_diag)
        logger.debug("Cell93 local engine run diagnostics %s", cell93_local_diag)
    return inserted_edges


def _expand_representative_label(
    state: AnytimeSearchState,
    label: PortalLabel,
    skip_edge_ids: Set[int] | None = None,
    trace: BackwardPopTrace | None = None,
) -> int:
    skip_edge_ids = skip_edge_ids or set()
    current_cell = state.trace_cell(label.portal)
    generated_children = 0
    if label.direction == "fwd":
        edges = state.overlay.out_edges.get(label.portal, [])
        logger.debug(
            "Expand label portal=%s dir=%s overlay_edges=%s",
            label.portal,
            label.direction,
            len(edges),
        )
        for edge in edges:
            if id(edge) in skip_edge_ids:
                continue
            next_cell = state.trace_cell(edge.dst)
            cell_history = _extend_label_cell_history(state, label, edge)
            if cell_history is None:
                _record_rejected_third_cell_visit(state, label.direction)
                log_child_generation(
                    state,
                    label,
                    edge=edge,
                    status="rejected",
                    reason="third_cell_visit",
                )
                continue
            new_metrics = label.metrics.plus(edge.metrics)
            if state.is_partially_hopeless(new_metrics, label.direction):
                log_child_generation(
                    state,
                    label,
                    edge=edge,
                    status="rejected",
                    reason="metric_upper_bound",
                )
                continue
            _record_second_cell_visits_allowed(
                state,
                label.direction,
                cell_history.second_visits_added,
            )
            next_last_road_id, next_road_changes = _extend_label_road_continuity(
                label,
                edge,
            )
            next_label = PortalLabel(
                portal=edge.dst,
                direction="fwd",
                metrics=new_metrics,
                priority=state.partial_priority(new_metrics),
                visited_cells=cell_history.visited_cells,
                revisited_cells=cell_history.revisited_cells,
                last_road_id=next_last_road_id,
                road_changes=next_road_changes,
                parent=label,
                parent_edge=edge,
            )
            generated_children += 1
            prejoin_archived = _attempt_child_joins_before_terminal_gate(
                state,
                label,
                next_label,
                edge=edge,
                context="expand",
            )
            if _handle_terminal_cell_child(
                state,
                label,
                next_label,
                edge=edge,
                context="expand",
                prejoin_archived=prejoin_archived,
                trace=trace,
            ):
                continue
            logger.debug(
                "Expand successor src=%s dst=%s dir=%s kind=%s priority=%.4f route=%s",
                label.portal,
                edge.dst,
                "fwd",
                edge.kind,
                next_label.priority,
                _fmt_route_vector(next_label.metrics),
            )
            if state.enqueue(next_label):
                log_child_generation(
                    state,
                    label,
                    child_label=next_label,
                    edge=edge,
                    status="enqueued",
                )
            else:
                log_child_generation(
                    state,
                    label,
                    child_label=next_label,
                    edge=edge,
                    status="rejected",
                    reason="length_completion_lower_bound",
                )
        return generated_children

    edges = state.overlay.in_edges.get(label.portal, [])
    logger.debug(
        "Expand label portal=%s dir=%s overlay_edges=%s",
        label.portal,
        label.direction,
        len(edges),
    )
    logger.debug(
        "Backward overlay in-edge count portal=%s in_edges=%s",
        label.portal,
        len(edges),
    )
    for edge in edges:
        if id(edge) in skip_edge_ids:
            logger.debug(
                "Backward successor skipped portal=%s edge_src=%s reason=newly_enqueued",
                label.portal,
                edge.src,
            )
            continue
        next_cell = state.trace_cell(edge.src)
        cell_history = _extend_label_cell_history(state, label, edge)
        if cell_history is None:
            state.audit.backward_rejected_visited_cells += 1
            _record_rejected_third_cell_visit(state, label.direction)
            _record_backward_rejection(state, "third_cell_visit")
            _record_trace_child_rejection(trace, "third_cell_visit")
            logger.debug(
                "Backward successor skipped portal=%s edge_src=%s reason=third_cell_visit cell=%s",
                label.portal,
                edge.src,
                next_cell,
            )
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="third_cell_visit",
            )
            continue
        new_metrics = edge.metrics.plus(label.metrics)
        if state.is_partially_hopeless(new_metrics, label.direction):
            if new_metrics.length > float(state.query.constraints.upper[0]):
                _record_trace_child_rejection(trace, "length")
            if new_metrics.elevation > float(state.query.constraints.upper[1]):
                _record_trace_child_rejection(trace, "elevation")
            logger.debug(
                "Backward successor skipped portal=%s edge_src=%s reason=hopeless route=%s",
                label.portal,
                edge.src,
                _fmt_route_vector(new_metrics),
            )
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="metric_upper_bound",
            )
            continue
        _record_second_cell_visits_allowed(
            state,
            label.direction,
            cell_history.second_visits_added,
        )
        next_last_road_id, next_road_changes = _extend_label_road_continuity(
            label,
            edge,
        )
        next_label = PortalLabel(
            portal=edge.src,
            direction="bwd",
            metrics=new_metrics,
            priority=state.partial_priority(new_metrics),
            visited_cells=cell_history.visited_cells,
            revisited_cells=cell_history.revisited_cells,
            last_road_id=next_last_road_id,
            road_changes=next_road_changes,
            parent=label,
            parent_edge=edge,
        )
        generated_children += 1
        backward_culdesac_entry = False
        if edge.kind == "inter" and next_cell != current_cell:
            backward_culdesac_entry = _record_backward_child_entering_new_cell(
                state,
                label,
                next_label,
                edge=edge,
                context="expand",
            )
        prejoin_archived = _attempt_child_joins_before_terminal_gate(
            state,
            label,
            next_label,
            edge=edge,
            context="expand",
        )
        if _handle_terminal_cell_child(
            state,
            label,
            next_label,
            edge=edge,
            context="expand",
            prejoin_archived=prejoin_archived,
            trace=trace,
        ):
            continue
        if _prune_backward_culdesac_child(
            state,
            next_label,
            dead_cell_entry=backward_culdesac_entry,
            context="expand",
            trace=trace,
        ):
            continue
        logger.debug(
            "Expand successor src=%s dst=%s dir=%s kind=%s priority=%.4f route=%s",
            label.portal,
            edge.src,
            "bwd",
            edge.kind,
            next_label.priority,
            _fmt_route_vector(next_label.metrics),
        )
        if state.enqueue(next_label):
            log_child_generation(
                state,
                label,
                child_label=next_label,
                edge=edge,
                status="enqueued",
            )
            logger.debug(
                "Backward successor enqueued parent=%s child=%s kind=%s priority=%.4f route=%s",
                label.portal,
                next_label.portal,
                edge.kind,
                next_label.priority,
                _fmt_route_vector(next_label.metrics),
            )
        else:
            _record_trace_child_rejection(trace, "length_completion_lower_bound")
            log_child_generation(
                state,
                label,
                child_label=next_label,
                edge=edge,
                status="rejected",
                reason="length_completion_lower_bound",
            )
    return generated_children


def _directional_overlay_degree(state: AnytimeSearchState, label: PortalLabel) -> int:
    if label.direction == "fwd":
        return len(state.overlay.out_edges.get(label.portal, []))
    return len(state.overlay.in_edges.get(label.portal, []))


def _local_engine_has_pending_work(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> bool:
    cell_id = state.partition.get(label.portal)
    if cell_id is None:
        return False
    engine = state.local_engines.get((cell_id, label.portal, label.direction))
    return engine is not None and bool(engine.frontier) and not engine.exhausted


def _has_usable_forward_cross_cell_edge(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> bool:
    current_cell = state.trace_cell(label.portal)
    return any(
        state.trace_cell(edge.dst) != current_cell
        and not _edge_extension_would_exceed_cell_visit_limit(state, label, edge)
        for edge in state.overlay.out_edges.get(label.portal, [])
    )


def _enqueue_child_label_through_edge(
    state: AnytimeSearchState,
    label: PortalLabel,
    edge: OverlayEdge,
    trace: BackwardPopTrace | None = None,
    child_source: Literal[
        "local_shortcut",
        "repair",
        "directional_repair",
        "bridge",
    ] = "local_shortcut",
) -> bool:
    current_cell = state.trace_cell(label.portal)
    backward_culdesac_entry = False
    if label.direction == "fwd":
        if edge.src != label.portal:
            return False
        next_cell = state.trace_cell(edge.dst)
        cell_history = _extend_label_cell_history(state, label, edge)
        if cell_history is None:
            _record_rejected_third_cell_visit(state, label.direction)
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="third_cell_visit",
            )
            return False
        new_metrics = label.metrics.plus(edge.metrics)
        if state.is_partially_hopeless(new_metrics, label.direction):
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="metric_upper_bound",
            )
            return False
        _record_second_cell_visits_allowed(
            state,
            label.direction,
            cell_history.second_visits_added,
        )
        next_last_road_id, next_road_changes = _extend_label_road_continuity(
            label,
            edge,
        )
        next_label = PortalLabel(
            portal=edge.dst,
            direction="fwd",
            metrics=new_metrics,
            priority=state.partial_priority(new_metrics),
            visited_cells=cell_history.visited_cells,
            revisited_cells=cell_history.revisited_cells,
            last_road_id=next_last_road_id,
            road_changes=next_road_changes,
            parent=label,
            parent_edge=edge,
        )
    else:
        if edge.dst != label.portal:
            return False
        next_cell = state.trace_cell(edge.src)
        cell_history = _extend_label_cell_history(state, label, edge)
        if cell_history is None:
            state.audit.backward_rejected_visited_cells += 1
            _record_rejected_third_cell_visit(state, label.direction)
            _record_backward_rejection(state, "third_cell_visit")
            _record_trace_child_rejection(trace, "third_cell_visit")
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="third_cell_visit",
            )
            return False
        new_metrics = edge.metrics.plus(label.metrics)
        if state.is_partially_hopeless(new_metrics, label.direction):
            if new_metrics.length > float(state.query.constraints.upper[0]):
                _record_trace_child_rejection(trace, "length")
            if new_metrics.elevation > float(state.query.constraints.upper[1]):
                _record_trace_child_rejection(trace, "elevation")
            log_child_generation(
                state,
                label,
                edge=edge,
                status="rejected",
                reason="metric_upper_bound",
            )
            return False
        _record_second_cell_visits_allowed(
            state,
            label.direction,
            cell_history.second_visits_added,
        )
        next_last_road_id, next_road_changes = _extend_label_road_continuity(
            label,
            edge,
        )
        next_label = PortalLabel(
            portal=edge.src,
            direction="bwd",
            metrics=new_metrics,
            priority=state.partial_priority(new_metrics),
            visited_cells=cell_history.visited_cells,
            revisited_cells=cell_history.revisited_cells,
            last_road_id=next_last_road_id,
            road_changes=next_road_changes,
            parent=label,
            parent_edge=edge,
        )
        if edge.kind == "inter" and next_cell != current_cell:
            backward_culdesac_entry = _record_backward_child_entering_new_cell(
                state,
                label,
                next_label,
                edge=edge,
                context=f"spawn_child:{child_source}",
            )
        if trace is not None and child_source == "local_shortcut":
            trace.children_from_local_shortcuts += 1

    prejoin_archived = _attempt_child_joins_before_terminal_gate(
        state,
        label,
        next_label,
        edge=edge,
        context="spawn_child",
    )
    if _handle_terminal_cell_child(
        state,
        label,
        next_label,
        edge=edge,
        context="spawn_child",
        prejoin_archived=prejoin_archived,
        trace=trace,
    ):
        return False
    if _prune_backward_culdesac_child(
        state,
        next_label,
        dead_cell_entry=backward_culdesac_entry,
        context=f"spawn_child:{child_source}",
        trace=trace,
    ):
        return False

    logger.debug(
        "Spawn child label parent=%s child=%s dir=%s kind=%s priority=%.4f route=%s",
        label.portal,
        next_label.portal,
        label.direction,
        edge.kind,
        next_label.priority,
        _fmt_route_vector(next_label.metrics),
    )
    if not state.enqueue(next_label):
        if label.direction == "bwd":
            _record_trace_child_rejection(trace, "length_completion_lower_bound")
        log_child_generation(
            state,
            label,
            child_label=next_label,
            edge=edge,
            status="rejected",
            reason="length_completion_lower_bound",
        )
        return False
    log_child_generation(
        state,
        label,
        child_label=next_label,
        edge=edge,
        status="enqueued",
    )
    if label.direction == "bwd":
        logger.debug(
            "Backward successor enqueued parent=%s child=%s kind=%s priority=%.4f route=%s",
            label.portal,
            next_label.portal,
            edge.kind,
            next_label.priority,
            _fmt_route_vector(next_label.metrics),
        )
    return True


def _has_usable_backward_cross_cell_edge(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> bool:
    current_cell = state.trace_cell(label.portal)
    return any(
        state.trace_cell(edge.src) != current_cell
        and not _edge_extension_would_exceed_cell_visit_limit(state, label, edge)
        for edge in state.overlay.in_edges.get(label.portal, [])
    )


def _crossing_exit_is_represented(
    state: AnytimeSearchState,
    pred: int,
    node: int,
) -> bool:
    if pred not in state.active_portals or node not in state.active_portals:
        return False
    return any(
        edge.dst == node
        for edge in state.overlay.out_edges.get(pred, [])
    )


def _forward_crossing_exit_is_represented(
    state: AnytimeSearchState,
    node: int,
    dst: int,
) -> bool:
    if node not in state.active_portals or dst not in state.active_portals:
        return False
    return any(
        edge.dst == dst
        for edge in state.overlay.out_edges.get(node, [])
    )


def _find_forward_directional_repair_candidates(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> List[ForwardDirectionalRepairCandidate]:
    cell_id = state.trace_cell(label.portal)
    allowed_nodes = state.nodes_by_cell.get(cell_id, set())
    if not allowed_nodes:
        return []

    frontier: List[tuple] = [
        (
            0.0,
            0.0,
            next(state.local_counter),
            label.portal,
            RouteAccumulator(),
            (label.portal,),
            None,
            None,
            0,
        )
    ]
    best: Dict[int, Tuple[float, float]] = {label.portal: (0.0, 0.0)}
    candidates: Dict[Tuple[int, int], ForwardDirectionalRepairCandidate] = {}
    expanded = 0
    scan_limit = max(0, state.config.forward_directional_repair_scan_limit)

    while (
        frontier
        and expanded < scan_limit
        and time.perf_counter() < state.deadline
    ):
        (
            length,
            elevation,
            _,
            node,
            metrics,
            path_nodes,
            first_road_id,
            last_road_id,
            road_changes,
        ) = heapq.heappop(frontier)
        if best.get(node) != (length, elevation):
            continue
        expanded += 1

        to, weights, _ = state.G.neighbors(node)
        for idx, nxt_raw in enumerate(to):
            nxt = int(nxt_raw)
            nxt_cell = state.partition.get(nxt)
            if nxt_cell is None:
                continue
            edge_metrics = RouteAccumulator.from_edge_weights(weights[idx])
            edge_road_id = _road_id_from_neighbor(state.G, node, idx)
            if nxt_cell != cell_id:
                if nxt not in state.kept_nodes:
                    continue
                state.audit.forward_directional_repair_candidates_seen += 1
                if _cell_visit_count(label, nxt_cell) >= _effective_max_cell_visits(state):
                    continue
                state.audit.forward_directional_repair_candidates_unvisited += 1
                if _forward_crossing_exit_is_represented(state, node, nxt):
                    continue

                combined_metrics = metrics.plus(edge_metrics)
                (
                    combined_first_road_id,
                    combined_last_road_id,
                    combined_road_changes,
                ) = _append_road_id(
                    first_road_id,
                    last_road_id,
                    road_changes,
                    edge_road_id,
                )
                candidate = ForwardDirectionalRepairCandidate(
                    node=node,
                    dst=nxt,
                    dst_cell=nxt_cell,
                    local_metrics=metrics,
                    combined_metrics=combined_metrics,
                    path_nodes=path_nodes + (nxt,),
                    first_road_id=combined_first_road_id,
                    last_road_id=combined_last_road_id,
                    road_changes=combined_road_changes,
                )
                key = (nxt, nxt_cell)
                existing = candidates.get(key)
                if existing is None or (
                    candidate.local_metrics.length,
                    candidate.local_metrics.elevation,
                    *_extension_road_continuity_sort_fields(
                        label,
                        candidate.first_road_id,
                        candidate.last_road_id,
                        candidate.road_changes,
                    ),
                    candidate.road_changes,
                ) < (
                    existing.local_metrics.length,
                    existing.local_metrics.elevation,
                    *_extension_road_continuity_sort_fields(
                        label,
                        existing.first_road_id,
                        existing.last_road_id,
                        existing.road_changes,
                    ),
                    existing.road_changes,
                ):
                    candidates[key] = candidate
                continue

            if nxt not in allowed_nodes:
                continue
            new_metrics = metrics.plus(edge_metrics)
            new_key = (new_metrics.length, new_metrics.elevation)
            if new_key >= best.get(nxt, (float("inf"), float("inf"))):
                continue
            best[nxt] = new_key
            (
                new_first_road_id,
                new_last_road_id,
                new_road_changes,
            ) = _append_road_id(
                first_road_id,
                last_road_id,
                road_changes,
                edge_road_id,
            )
            heapq.heappush(
                frontier,
                (
                    new_metrics.length,
                    new_metrics.elevation,
                    next(state.local_counter),
                    nxt,
                    new_metrics,
                    path_nodes + (nxt,),
                    new_first_road_id,
                    new_last_road_id,
                    new_road_changes,
                ),
            )

    old_order = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.local_metrics.length,
            candidate.local_metrics.elevation,
            candidate.road_changes,
            candidate.node,
            candidate.dst,
        ),
    )
    new_order = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.local_metrics.length,
            candidate.local_metrics.elevation,
            *_extension_road_continuity_sort_fields(
                label,
                candidate.first_road_id,
                candidate.last_road_id,
                candidate.road_changes,
            ),
            candidate.road_changes,
            candidate.node,
            candidate.dst,
        ),
    )
    if [
        (candidate.node, candidate.dst, candidate.path_nodes)
        for candidate in old_order
    ] != [
        (candidate.node, candidate.dst, candidate.path_nodes)
        for candidate in new_order
    ]:
        state.audit.forward_directional_repair_ordering_changed_by_road_continuity += 1
    return new_order


def refine_forward_directional_portals(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> List[OverlayEdge]:
    if label.direction != "fwd":
        return []
    cell_id = state.trace_cell(label.portal)
    attempt_key = (cell_id, label.portal)
    if attempt_key in state.forward_directional_refinement_attempted_portals:
        return []
    if _has_usable_forward_cross_cell_edge(state, label):
        return []

    engine = state.local_engines.get((cell_id, label.portal, label.direction))
    if engine is None or (bool(engine.frontier) and not engine.exhausted):
        return []

    cell_budget = max(0, state.config.max_forward_directional_repairs_per_cell)
    cell_edges_used = state.forward_directional_repair_edges_by_cell.get(cell_id, 0)
    remaining_cell_budget = max(0, cell_budget - cell_edges_used)
    if remaining_cell_budget <= 0:
        _increment_count(
            state.audit.forward_directional_repair_budget_exhausted_by_cell,
            cell_id,
        )
        logger.debug(
            "Forward directional portal refinement skipped cell=%s portal=%s "
            "reason=cell_budget_exhausted used=%s budget=%s",
            cell_id,
            label.portal,
            cell_edges_used,
            cell_budget,
        )
        return []

    state.forward_directional_refinement_attempted_portals.add(attempt_key)
    state.audit.forward_directional_portal_refinements_attempted += 1
    _increment_count(state.audit.forward_directional_repair_attempted_cells, cell_id)
    candidates = _find_forward_directional_repair_candidates(state, label)
    limit = min(cell_budget, remaining_cell_budget)
    inserted_edges: List[OverlayEdge] = []

    for candidate in candidates[:limit]:
        state.kept_nodes.add(candidate.dst)
        state.active_portals.add(candidate.dst)
        state.nodes_by_cell.setdefault(candidate.dst_cell, set()).add(candidate.dst)
        edge = OverlayEdge(
            src=label.portal,
            dst=candidate.dst,
            metrics=candidate.combined_metrics,
            path_nodes=candidate.path_nodes,
            first_road_id=candidate.first_road_id,
            last_road_id=candidate.last_road_id,
            road_changes=candidate.road_changes,
            kind="inter",
        )
        if not state.add_overlay_edge(edge):
            continue
        inserted_edges.append(edge)
        state.forward_directional_repair_edges_by_cell[cell_id] = (
            state.forward_directional_repair_edges_by_cell.get(cell_id, 0) + 1
        )
        state.audit.forward_directional_portal_refinements_inserted += 1
        logger.debug(
            "Forward directional portal refinement cell=%s portal=%s "
            "crossing_node=%s dst=%s dst_cell=%s local_length=%.1f "
            "combined_route=%s path_len=%s",
            cell_id,
            label.portal,
            candidate.node,
            candidate.dst,
            candidate.dst_cell,
            candidate.local_metrics.length,
            _fmt_route_vector(candidate.combined_metrics),
            len(candidate.path_nodes),
        )
        _emit_trace_event(
            state,
            "directional_repair",
            portals=(label.portal, candidate.node, candidate.dst),
            cells=(cell_id, candidate.dst_cell),
            direction="fwd",
            portal=label.portal,
            crossing_node=candidate.node,
            repair_portal=candidate.dst,
            repair_cell=candidate.dst_cell,
            local_length=f"{candidate.local_metrics.length:.1f}",
            edge=edge.compact_summary(
                src_cell=cell_id,
                dst_cell=candidate.dst_cell,
            ),
        )
        if state.forward_directional_repair_edges_by_cell[cell_id] >= cell_budget:
            break

    if inserted_edges:
        _increment_count(state.audit.forward_directional_repair_cells, cell_id)
    logger.debug(
        "Forward directional portal refinement result cell=%s portal=%s "
        "budget_before=%s budget_after=%s candidates=%s inserted=%s",
        cell_id,
        label.portal,
        remaining_cell_budget,
        max(
            0,
            cell_budget - state.forward_directional_repair_edges_by_cell.get(cell_id, 0),
        ),
        len(candidates),
        len(inserted_edges),
    )
    return inserted_edges


def _find_backward_directional_repair_candidates(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> List[BackwardDirectionalRepairCandidate]:
    cell_id = state.trace_cell(label.portal)
    allowed_nodes = state.nodes_by_cell.get(cell_id, set())
    if not allowed_nodes:
        return []

    frontier: List[tuple] = [
        (
            0.0,
            0.0,
            next(state.local_counter),
            label.portal,
            RouteAccumulator(),
            (label.portal,),
            None,
            None,
            0,
        )
    ]
    best: Dict[int, Tuple[float, float]] = {label.portal: (0.0, 0.0)}
    candidates: Dict[Tuple[int, int], BackwardDirectionalRepairCandidate] = {}
    expanded = 0
    scan_limit = max(0, state.config.backward_directional_repair_scan_limit)

    while (
        frontier
        and expanded < scan_limit
        and time.perf_counter() < state.deadline
    ):
        (
            length,
            elevation,
            _,
            node,
            metrics,
            path_nodes,
            first_road_id,
            last_road_id,
            road_changes,
        ) = heapq.heappop(frontier)
        if best.get(node) != (length, elevation):
            continue
        expanded += 1

        for pred, edge_metrics, edge_road_id in state.reverse_adj.get(node, []):
            pred_cell = state.trace_cell(pred)
            if pred_cell != cell_id:
                state.audit.backward_directional_repair_candidates_seen += 1
                if _cell_visit_count(label, pred_cell) >= _effective_max_cell_visits(state):
                    continue
                state.audit.backward_directional_repair_candidates_unvisited += 1
                if _crossing_exit_is_represented(state, pred, node):
                    continue

                combined_metrics = edge_metrics.plus(metrics)
                (
                    combined_first_road_id,
                    combined_last_road_id,
                    combined_road_changes,
                ) = _prepend_road_id(
                    first_road_id,
                    last_road_id,
                    road_changes,
                    edge_road_id,
                )
                candidate = BackwardDirectionalRepairCandidate(
                    pred=pred,
                    node=node,
                    pred_cell=pred_cell,
                    local_metrics=metrics,
                    combined_metrics=combined_metrics,
                    path_nodes=(pred,) + path_nodes,
                    first_road_id=combined_first_road_id,
                    last_road_id=combined_last_road_id,
                    road_changes=combined_road_changes,
                )
                key = (pred, pred_cell)
                existing = candidates.get(key)
                if existing is None or (
                    candidate.local_metrics.length,
                    candidate.local_metrics.elevation,
                    *_extension_road_continuity_sort_fields(
                        label,
                        candidate.first_road_id,
                        candidate.last_road_id,
                        candidate.road_changes,
                    ),
                    candidate.road_changes,
                ) < (
                    existing.local_metrics.length,
                    existing.local_metrics.elevation,
                    *_extension_road_continuity_sort_fields(
                        label,
                        existing.first_road_id,
                        existing.last_road_id,
                        existing.road_changes,
                    ),
                    existing.road_changes,
                ):
                    candidates[key] = candidate
                continue

            if pred not in allowed_nodes:
                continue
            new_metrics = edge_metrics.plus(metrics)
            new_key = (new_metrics.length, new_metrics.elevation)
            if new_key >= best.get(pred, (float("inf"), float("inf"))):
                continue
            best[pred] = new_key
            (
                new_first_road_id,
                new_last_road_id,
                new_road_changes,
            ) = _prepend_road_id(
                first_road_id,
                last_road_id,
                road_changes,
                edge_road_id,
            )
            heapq.heappush(
                frontier,
                (
                    new_metrics.length,
                    new_metrics.elevation,
                    next(state.local_counter),
                    pred,
                    new_metrics,
                    (pred,) + path_nodes,
                    new_first_road_id,
                    new_last_road_id,
                    new_road_changes,
                ),
            )

    old_order = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.local_metrics.length,
            candidate.local_metrics.elevation,
            candidate.road_changes,
            candidate.pred,
            candidate.node,
        ),
    )
    new_order = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.local_metrics.length,
            candidate.local_metrics.elevation,
            *_extension_road_continuity_sort_fields(
                label,
                candidate.first_road_id,
                candidate.last_road_id,
                candidate.road_changes,
            ),
            candidate.road_changes,
            candidate.pred,
            candidate.node,
        ),
    )
    if [
        (candidate.pred, candidate.node, candidate.path_nodes)
        for candidate in old_order
    ] != [
        (candidate.pred, candidate.node, candidate.path_nodes)
        for candidate in new_order
    ]:
        state.audit.backward_directional_repair_ordering_changed_by_road_continuity += 1
    return new_order


def refine_backward_directional_portals(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> List[OverlayEdge]:
    if label.direction != "bwd" or len(label.visited_cells) <= 1:
        return []
    cell_id = state.trace_cell(label.portal)
    attempt_key = (cell_id, label.portal)
    if attempt_key in state.backward_directional_refinement_attempted_portals:
        return []
    if _has_usable_backward_cross_cell_edge(state, label):
        return []

    cell_budget = max(0, state.config.max_backward_directional_repairs_per_cell)
    cell_edges_used = state.backward_directional_repair_edges_by_cell.get(cell_id, 0)
    remaining_cell_budget = max(0, cell_budget - cell_edges_used)
    if remaining_cell_budget <= 0:
        _increment_count(
            state.audit.backward_directional_repair_budget_exhausted_by_cell,
            cell_id,
        )
        logger.debug(
            "Backward directional portal refinement skipped cell=%s portal=%s "
            "reason=cell_budget_exhausted used=%s budget=%s",
            cell_id,
            label.portal,
            cell_edges_used,
            cell_budget,
        )
        return []

    state.backward_directional_refinement_attempted_portals.add(attempt_key)
    state.audit.backward_directional_portal_refinements_attempted += 1
    _increment_count(state.audit.backward_directional_repair_attempted_cells, cell_id)
    candidates = _find_backward_directional_repair_candidates(state, label)
    limit = min(cell_budget, remaining_cell_budget)
    inserted_edges: List[OverlayEdge] = []

    for candidate in candidates[:limit]:
        state.kept_nodes.add(candidate.pred)
        state.active_portals.add(candidate.pred)
        state.nodes_by_cell.setdefault(candidate.pred_cell, set()).add(candidate.pred)
        edge = OverlayEdge(
            src=candidate.pred,
            dst=label.portal,
            metrics=candidate.combined_metrics,
            path_nodes=candidate.path_nodes,
            first_road_id=candidate.first_road_id,
            last_road_id=candidate.last_road_id,
            road_changes=candidate.road_changes,
            kind="inter",
        )
        if not state.add_overlay_edge(edge):
            continue
        inserted_edges.append(edge)
        state.backward_directional_repair_edges_by_cell[cell_id] = (
            state.backward_directional_repair_edges_by_cell.get(cell_id, 0) + 1
        )
        state.audit.backward_directional_portal_refinements_inserted += 1
        logger.debug(
            "Backward directional portal refinement cell=%s portal=%s "
            "crossing_node=%s pred=%s pred_cell=%s local_length=%.1f "
            "combined_route=%s path_len=%s",
            cell_id,
            label.portal,
            candidate.node,
            candidate.pred,
            candidate.pred_cell,
            candidate.local_metrics.length,
            _fmt_route_vector(candidate.combined_metrics),
            len(candidate.path_nodes),
        )
        _emit_trace_event(
            state,
            "directional_repair",
            portals=(label.portal, candidate.node, candidate.pred),
            cells=(cell_id, candidate.pred_cell),
            direction="bwd",
            portal=label.portal,
            crossing_node=candidate.node,
            repair_portal=candidate.pred,
            repair_cell=candidate.pred_cell,
            local_length=f"{candidate.local_metrics.length:.1f}",
            edge=edge.compact_summary(
                src_cell=candidate.pred_cell,
                dst_cell=cell_id,
            ),
        )
        if state.backward_directional_repair_edges_by_cell[cell_id] >= cell_budget:
            break

    if inserted_edges:
        _increment_count(state.audit.backward_directional_repair_cells, cell_id)
    logger.debug(
        "Backward directional portal refinement result cell=%s portal=%s "
        "budget_before=%s budget_after=%s candidates=%s inserted=%s",
        cell_id,
        label.portal,
        remaining_cell_budget,
        max(
            0,
            cell_budget - state.backward_directional_repair_edges_by_cell.get(cell_id, 0),
        ),
        len(candidates),
        len(inserted_edges),
    )
    return inserted_edges


def repair_backward_dead_portal(
    state: AnytimeSearchState,
    label: PortalLabel,
) -> List[OverlayEdge]:
    state.audit.backward_dead_portal_repairs_attempted += 1
    portal_cell = state.trace_cell(label.portal)
    inserted_edges: List[OverlayEdge] = []
    inserted_any = False

    for pred, edge_metrics, edge_road_id in state.reverse_adj.get(label.portal, []):
        pred_cell = state.trace_cell(pred)
        if pred_cell == portal_cell:
            continue

        state.kept_nodes.add(pred)
        state.active_portals.add(pred)
        state.nodes_by_cell.setdefault(pred_cell, set()).add(pred)

        edge = OverlayEdge(
            src=pred,
            dst=label.portal,
            metrics=edge_metrics,
            path_nodes=(pred, label.portal),
            first_road_id=edge_road_id,
            last_road_id=edge_road_id,
            road_changes=0,
            kind="inter",
        )
        inserted = state.add_overlay_edge(edge)
        logger.debug(
            "Backward dead portal repair portal=%s portal_cell=%s pred=%s "
            "pred_cell=%s inserted=%s active_pred=%s route=%s",
            label.portal,
            portal_cell,
            pred,
            pred_cell,
            inserted,
            pred in state.active_portals,
            _fmt_route_vector(edge_metrics),
        )
        if not inserted:
            continue

        inserted_any = True
        inserted_edges.append(edge)
        state.audit.backward_dead_portal_repairs_inserted += 1

    if inserted_any:
        _increment_count(state.audit.repaired_portals_by_cell, portal_cell)
    return inserted_edges


def _bounded_base_path_within_cell(
    state: AnytimeSearchState,
    source: int,
    target: int,
    cell_id: int,
) -> BasePathSegment | None:
    if source == target:
        return BasePathSegment(
            metrics=RouteAccumulator(),
            path_nodes=(source,),
            first_road_id=None,
            last_road_id=None,
            road_changes=0,
        )

    allowed_nodes = state.nodes_by_cell.get(cell_id, set())
    if source not in allowed_nodes or target not in allowed_nodes:
        return None

    frontier: List[tuple] = [
        (
            0.0,
            0.0,
            next(state.local_counter),
            source,
            RouteAccumulator(),
            (source,),
            None,
            None,
            0,
        )
    ]
    best: Dict[int, Tuple[float, float]] = {source: (0.0, 0.0)}
    expanded = 0
    scan_limit = max(0, state.config.bridge_refinement_scan_limit)

    while (
        frontier
        and expanded < scan_limit
        and time.perf_counter() < state.deadline
    ):
        (
            length,
            elevation,
            _,
            node,
            metrics,
            path_nodes,
            first_road_id,
            last_road_id,
            road_changes,
        ) = heapq.heappop(frontier)
        if best.get(node) != (length, elevation):
            continue
        if node == target:
            return BasePathSegment(
                metrics=metrics,
                path_nodes=path_nodes,
                first_road_id=first_road_id,
                last_road_id=last_road_id,
                road_changes=road_changes,
            )
        expanded += 1

        to, weights, _ = state.G.neighbors(node)
        for idx, nxt_raw in enumerate(to):
            nxt = int(nxt_raw)
            if nxt not in allowed_nodes:
                continue
            edge_metrics = RouteAccumulator.from_edge_weights(weights[idx])
            new_metrics = metrics.plus(edge_metrics)
            new_key = (new_metrics.length, new_metrics.elevation)
            if new_key >= best.get(nxt, (float("inf"), float("inf"))):
                continue
            best[nxt] = new_key
            edge_road_id = _road_id_from_neighbor(state.G, node, idx)
            (
                new_first_road_id,
                new_last_road_id,
                new_road_changes,
            ) = _append_road_id(
                first_road_id,
                last_road_id,
                road_changes,
                edge_road_id,
            )
            heapq.heappush(
                frontier,
                (
                    new_metrics.length,
                    new_metrics.elevation,
                    next(state.local_counter),
                    nxt,
                    new_metrics,
                    path_nodes + (nxt,),
                    new_first_road_id,
                    new_last_road_id,
                    new_road_changes,
                ),
            )
    return None


def _bridge_crossings(
    state: AnytimeSearchState,
    src_cell: int,
    dst_cell: int,
) -> List[BridgeCrossing]:
    return list(state.retained_crossings_by_cell_pair.get((src_cell, dst_cell), []))


def _bridge_crossing_proximity(
    state: AnytimeSearchState,
    crossing_src: int,
    crossing_dst: int,
    src_portals: Sequence[int],
    dst_portals: Sequence[int],
) -> Tuple[float, int, int]:
    xy_int = _graph_xy_int(state.G)
    if xy_int is None:
        return (float(abs(crossing_src - crossing_dst)), crossing_src, crossing_dst)

    src_xy = xy_int[crossing_src, :2].astype(np.float64)
    dst_xy = xy_int[crossing_dst, :2].astype(np.float64)
    src_distance = min(
        float(np.dot(src_xy - xy_int[portal, :2], src_xy - xy_int[portal, :2]))
        for portal in src_portals
    )
    dst_distance = min(
        float(np.dot(dst_xy - xy_int[portal, :2], dst_xy - xy_int[portal, :2]))
        for portal in dst_portals
    )
    return (src_distance + dst_distance, crossing_src, crossing_dst)


def _combine_bridge_segments(
    left: BasePathSegment,
    crossing_src: int,
    crossing_dst: int,
    crossing_metrics: RouteAccumulator,
    crossing_road_id: int | None,
    right: BasePathSegment,
) -> BasePathSegment:
    road_changes = (
        left.road_changes
        + right.road_changes
        + _road_change_delta(left.last_road_id, crossing_road_id)
        + _road_change_delta(crossing_road_id, right.first_road_id)
    )
    first_road_id = left.first_road_id
    if first_road_id is None:
        first_road_id = crossing_road_id
    if first_road_id is None:
        first_road_id = right.first_road_id
    last_road_id = right.last_road_id
    if last_road_id is None:
        last_road_id = crossing_road_id
    if last_road_id is None:
        last_road_id = left.last_road_id
    return BasePathSegment(
        metrics=left.metrics.plus(crossing_metrics).plus(right.metrics),
        path_nodes=_merge_segments(
            [
                left.path_nodes,
                (crossing_src, crossing_dst),
                right.path_nodes,
            ]
        ),
        first_road_id=first_road_id,
        last_road_id=last_road_id,
        road_changes=road_changes,
    )


def _bridge_candidates_for_direction(
    state: AnytimeSearchState,
    *,
    src_cell: int,
    dst_cell: int,
    src_portals: Sequence[int],
    dst_portals: Sequence[int],
    forward_portals: Set[int],
    backward_portals: Set[int],
) -> List[BridgeRefinementCandidate]:
    crossings = _bridge_crossings(state, src_cell, dst_cell)
    crossings.sort(
        key=lambda crossing: _bridge_crossing_proximity(
            state,
            crossing[0],
            crossing[1],
            src_portals,
            dst_portals,
        )
    )
    crossings = crossings[: max(0, state.config.bridge_refinement_scan_limit)]
    path_cache: Dict[Tuple[int, int, int], BasePathSegment | None] = {}
    candidates: List[BridgeRefinementCandidate] = []

    for crossing_src, crossing_dst, crossing_metrics, crossing_road_id in crossings:
        for src_portal in src_portals:
            left_key = (src_portal, crossing_src, src_cell)
            if left_key not in path_cache:
                path_cache[left_key] = _bounded_base_path_within_cell(
                    state,
                    src_portal,
                    crossing_src,
                    src_cell,
                )
            left = path_cache[left_key]
            if left is None:
                continue
            for dst_portal in dst_portals:
                right_key = (crossing_dst, dst_portal, dst_cell)
                if right_key not in path_cache:
                    path_cache[right_key] = _bounded_base_path_within_cell(
                        state,
                        crossing_dst,
                        dst_portal,
                        dst_cell,
                    )
                right = path_cache[right_key]
                if right is None:
                    continue
                combined = _combine_bridge_segments(
                    left,
                    crossing_src,
                    crossing_dst,
                    crossing_metrics,
                    crossing_road_id,
                    right,
                )
                joins_covered_directions = (
                    src_portal in forward_portals
                    and dst_portal in backward_portals
                )
                candidates.append(
                    BridgeRefinementCandidate(
                        edge=OverlayEdge(
                            src=src_portal,
                            dst=dst_portal,
                            metrics=combined.metrics,
                            path_nodes=combined.path_nodes,
                            first_road_id=combined.first_road_id,
                            last_road_id=combined.last_road_id,
                            road_changes=combined.road_changes,
                            kind="inter",
                            bridge_cell_pair=(src_cell, dst_cell),
                            bridge_corridor=(crossing_src, crossing_dst),
                        ),
                        forward_portal=(
                            src_portal
                            if src_portal in forward_portals
                            else dst_portal
                        ),
                        backward_portal=(
                            dst_portal
                            if dst_portal in backward_portals
                            else src_portal
                        ),
                        crossing_src=crossing_src,
                        crossing_dst=crossing_dst,
                        joins_covered_directions=joins_covered_directions,
                        connector_length=left.metrics.length + right.metrics.length,
                    )
                )
    return candidates


def _cell_pair_has_one_edge_join_coverage(
    state: AnytimeSearchState,
    forward_portals: Set[int],
    backward_portals: Set[int],
) -> bool:
    return any(
        edge.dst in backward_portals
        for portal in forward_portals
        for edge in state.overlay.out_edges.get(portal, [])
    )


def _covered_portals_by_cell(
    state: AnytimeSearchState,
    direction: Direction,
) -> Dict[int, Set[int]]:
    by_cell: Dict[int, Set[int]] = {}
    for portal in _covered_portals(state, direction):
        by_cell.setdefault(state.trace_cell(portal), set()).add(portal)
    return by_cell


def _bridge_cell_pair_is_eligible(
    state: AnytimeSearchState,
    cell_pair: Tuple[int, int],
    fwd_portals: Set[int],
    bwd_portals: Set[int],
) -> bool:
    fwd_cell, bwd_cell = cell_pair
    if fwd_cell == bwd_cell:
        return False
    if cell_pair in state.bridge_refinement_attempted_cell_pairs:
        return False

    limit = max(0, state.config.max_bridge_edges_per_cell_pair)
    selected_crossings = state.bridge_inserted_crossings_by_cell_pair.get(
        cell_pair,
        set(),
    )
    if len(selected_crossings) >= limit:
        return False

    direct_crossings = _bridge_crossings(state, fwd_cell, bwd_cell)
    reverse_crossings = _bridge_crossings(state, bwd_cell, fwd_cell)
    if not direct_crossings and not reverse_crossings:
        return False

    if (
        not selected_crossings
        and _cell_pair_has_one_edge_join_coverage(
            state,
            fwd_portals,
            bwd_portals,
        )
    ):
        return False

    return True


def _full_bridge_eligible_cell_pairs(
    state: AnytimeSearchState,
    fwd_by_cell: Dict[int, Set[int]] | None = None,
    bwd_by_cell: Dict[int, Set[int]] | None = None,
) -> Set[Tuple[int, int]]:
    fwd_cells = fwd_by_cell if fwd_by_cell is not None else _covered_portals_by_cell(
        state,
        "fwd",
    )
    bwd_cells = bwd_by_cell if bwd_by_cell is not None else _covered_portals_by_cell(
        state,
        "bwd",
    )
    eligible: Set[Tuple[int, int]] = set()
    for fwd_cell, fwd_portals in fwd_cells.items():
        for bwd_cell, bwd_portals in bwd_cells.items():
            cell_pair = (fwd_cell, bwd_cell)
            if _bridge_cell_pair_is_eligible(
                state,
                cell_pair,
                fwd_portals,
                bwd_portals,
            ):
                eligible.add(cell_pair)
    return eligible


def _bridge_candidate_has_cell_conflict(
    state: AnytimeSearchState,
    candidate: BridgeRefinementCandidate,
    fwd_label: PortalLabel,
    bwd_label: PortalLabel,
) -> bool:
    path_nodes = _reconstruct_one_edge_join_path(
        fwd_label,
        candidate.edge,
        bwd_label,
    )
    if path_nodes is None:
        return True
    ok, _, _ = _path_cell_visit_limit_result(state, path_nodes)
    return not ok


def _bridge_path_overlap(
    candidate: BridgeRefinementCandidate,
    selected_edges: Sequence[OverlayEdge],
) -> float:
    if not selected_edges:
        return 0.0
    candidate_nodes = set(candidate.edge.path_nodes)
    selected_nodes = {
        node
        for edge in selected_edges
        for node in edge.path_nodes
    }
    union = candidate_nodes | selected_nodes
    if not union:
        return 0.0
    return len(candidate_nodes & selected_nodes) / len(union)


def _bridge_selection_key(
    selection: BridgeJoinSelection,
    selected_edges: Sequence[OverlayEdge],
) -> tuple:
    route_vector = selection.metrics.route_vector()
    candidate = selection.candidate
    return (
        selection.score,
        candidate.connector_length,
        selection.metrics.elevation,
        -float(route_vector[2]),
        -float(route_vector[3]),
        selection.road_changes,
        _bridge_path_overlap(candidate, selected_edges),
        candidate.edge.src,
        candidate.edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
    )


def _bridge_fallback_candidate_key(
    candidate: BridgeRefinementCandidate,
    selected_edges: Sequence[OverlayEdge],
) -> tuple:
    edge = candidate.edge
    return (
        candidate.connector_length,
        edge.metrics.length,
        edge.metrics.elevation,
        edge.road_changes,
        _bridge_path_overlap(candidate, selected_edges),
        edge.src,
        edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
    )


def _connector_avg_popularity(metrics: RouteAccumulator) -> float:
    if metrics.length <= 0.0:
        return 0.0
    return metrics.popularity_length / metrics.length


def _connector_avg_width(metrics: RouteAccumulator) -> float:
    if metrics.length <= 0.0:
        return float("inf")
    return metrics.street_width_length / metrics.length


def _bridge_fallback_quality_candidate_key(
    candidate: BridgeRefinementCandidate,
    selected_edges: Sequence[OverlayEdge],
) -> tuple:
    edge = candidate.edge
    return (
        _connector_avg_width(edge.metrics),
        -_connector_avg_popularity(edge.metrics),
        edge.metrics.length,
        edge.metrics.elevation,
        edge.road_changes,
        _bridge_path_overlap(candidate, selected_edges),
        edge.src,
        edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
    )


def _bridge_candidate_path_key(
    candidate: BridgeRefinementCandidate,
) -> Tuple[int, int, int, int, Tuple[int, ...]]:
    edge = candidate.edge
    return (
        edge.src,
        edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
        edge.path_nodes,
    )


def _bridge_candidate_quality_summary(
    candidate: BridgeRefinementCandidate,
) -> Dict[str, float | int]:
    edge = candidate.edge
    return {
        "length": edge.metrics.length,
        "elevation": edge.metrics.elevation,
        "avg_pop": _connector_avg_popularity(edge.metrics),
        "avg_width": _connector_avg_width(edge.metrics),
        "road_changes": edge.road_changes,
    }


def _select_complementary_bridge_fallback_candidates(
    state: AnytimeSearchState,
    cell_pair: Tuple[int, int],
    primary_options: Dict[Tuple[int, int], BridgeRefinementCandidate],
    quality_options: Dict[Tuple[int, int], BridgeRefinementCandidate],
    selected_edges: Sequence[OverlayEdge],
    remaining_slots: int,
) -> List[Tuple[Tuple[int, int], BridgeRefinementCandidate, BridgeFallbackSelectionRole]]:
    if remaining_slots <= 0 or not primary_options:
        return []

    primary_order = sorted(
        primary_options.items(),
        key=lambda item: _bridge_fallback_candidate_key(item[1], selected_edges),
    )
    if remaining_slots == 1:
        return [
            (corridor_id, candidate, "primary")
            for corridor_id, candidate
            in primary_order
        ]

    if len(primary_options) > remaining_slots:
        state.audit.complementary_connector_sets_considered += 1

    chosen: List[
        Tuple[Tuple[int, int], BridgeRefinementCandidate, BridgeFallbackSelectionRole]
    ] = []
    chosen_corridors: Set[Tuple[int, int]] = set()
    chosen_paths: Set[Tuple[int, int, int, int, Tuple[int, ...]]] = set()

    def add_choice(
        corridor_id: Tuple[int, int],
        candidate: BridgeRefinementCandidate,
        role: BridgeFallbackSelectionRole,
    ) -> bool:
        path_key = _bridge_candidate_path_key(candidate)
        if corridor_id in chosen_corridors or path_key in chosen_paths:
            return False
        chosen.append((corridor_id, candidate, role))
        chosen_corridors.add(corridor_id)
        chosen_paths.add(path_key)
        return True

    primary_corridor, primary_candidate = primary_order[0]
    add_choice(primary_corridor, primary_candidate, "primary")

    quality_order = sorted(
        quality_options.items(),
        key=lambda item: _bridge_fallback_quality_candidate_key(item[1], selected_edges),
    )
    for corridor_id, candidate in quality_order:
        if not add_choice(corridor_id, candidate, "quality"):
            continue
        state.audit.complementary_quality_candidate_distinct += 1
        if len(state.audit.complementary_quality_candidate_examples) < 20:
            state.audit.complementary_quality_candidate_examples.append(
                {
                    "cell_pair": cell_pair,
                    "primary_crossing": (
                        primary_candidate.crossing_src,
                        primary_candidate.crossing_dst,
                    ),
                    "primary": _bridge_candidate_quality_summary(primary_candidate),
                    "quality_crossing": (
                        candidate.crossing_src,
                        candidate.crossing_dst,
                    ),
                    "quality": _bridge_candidate_quality_summary(candidate),
                }
            )
        logger.debug(
            "Complementary bridge fallback selected cells=%s->%s "
            "primary_crossing=%s->%s primary=%s "
            "quality_crossing=%s->%s quality=%s",
            cell_pair[0],
            cell_pair[1],
            primary_candidate.crossing_src,
            primary_candidate.crossing_dst,
            _bridge_candidate_quality_summary(primary_candidate),
            candidate.crossing_src,
            candidate.crossing_dst,
            _bridge_candidate_quality_summary(candidate),
        )
        break

    for corridor_id, candidate in primary_order:
        add_choice(corridor_id, candidate, "structural_fill")

    return chosen


def _generic_overlay_edge_rejection_reason(
    state: AnytimeSearchState,
    edge: OverlayEdge,
) -> str | None:
    bucket = state.overlay.out_edges.get(edge.src, [])
    same_pair = [e for e in bucket if e.dst == edge.dst and e.kind == edge.kind]
    if any(existing.path_nodes == edge.path_nodes for existing in same_pair):
        return "duplicate_path"
    if len(same_pair) >= state.config.max_shortcuts_per_pair:
        edge_key = _overlay_edge_capacity_key(edge)
        worst = max(same_pair, key=_overlay_edge_capacity_key)
        if edge_key >= _overlay_edge_capacity_key(worst):
            return "pair_capacity"
    return None


def _path_edges_are_real_directed(
    state: AnytimeSearchState,
    path_nodes: Sequence[int],
) -> bool:
    if len(path_nodes) < 2:
        return False
    for u, v in zip(path_nodes, path_nodes[1:]):
        to, _, _ = state.G.neighbors(int(u))
        if not any(int(nxt) == int(v) for nxt in to):
            return False
    return True


def _unique_csr_edge_index_for_pair(
    G: CompactDiGraph,
    u: int,
    v: int,
) -> int | None:
    start = int(G.offsets[int(u)])
    end = int(G.offsets[int(u) + 1])
    matches = [idx for idx in range(start, end) if int(G.to[idx]) == int(v)]
    if len(matches) != 1:
        return None
    return matches[0]


def _recompute_unique_path_metrics(
    state: AnytimeSearchState,
    path_nodes: Sequence[int],
) -> Tuple[RouteAccumulator, int | None, int | None, int] | None:
    if len(path_nodes) < 2:
        return None
    metrics = RouteAccumulator()
    first_road_id: int | None = None
    last_road_id: int | None = None
    road_changes = 0
    for u, v in zip(path_nodes, path_nodes[1:]):
        edge_idx = _unique_csr_edge_index_for_pair(state.G, int(u), int(v))
        if edge_idx is None:
            return None
        metrics = metrics.plus(RouteAccumulator.from_edge_weights(state.G.w[edge_idx]))
        (
            first_road_id,
            last_road_id,
            road_changes,
        ) = _append_road_id(
            first_road_id,
            last_road_id,
            road_changes,
            _road_id_from_csr_edge(state.G, edge_idx),
        )
    return metrics, first_road_id, last_road_id, road_changes


def _route_accumulators_close(
    left: RouteAccumulator,
    right: RouteAccumulator,
) -> bool:
    return (
        abs(left.length - right.length) <= 1e-6
        and abs(left.elevation - right.elevation) <= 1e-6
        and abs(left.popularity_length - right.popularity_length) <= 1e-6
        and abs(left.street_width_length - right.street_width_length) <= 1e-6
    )


def _bridge_candidate_is_materializable(
    state: AnytimeSearchState,
    candidate: BridgeRefinementCandidate,
) -> bool:
    edge = candidate.edge
    if edge.kind != "inter":
        return False
    if not edge.path_nodes:
        return False
    if edge.path_nodes[0] != edge.src or edge.path_nodes[-1] != edge.dst:
        return False
    if edge.bridge_cell_pair is None or edge.bridge_corridor is None:
        return False
    if edge.bridge_cell_pair != (
        state.trace_cell(edge.src),
        state.trace_cell(edge.dst),
    ):
        return False
    if edge.bridge_corridor != (candidate.crossing_src, candidate.crossing_dst):
        return False
    retained_crossings = state.retained_crossings_by_cell_pair.get(
        edge.bridge_cell_pair,
        [],
    )
    if not any(
        crossing_src == candidate.crossing_src
        and crossing_dst == candidate.crossing_dst
        for crossing_src, crossing_dst, _, _ in retained_crossings
    ):
        return False
    if not _path_contains_segment(
        edge.path_nodes,
        (candidate.crossing_src, candidate.crossing_dst),
    ):
        return False
    recomputed = _recompute_unique_path_metrics(state, edge.path_nodes)
    if recomputed is None:
        return False
    metrics, first_road_id, last_road_id, road_changes = recomputed
    return (
        _route_accumulators_close(metrics, edge.metrics)
        and first_road_id == edge.first_road_id
        and last_road_id == edge.last_road_id
        and road_changes == edge.road_changes
    )


def _best_feasible_bridge_join_selection(
    state: AnytimeSearchState,
    candidate: BridgeRefinementCandidate,
    selected_edges: Sequence[OverlayEdge],
) -> BridgeJoinSelection | None:
    if not candidate.joins_covered_directions:
        return None
    selections: List[BridgeJoinSelection] = []
    fwd_labels = _bounded_representative_labels(
        state,
        state.labels["fwd"].get(candidate.edge.src, []),
    )
    bwd_labels = _bounded_representative_labels(
        state,
        state.labels["bwd"].get(candidate.edge.dst, []),
    )
    for fwd_label in fwd_labels:
        for bwd_label in bwd_labels:
            if _bridge_candidate_has_cell_conflict(
                state,
                candidate,
                fwd_label,
                bwd_label,
            ):
                continue
            metrics = (
                fwd_label.metrics
                .plus(candidate.edge.metrics)
                .plus(bwd_label.metrics)
            )
            if not state.query.constraints.is_feasible(metrics):
                continue
            selections.append(
                BridgeJoinSelection(
                    candidate=candidate,
                    fwd_label=fwd_label,
                    bwd_label=bwd_label,
                    metrics=metrics,
                    score=state.query.constraints.score(metrics),
                    road_changes=_one_edge_join_road_changes(
                        fwd_label,
                        candidate.edge,
                        bwd_label,
                    ),
                )
            )
    if not selections:
        return None
    return min(
        selections,
        key=lambda selection: _bridge_selection_key(
            selection,
            selected_edges,
        ),
    )


def detect_pending_bridge_cell_pairs(state: AnytimeSearchState) -> None:
    state.pending_bridge_cell_pairs.update(_full_bridge_eligible_cell_pairs(state))


def _covered_cells_from_representatives(
    state: AnytimeSearchState,
    direction: Direction,
) -> frozenset[int]:
    return frozenset(
        state.trace_cell(portal)
        for portal, labels in state.labels[direction].items()
        if labels
    )


def detect_bridge_pairs_if_coverage_changed(
    state: AnytimeSearchState,
) -> bool:
    state.audit.bridge_detection_coverage_checks += 1
    start = time.perf_counter()
    current_fwd_cells = _covered_cells_from_representatives(state, "fwd")
    current_bwd_cells = _covered_cells_from_representatives(state, "bwd")
    new_fwd_cells = set(current_fwd_cells) - state.bridge_seen_fwd_cells
    new_bwd_cells = set(current_bwd_cells) - state.bridge_seen_bwd_cells
    if not new_fwd_cells and not new_bwd_cells:
        state.audit.bridge_detection_skipped_unchanged_coverage += 1
        elapsed = time.perf_counter() - start
        state.audit.bridge_detection_total_time_s += elapsed
        state.audit.bridge_incremental_detection_total_time_s += elapsed
        return False

    state.audit.bridge_detection_calls += 1
    state.audit.bridge_incremental_detection_calls += 1
    state.audit.bridge_incremental_new_fwd_cells += len(new_fwd_cells)
    state.audit.bridge_incremental_new_bwd_cells += len(new_bwd_cells)
    fwd_by_cell = _covered_portals_by_cell(state, "fwd")
    bwd_by_cell = _covered_portals_by_cell(state, "bwd")
    discovered: Set[Tuple[int, int]] = set()

    for fwd_cell in sorted(new_fwd_cells):
        for bwd_cell in sorted(state.retained_cell_neighbors.get(fwd_cell, set())):
            state.audit.bridge_incremental_neighbor_lookups += 1
            bwd_portals = bwd_by_cell.get(bwd_cell)
            if not bwd_portals:
                continue
            fwd_portals = fwd_by_cell.get(fwd_cell)
            if not fwd_portals:
                continue
            cell_pair = (fwd_cell, bwd_cell)
            if _bridge_cell_pair_is_eligible(
                state,
                cell_pair,
                fwd_portals,
                bwd_portals,
            ):
                discovered.add(cell_pair)

    for bwd_cell in sorted(new_bwd_cells):
        for fwd_cell in sorted(state.retained_cell_neighbors.get(bwd_cell, set())):
            state.audit.bridge_incremental_neighbor_lookups += 1
            fwd_portals = fwd_by_cell.get(fwd_cell)
            if not fwd_portals:
                continue
            bwd_portals = bwd_by_cell.get(bwd_cell)
            if not bwd_portals:
                continue
            cell_pair = (fwd_cell, bwd_cell)
            if _bridge_cell_pair_is_eligible(
                state,
                cell_pair,
                fwd_portals,
                bwd_portals,
            ):
                discovered.add(cell_pair)

    before_pending = set(state.pending_bridge_cell_pairs)
    state.pending_bridge_cell_pairs.update(discovered)
    state.audit.bridge_incremental_pending_pairs_discovered += len(
        state.pending_bridge_cell_pairs - before_pending
    )

    if state.config.validate_incremental_bridge_detection:
        full_pairs = _full_bridge_eligible_cell_pairs(state, fwd_by_cell, bwd_by_cell)
        missing = full_pairs - discovered - before_pending
        if missing:
            state.audit.bridge_incremental_crosscheck_mismatches += len(missing)
            logger.debug(
                "Incremental bridge detection missed eligible pairs=%s "
                "new_fwd_cells=%s new_bwd_cells=%s",
                sorted(missing),
                sorted(new_fwd_cells),
                sorted(new_bwd_cells),
            )

    state.bridge_seen_fwd_cells.update(new_fwd_cells)
    state.bridge_seen_bwd_cells.update(new_bwd_cells)
    elapsed = time.perf_counter() - start
    state.audit.bridge_detection_total_time_s += elapsed
    state.audit.bridge_incremental_detection_total_time_s += elapsed
    state.last_bridge_detection_fwd_cells = current_fwd_cells
    state.last_bridge_detection_bwd_cells = current_bwd_cells
    return bool(discovered)


def _insert_bridge_selection(
    state: AnytimeSearchState,
    cell_pair: Tuple[int, int],
    selection: BridgeJoinSelection,
) -> bool:
    candidate = selection.candidate
    edge = candidate.edge
    crossing_key = (candidate.crossing_src, candidate.crossing_dst)
    selected_crossings = state.bridge_inserted_crossings_by_cell_pair.setdefault(
        cell_pair,
        set(),
    )
    if crossing_key in selected_crossings or not state.add_overlay_edge(edge):
        return False

    selected_crossings.add(crossing_key)
    state.bridge_representatives_by_cell_pair.setdefault(
        cell_pair,
        [],
    ).append(edge)
    state.audit.bridge_edges_inserted += 1
    logger.debug(
        "Bridge representative cells=%s->%s edge=%s->%s "
        "crossing=%s->%s score=%.4f connector_length=%.1f path_len=%s",
        cell_pair[0],
        cell_pair[1],
        edge.src,
        edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
        selection.score,
        candidate.connector_length,
        len(edge.path_nodes),
    )

    fwd_labels = _bounded_representative_labels(
        state,
        state.labels["fwd"].get(edge.src, []),
    )
    bwd_labels = _bounded_representative_labels(
        state,
        state.labels["bwd"].get(edge.dst, []),
    )
    for fwd_label in fwd_labels:
        for bwd_label in bwd_labels:
            state.audit.bridge_join_attempts += 1
            if _emit_one_edge_join_candidate(
                state,
                fwd_label,
                edge,
                bwd_label,
                trigger_dir="fwd",
                bridge_join_context="immediate_bridge",
            ):
                state.audit.bridge_join_successes += 1
                state.audit.bridge_immediate_join_successes += 1

    if not bwd_labels:
        for fwd_label in fwd_labels:
            if _enqueue_child_label_through_edge(
                state,
                fwd_label,
                edge,
                child_source="bridge",
            ):
                state.audit.bridge_repair_children_generated += 1
    if not fwd_labels:
        for bwd_label in bwd_labels:
            if _enqueue_child_label_through_edge(
                state,
                bwd_label,
                edge,
                child_source="bridge",
            ):
                state.audit.bridge_repair_children_generated += 1
    return True


def _insert_fallback_bridge_candidate(
    state: AnytimeSearchState,
    cell_pair: Tuple[int, int],
    candidate: BridgeRefinementCandidate,
    *,
    selection_role: BridgeFallbackSelectionRole = "primary",
) -> bool:
    state.audit.bridge_fallback_edges_attempted += 1
    edge = candidate.edge
    crossing_key = (candidate.crossing_src, candidate.crossing_dst)
    selected_crossings = state.bridge_inserted_crossings_by_cell_pair.setdefault(
        cell_pair,
        set(),
    )
    if crossing_key in selected_crossings:
        logger.debug(
            "Bridge fallback reject cells=%s->%s src=%s dst=%s reason=crossing_already_represented",
            cell_pair[0],
            cell_pair[1],
            edge.src,
            edge.dst,
        )
        return False
    if not _bridge_candidate_is_materializable(state, candidate):
        logger.debug(
            "Bridge fallback reject cells=%s->%s src=%s dst=%s reason=invalid_reconstructible_path",
            cell_pair[0],
            cell_pair[1],
            edge.src,
            edge.dst,
        )
        return False
    overlay_rejection_reason = _generic_overlay_edge_rejection_reason(state, edge)
    if selection_role == "quality":
        state.audit.complementary_quality_candidate_insert_attempted += 1
    if not state.add_overlay_edge(edge):
        if selection_role == "quality" and overlay_rejection_reason == "pair_capacity":
            state.audit.complementary_quality_candidate_rejected_by_overlay_capacity += 1
        logger.debug(
            "Bridge fallback reject cells=%s->%s src=%s dst=%s "
            "role=%s reason=overlay_insert_rejected overlay_reason=%s",
            cell_pair[0],
            cell_pair[1],
            edge.src,
            edge.dst,
            selection_role,
            overlay_rejection_reason,
        )
        return False
    if selection_role == "quality":
        state.audit.complementary_quality_candidate_inserted += 1

    selected_crossings.add(crossing_key)
    state.bridge_representatives_by_cell_pair.setdefault(cell_pair, []).append(edge)
    state.audit.bridge_edges_inserted += 1
    state.audit.bridge_fallback_edges_inserted += 1

    fwd_labels = _bounded_representative_labels(
        state,
        state.labels["fwd"].get(edge.src, []),
    )
    bwd_labels = _bounded_representative_labels(
        state,
        state.labels["bwd"].get(edge.dst, []),
    )

    for fwd_label in fwd_labels:
        for bwd_label in bwd_labels:
            state.audit.bridge_join_attempts += 1
            if _emit_one_edge_join_candidate(
                state,
                fwd_label,
                edge,
                bwd_label,
                trigger_dir="fwd",
            ):
                state.audit.bridge_join_successes += 1

    for fwd_label in fwd_labels:
        state.audit.bridge_fallback_children_generated += 1
        if _enqueue_child_label_through_edge(
            state,
            fwd_label,
            edge,
            child_source="bridge",
        ):
            state.audit.bridge_fallback_children_accepted += 1
            state.audit.bridge_repair_children_generated += 1

    for bwd_label in bwd_labels:
        state.audit.bridge_fallback_children_generated += 1
        if _enqueue_child_label_through_edge(
            state,
            bwd_label,
            edge,
            child_source="bridge",
        ):
            state.audit.bridge_fallback_children_accepted += 1
            state.audit.bridge_repair_children_generated += 1

    logger.debug(
        "Bridge fallback inserted cells=%s->%s edge=%s->%s crossing=%s->%s "
        "role=%s connector_quality=%s children_attempted=%s children_accepted=%s",
        cell_pair[0],
        cell_pair[1],
        edge.src,
        edge.dst,
        candidate.crossing_src,
        candidate.crossing_dst,
        selection_role,
        _bridge_candidate_quality_summary(candidate),
        len(fwd_labels) + len(bwd_labels),
        state.audit.bridge_fallback_children_accepted,
    )
    return True


def refine_adjacent_coverage_bridges(
    state: AnytimeSearchState,
    *,
    run_detection: bool = True,
) -> List[OverlayEdge]:
    if run_detection:
        detect_pending_bridge_cell_pairs(state)
    limit = max(0, state.config.max_bridge_edges_per_cell_pair)
    if limit == 0 or not state.pending_bridge_cell_pairs:
        return []

    fwd_portals = _covered_portals(state, "fwd")
    bwd_portals = _covered_portals(state, "bwd")
    selected_edges = [
        edge
        for edges in state.bridge_representatives_by_cell_pair.values()
        for edge in edges
    ]
    options_by_pair: Dict[
        Tuple[int, int],
        Dict[Tuple[int, int], BridgeJoinSelection],
    ] = {}
    fallback_options_by_pair: Dict[
        Tuple[int, int],
        Dict[Tuple[int, int], BridgeRefinementCandidate],
    ] = {}
    fallback_quality_options_by_pair: Dict[
        Tuple[int, int],
        Dict[Tuple[int, int], BridgeRefinementCandidate],
    ] = {}

    for cell_pair in sorted(state.pending_bridge_cell_pairs):
        fwd_cell, bwd_cell = cell_pair
        pair_fwd_portals = {
            portal
            for portal in fwd_portals
            if state.trace_cell(portal) == fwd_cell
        }
        pair_bwd_portals = {
            portal
            for portal in bwd_portals
            if state.trace_cell(portal) == bwd_cell
        }
        if not pair_fwd_portals or not pair_bwd_portals:
            continue
        state.bridge_refinement_attempted_cell_pairs.add(cell_pair)
        state.audit.bridge_refinements_attempted += 1
        pair_key = f"{fwd_cell}->{bwd_cell}"
        state.audit.bridge_refinement_cell_pairs[pair_key] = (
            state.audit.bridge_refinement_cell_pairs.get(pair_key, 0) + 1
        )
        candidates = _bridge_candidates_for_direction(
            state,
            src_cell=fwd_cell,
            dst_cell=bwd_cell,
            src_portals=sorted(pair_fwd_portals),
            dst_portals=sorted(pair_bwd_portals),
            forward_portals=pair_fwd_portals,
            backward_portals=pair_bwd_portals,
        )
        state.audit.bridge_candidates_reconstructible += len(candidates)
        selected_crossings = (
            state.bridge_inserted_crossings_by_cell_pair.setdefault(
                cell_pair,
                set(),
            )
        )
        corridor_options: Dict[Tuple[int, int], BridgeJoinSelection] = {}
        fallback_corridor_options: Dict[
            Tuple[int, int],
            BridgeRefinementCandidate,
        ] = {}
        fallback_quality_corridor_options: Dict[
            Tuple[int, int],
            BridgeRefinementCandidate,
        ] = {}
        for candidate in candidates:
            corridor_id = (candidate.crossing_src, candidate.crossing_dst)
            if corridor_id in selected_crossings:
                continue
            selection = _best_feasible_bridge_join_selection(
                state,
                candidate,
                selected_edges,
            )
            if selection is None:
                existing_candidate = fallback_corridor_options.get(corridor_id)
                if existing_candidate is None or _bridge_fallback_candidate_key(
                    candidate,
                    selected_edges,
                ) < _bridge_fallback_candidate_key(
                    existing_candidate,
                    selected_edges,
                ):
                    fallback_corridor_options[corridor_id] = candidate
                existing_quality_candidate = fallback_quality_corridor_options.get(
                    corridor_id
                )
                if (
                    existing_quality_candidate is None
                    or _bridge_fallback_quality_candidate_key(
                        candidate,
                        selected_edges,
                    )
                    < _bridge_fallback_quality_candidate_key(
                        existing_quality_candidate,
                        selected_edges,
                    )
                ):
                    fallback_quality_corridor_options[corridor_id] = candidate
                continue
            state.audit.bridge_immediate_feasible_selections += 1
            existing = corridor_options.get(corridor_id)
            if existing is None or _bridge_selection_key(
                selection,
                selected_edges,
            ) < _bridge_selection_key(
                existing,
                selected_edges,
            ):
                corridor_options[corridor_id] = selection
        if corridor_options:
            options_by_pair[cell_pair] = corridor_options
        elif fallback_corridor_options:
            fallback_options_by_pair[cell_pair] = fallback_corridor_options
            fallback_quality_options_by_pair[cell_pair] = fallback_quality_corridor_options

    inserted_edges: List[OverlayEdge] = []

    # Stage 1: give every currently feasible adjacent cell pair one corridor.
    for cell_pair in sorted(options_by_pair):
        selected = state.bridge_representatives_by_cell_pair.get(cell_pair, [])
        if selected:
            continue
        corridor_options = options_by_pair[cell_pair]
        ordered = sorted(
            corridor_options.items(),
            key=lambda item: _bridge_selection_key(
                item[1],
                selected_edges,
            ),
        )
        for corridor_id, selection in ordered:
            if not _insert_bridge_selection(state, cell_pair, selection):
                corridor_options.pop(corridor_id, None)
                continue
            inserted_edges.append(selection.candidate.edge)
            selected_edges.append(selection.candidate.edge)
            corridor_options.pop(corridor_id, None)
            break

    # Stage 3: only now admit additional distinct corridors under the pair cap.
    while True:
        pool: List[
            Tuple[
                Tuple[int, int],
                Tuple[int, int],
                BridgeJoinSelection,
            ]
        ] = []
        for cell_pair, corridor_options in options_by_pair.items():
            if (
                len(state.bridge_representatives_by_cell_pair.get(cell_pair, []))
                >= limit
            ):
                continue
            for corridor_id, selection in corridor_options.items():
                pool.append((cell_pair, corridor_id, selection))
        if not pool:
            break
        cell_pair, corridor_id, selection = min(
            pool,
            key=lambda item: _bridge_selection_key(
                item[2],
                selected_edges,
            ),
        )
        options_by_pair[cell_pair].pop(corridor_id, None)
        if not _insert_bridge_selection(state, cell_pair, selection):
            continue
        inserted_edges.append(selection.candidate.edge)
        selected_edges.append(selection.candidate.edge)

    for cell_pair in sorted(fallback_options_by_pair):
        selected = state.bridge_representatives_by_cell_pair.get(cell_pair, [])
        remaining_slots = max(0, limit - len(selected))
        if remaining_slots <= 0:
            continue
        ordered_fallback = _select_complementary_bridge_fallback_candidates(
            state,
            cell_pair,
            fallback_options_by_pair[cell_pair],
            fallback_quality_options_by_pair.get(cell_pair, {}),
            selected_edges,
            remaining_slots,
        )
        for corridor_id, candidate, selection_role in ordered_fallback:
            if (
                len(state.bridge_representatives_by_cell_pair.get(cell_pair, []))
                >= limit
            ):
                break
            if not _insert_fallback_bridge_candidate(
                state,
                cell_pair,
                candidate,
                selection_role=selection_role,
            ):
                continue
            inserted_edges.append(candidate.edge)
            selected_edges.append(candidate.edge)

    state.pending_bridge_cell_pairs.clear()
    return inserted_edges


def _choose_direction(state: AnytimeSearchState) -> Direction | None:
    fwd_heap = state.frontier["fwd"]
    bwd_heap = state.frontier["bwd"]

    if not fwd_heap and not bwd_heap:
        return None
    if not fwd_heap:
        return "bwd"
    if not bwd_heap:
        return "fwd"

    if state.turn % 2 == 0:
        return "fwd"
    return "bwd"


def advance_overlay_and_learn_shortcuts(
    state: AnytimeSearchState,
    *,
    step_budget: int | None = None,
) -> int:
    budget = state.config.advance_round_budget if step_budget is None else step_budget
    accepted = 0

    for _ in range(budget):
        if time.perf_counter() >= state.deadline:
            break

        direction = _choose_direction(state)
        log_frontier_state(
            state,
            action="choose_direction",
            direction=direction,
        )
        logger.debug(
            "Advance turn=%s frontier_fwd=%s frontier_bwd=%s chosen=%s",
            state.turn,
            len(state.frontier["fwd"]),
            len(state.frontier["bwd"]),
            direction,
        )
        if direction is None:
            break

        heap = state.frontier[direction]
        if not heap:
            break

        _, _, _, label = heapq.heappop(heap)
        _increment_direction_count(state.audit.frontier_pops, label.direction)
        log_label_pop(state, label)
        log_frontier_state(
            state,
            action="after_pop",
            direction=label.direction,
        )
        backward_trace = (
            _make_backward_pop_trace(state, label)
            if label.direction == "bwd"
            else None
        )
        logger.debug(
            "Advance pop portal=%s dir=%s priority=%.4f route=%s",
            label.portal,
            label.direction,
            label.priority,
            _fmt_route_vector(label.metrics),
        )
        if label.direction == "bwd":
            logger.debug(
                "Backward label pop portal=%s priority=%.4f route=%s frontier_bwd_after_pop=%s",
                label.portal,
                label.priority,
                _fmt_route_vector(label.metrics),
                len(state.frontier["bwd"]),
            )
        _emit_join(state, label)
        before_backward_length_rejects = state.audit.backward_rejected_length
        before_backward_elevation_rejects = state.audit.backward_rejected_elevation
        before_backward_capacity_rejects = (
            state.audit.backward_rejected_representative_capacity
        )
        kept_representative = _keep_representative(state, label)
        if backward_trace is not None:
            if state.audit.backward_rejected_length > before_backward_length_rejects:
                _record_trace_child_rejection(backward_trace, "length")
            if state.audit.backward_rejected_elevation > before_backward_elevation_rejects:
                _record_trace_child_rejection(backward_trace, "elevation")
            if (
                state.audit.backward_rejected_representative_capacity
                > before_backward_capacity_rejects
            ):
                _record_trace_child_rejection(
                    backward_trace,
                    "representative_capacity",
                )
        if not kept_representative:
            if backward_trace is not None:
                _log_backward_pop_trace(backward_trace)
            state.turn += 1
            continue

        new_edges: List[OverlayEdge] = []
        repair_edges: List[OverlayEdge] = []
        directional_repair_edges: List[OverlayEdge] = []
        forward_directional_repair_edges: List[OverlayEdge] = []
        detect_bridge_pairs_if_coverage_changed(state)
        overlay_degree = _directional_overlay_degree(state, label)
        if overlay_degree <= 1:
            if backward_trace is not None:
                backward_trace.local_triggered = True
                backward_trace.local_cell_id = state.partition.get(label.portal)
            logger.debug(
                "Local trigger activation portal=%s dir=%s directional_degree=%s",
                label.portal,
                label.direction,
                overlay_degree,
            )
            log_local_discovery(
                state,
                label,
                action="trigger",
            )
            new_edges = grow_portal_shortcuts_in_cell(
                state,
                label.portal,
                label.direction,
                trace=backward_trace,
            )
            local_edges_to_enqueue = list(new_edges)
            log_local_discovery(
                state,
                label,
                action="batch_complete",
                discovered_edges=len(new_edges),
            )
            while (
                _directional_overlay_degree(state, label) <= 1
                and _local_engine_has_pending_work(state, label)
                and time.perf_counter() < state.deadline
            ):
                logger.debug(
                    "Local trigger resume portal=%s dir=%s directional_degree=%s",
                    label.portal,
                    label.direction,
                    _directional_overlay_degree(state, label),
                )
                log_local_discovery(
                    state,
                    label,
                    action="resume",
                )
                if backward_trace is not None:
                    backward_trace.local_batches_resumed += 1
                new_edges = grow_portal_shortcuts_in_cell(
                    state,
                    label.portal,
                    label.direction,
                    trace=backward_trace,
                )
                local_edges_to_enqueue.extend(new_edges)
                log_local_discovery(
                    state,
                    label,
                    action="batch_complete",
                    discovered_edges=len(new_edges),
                )
            ordered_local_edges = sorted(
                local_edges_to_enqueue,
                key=lambda edge: _local_shortcut_candidate_order_key(label, edge),
            )
            if [
                (edge.src, edge.dst, edge.path_nodes)
                for edge in local_edges_to_enqueue
            ] != [
                (edge.src, edge.dst, edge.path_nodes)
                for edge in ordered_local_edges
            ]:
                state.audit.local_shortcut_ordering_changed_by_road_continuity += 1
            for edge in ordered_local_edges:
                _enqueue_child_label_through_edge(
                    state,
                    label,
                    edge,
                    trace=backward_trace,
                )
        if label.direction == "fwd":
            forward_directional_repair_edges = refine_forward_directional_portals(
                state,
                label,
            )
            for edge in forward_directional_repair_edges:
                if _enqueue_child_label_through_edge(
                    state,
                    label,
                    edge,
                    trace=backward_trace,
                    child_source="directional_repair",
                ):
                    state.audit.forward_directional_repair_children_generated += 1
        if label.direction == "bwd":
            directional_repair_edges = refine_backward_directional_portals(
                state,
                label,
            )
            for edge in directional_repair_edges:
                if _enqueue_child_label_through_edge(
                    state,
                    label,
                    edge,
                    trace=backward_trace,
                    child_source="directional_repair",
                ):
                    state.audit.backward_directional_repair_children_generated += 1

            local_engine = state.local_engines.get(
                (state.trace_cell(label.portal), label.portal, label.direction)
            )
            local_exhausted = local_engine is not None and local_engine.exhausted
            local_zero_shortcuts = (
                backward_trace is not None
                and backward_trace.local_triggered
                and backward_trace.local_shortcuts_discovered == 0
            )
            crossing_predecessors = [
                pred
                for pred, _, _ in state.reverse_adj.get(label.portal, [])
                if state.trace_cell(pred) != state.trace_cell(label.portal)
            ]
            if (
                _usable_backward_edge_count(state, label) == 0
                and (local_exhausted or local_zero_shortcuts)
                and crossing_predecessors
            ):
                repair_edges = repair_backward_dead_portal(state, label)
                for edge in repair_edges:
                    if _enqueue_child_label_through_edge(
                        state,
                        label,
                        edge,
                        trace=backward_trace,
                        child_source="repair",
                    ):
                        state.audit.backward_dead_portal_repair_children_generated += 1
        existing_children_generated = _expand_representative_label(
            state,
            label,
            skip_edge_ids={
                id(edge)
                for edge in [
                    *new_edges,
                    *repair_edges,
                    *directional_repair_edges,
                    *forward_directional_repair_edges,
                ]
            },
            trace=backward_trace,
        )
        if backward_trace is not None:
            backward_trace.children_from_existing_overlay += existing_children_generated
            _log_backward_pop_trace(backward_trace)

        accepted += 1
        state.turn += 1

    log_frontier_state(state, action="advance_end")
    logger.debug("Advance end accepted=%s", accepted)
    return accepted


def refine_portal_set(state: AnytimeSearchState) -> None:
    # Placeholder for targeted portal activation in later iterations.
    return None


def deepen_local_knowledge(state: AnytimeSearchState) -> None:
    # Placeholder for later anytime local-engine scheduling.
    return None


def improve_best_routes(state: AnytimeSearchState) -> None:
    # Placeholder for later archive-driven improvement rounds.
    return None


def repair_important_portal_pairs_if_needed(state: AnytimeSearchState) -> None:
    # Placeholder for later query-driven boundary repair.
    return None


def _covered_portals(state: AnytimeSearchState, direction: Direction) -> Set[int]:
    return {
        portal
        for portal, labels in state.labels[direction].items()
        if labels
    }


def _portal_coverage_diagnostics(
    state: AnytimeSearchState,
) -> Tuple[Set[int], Set[int], Set[int], List[int] | str]:
    fwd_portals = _covered_portals(state, "fwd")
    bwd_portals = _covered_portals(state, "bwd")
    shared_portals = fwd_portals & bwd_portals
    if len(shared_portals) <= 20:
        shared_report: List[int] | str = sorted(shared_portals)
    else:
        shared_report = f"{len(shared_portals)} shared portals; list omitted"
    return fwd_portals, bwd_portals, shared_portals, shared_report


def debug_dump_search_state(
    state: AnytimeSearchState,
    query: SparsePortalQuery,
    max_portals: int = 20,
) -> Dict[str, object]:
    limit = max(0, max_portals)
    fwd_portals, bwd_portals, shared_portals, _ = (
        _portal_coverage_diagnostics(state)
    )
    overlay_nodes = (
        set(state.overlay.out_edges)
        | set(state.overlay.in_edges)
        | state.active_portals
    )
    top_cells = sorted(
        state.touched_cells.items(),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    top_rejections = sorted(
        state.audit.backward_rejection_reason_count.items(),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    summary: Dict[str, object] = {
        "query": {"source": query.source, "target": query.target},
        "overlay_nodes": len(overlay_nodes),
        "overlay_edges": _overlay_edge_count(state),
        "queue_sizes": {
            "fwd": len(state.frontier["fwd"]),
            "bwd": len(state.frontier["bwd"]),
        },
        "representative_portals": {
            "fwd": len(fwd_portals),
            "bwd": len(bwd_portals),
        },
        "shared_representative_portals": sorted(shared_portals)[:limit],
        "archive_size": len(state.archive.entries),
        "top_cells_touched": top_cells,
        "top_rejection_reasons": top_rejections,
        "audit": state.audit.as_dict(),
    }
    logger.info("search_state_dump %s", summary)
    return summary


def _one_edge_join_coverage_diagnostics(
    state: AnytimeSearchState,
    fwd_portals: Set[int],
    bwd_portals: Set[int],
) -> Tuple[int, int]:
    fwd_with_one_edge_to_bwd = sum(
        1
        for portal in fwd_portals
        if any(edge.dst in bwd_portals for edge in state.overlay.out_edges.get(portal, []))
    )
    bwd_with_one_edge_from_fwd = sum(
        1
        for portal in bwd_portals
        if any(edge.src in fwd_portals for edge in state.overlay.in_edges.get(portal, []))
    )
    return fwd_with_one_edge_to_bwd, bwd_with_one_edge_from_fwd


def _target_side_cell_diagnostics(state: AnytimeSearchState) -> List[Dict[str, object]]:
    target_cell = state.trace_cell(state.query.target)
    reached_cells: Set[int] = {target_cell}
    for trace in state.audit.backward_pop_traces:
        reached_cells.add(trace.cell_id)
        reached_cells.update(trace.visited_cells)

    rows: List[Dict[str, object]] = []
    for cell_id in sorted(reached_cells):
        active_portals = sorted(
            portal
            for portal in state.active_portals
            if state.trace_cell(portal) == cell_id
        )
        zero_in_degree = [
            portal
            for portal in active_portals
            if not state.overlay.in_edges.get(portal)
        ]
        zero_out_degree = [
            portal
            for portal in active_portals
            if not state.overlay.out_edges.get(portal)
        ]
        zero_usable_backward_degree = [
            portal
            for portal in active_portals
            if _usable_backward_degree_for_cell_entry(state, portal, cell_id) == 0
        ]
        rows.append(
            {
                "cell": cell_id,
                "active_portal_count": len(active_portals),
                "active_portals": active_portals,
                "local_shortcuts_discovered": (
                    state.audit.local_shortcuts_discovered_by_cell.get(cell_id, 0)
                ),
                "zero_in_degree_portals": zero_in_degree,
                "zero_out_degree_portals": zero_out_degree,
                "zero_usable_backward_degree_portals": (
                    zero_usable_backward_degree
                ),
            }
        )
    return rows


def _backward_trace_summary(state: AnytimeSearchState) -> Dict[str, int]:
    traces = state.audit.backward_pop_traces
    return {
        "backward_pop_trace_length": len(traces),
        "backward_pops_with_zero_in_edges": sum(
            1 for trace in traces if trace.in_edges_count == 0
        ),
        "backward_pops_with_zero_usable_edges": sum(
            1 for trace in traces if trace.usable_backward_edges == 0
        ),
        "backward_pops_triggering_local_engine": sum(
            1 for trace in traces if trace.local_triggered
        ),
        "backward_local_triggers_with_zero_shortcuts": sum(
            1
            for trace in traces
            if trace.local_triggered and trace.local_shortcuts_discovered == 0
        ),
        "backward_shortcuts_discovered_total": sum(
            trace.local_shortcuts_discovered for trace in traces
        ),
        "backward_children_from_local_shortcuts_total": sum(
            trace.children_from_local_shortcuts for trace in traces
        ),
        "backward_children_from_existing_overlay_total": sum(
            trace.children_from_existing_overlay for trace in traces
        ),
    }


def anytime_sparse_portal_search(
    G: CompactDiGraph,
    partition: Dict[int, int],
    boundary_nodes: Set[int],
    kept_nodes: Iterable[int],
    query: SparsePortalQuery,
    config: SparsePortalConfig | None = None,
) -> AnytimeSearchState:
    cfg = config if config is not None else SparsePortalConfig()
    kept_node_set = set(kept_nodes)
    kept_node_set.add(query.source)
    kept_node_set.add(query.target)

    logger.debug(
        "Search start source=%s target=%s time_budget_s=%.3f archive_size=%s",
        query.source,
        query.target,
        query.time_budget_s,
        query.archive_size,
    )
    logger.debug(
        "Search input kept_nodes=%s boundary_nodes=%s",
        len(kept_node_set),
        len(boundary_nodes),
    )

    active_portals = _select_active_portals(
        G=G,
        boundary_nodes=boundary_nodes,
        partition=partition,
        kept_nodes=kept_node_set,
        source=query.source,
        target=query.target,
        max_per_cell=cfg.max_active_portals_per_cell,
    )
    logger.debug("Search active_portals=%s", len(active_portals))

    reverse_adj = _build_reverse_adjacency(G, kept_node_set)
    crossing_index_start = time.perf_counter()
    (
        retained_cell_neighbors,
        retained_crossings_by_cell_pair,
        retained_directed_inter_cell_crossings,
        retained_undirected_cell_pair_count,
    ) = _build_retained_cell_crossing_index(G, partition, kept_node_set)
    retained_cell_crossing_index_time_s = time.perf_counter() - crossing_index_start
    logger.debug(
        "Retained cell crossing index time_s=%.6f directed_crossings=%s "
        "directed_cell_pairs=%s undirected_cell_pairs=%s neighbor_entries=%s",
        retained_cell_crossing_index_time_s,
        retained_directed_inter_cell_crossings,
        len(retained_crossings_by_cell_pair),
        retained_undirected_cell_pair_count,
        sum(len(neighbors) for neighbors in retained_cell_neighbors.values()),
    )
    dijkstra_source_start = time.perf_counter()
    min_len_from_source = _dijkstra_forward_retained_length(
        G,
        kept_node_set,
        query.source,
    )
    dijkstra_source_time_s = time.perf_counter() - dijkstra_source_start
    dijkstra_target_start = time.perf_counter()
    min_len_to_target = _dijkstra_reverse_retained_length(
        reverse_adj,
        G.n_nodes,
        kept_node_set,
        query.target,
    )
    dijkstra_target_time_s = time.perf_counter() - dijkstra_target_start
    dijkstra_source_finite_count = int(np.isfinite(min_len_from_source).sum())
    dijkstra_target_finite_count = int(np.isfinite(min_len_to_target).sum())
    s_to_t = float(min_len_from_source[query.target])
    t_from_s = float(min_len_to_target[query.source])
    logger.debug(
        "Retained length lower-bound Dijkstra source_time_s=%.6f "
        "target_reverse_time_s=%.6f finite_from_source=%s finite_to_target=%s "
        "min_len_from_source[target]=%s min_len_to_target[source]=%s "
        "agree=%s",
        dijkstra_source_time_s,
        dijkstra_target_time_s,
        dijkstra_source_finite_count,
        dijkstra_target_finite_count,
        f"{s_to_t:.3f}" if np.isfinite(s_to_t) else "inf",
        f"{t_from_s:.3f}" if np.isfinite(t_from_s) else "inf",
        bool(
            np.isfinite(s_to_t)
            and np.isfinite(t_from_s)
            and abs(s_to_t - t_from_s) <= COMPLETION_LENGTH_LB_TOLERANCE_M
        ),
    )

    state = AnytimeSearchState(
        G=G,
        query=query,
        config=cfg,
        partition=partition,
        kept_nodes=kept_node_set,
        active_portals=active_portals,
        nodes_by_cell=_build_nodes_by_cell(partition, kept_node_set),
        reverse_adj=reverse_adj,
        retained_cell_neighbors=retained_cell_neighbors,
        retained_crossings_by_cell_pair=retained_crossings_by_cell_pair,
        retained_cell_crossing_index_time_s=retained_cell_crossing_index_time_s,
        retained_directed_inter_cell_crossings=(
            retained_directed_inter_cell_crossings
        ),
        retained_directed_cell_pair_count=len(retained_crossings_by_cell_pair),
        retained_undirected_cell_pair_count=retained_undirected_cell_pair_count,
        overlay=OverlayGraph(),
        archive=SolutionArchive(max_size=query.archive_size),
        labels={"fwd": {}, "bwd": {}},
        frontier={"fwd": [], "bwd": []},
        local_engines={},
        deadline=time.perf_counter() + query.time_budget_s,
        min_len_from_source=min_len_from_source,
        min_len_to_target=min_len_to_target,
        dijkstra_source_time_s=dijkstra_source_time_s,
        dijkstra_target_time_s=dijkstra_target_time_s,
        dijkstra_source_finite_count=dijkstra_source_finite_count,
        dijkstra_target_finite_count=dijkstra_target_finite_count,
    )

    _build_initial_overlay(state)
    _record_cell93_static_diagnostics(state)
    log_search_summary(state, phase="initialized")
    logger.debug("Search initial_overlay_out_edges=%s", _overlay_edge_count(state))
    source_label = _make_root_label(state, query.source, "fwd")
    target_label = _make_root_label(state, query.target, "bwd")
    target_in_edges = state.overlay.in_edges.get(query.target, [])
    target_usable_in_edges = sum(
        1
        for edge in target_in_edges
        if _is_backward_edge_usable_for_label(state, target_label, edge)
    )
    logger.debug(
        "Target portal id=%s kept=%s active=%s cell=%s in_degree=%s out_degree=%s usable_in_edges=%s",
        query.target,
        query.target in state.kept_nodes,
        query.target in state.active_portals,
        state.partition.get(query.target),
        len(target_in_edges),
        len(state.overlay.out_edges.get(query.target, [])),
        target_usable_in_edges,
    )
    state.enqueue(source_label)
    state.enqueue(target_label)
    log_frontier_state(state, action="roots_enqueued")
    logger.debug(
        "Initial frontier sizes after roots fwd=%s bwd=%s source=%s target=%s",
        len(state.frontier["fwd"]),
        len(state.frontier["bwd"]),
        query.source,
        query.target,
    )

    while time.perf_counter() < state.deadline:
        progressed = advance_overlay_and_learn_shortcuts(state)
        if time.perf_counter() < state.deadline:
            detect_bridge_pairs_if_coverage_changed(state)
            if state.pending_bridge_cell_pairs:
                refine_adjacent_coverage_bridges(state, run_detection=False)
        if progressed == 0 and not state.frontier["fwd"] and not state.frontier["bwd"]:
            detect_bridge_pairs_if_coverage_changed(state)
            if state.pending_bridge_cell_pairs:
                refine_adjacent_coverage_bridges(state, run_detection=False)
            if state.frontier["fwd"] or state.frontier["bwd"]:
                continue
            break
        if not state.archive.entries:
            refine_portal_set(state)
            deepen_local_knowledge(state)
        else:
            improve_best_routes(state)
            repair_important_portal_pairs_if_needed(state)

    log_search_summary(state, phase="complete")
    logger.debug(
        "Search end archive_size=%s active_portals=%s overlay_out_edges=%s local_engines=%s",
        len(state.archive.entries),
        len(state.active_portals),
        _overlay_edge_count(state),
        len(state.local_engines),
    )
    (
        fwd_covered_portals,
        bwd_covered_portals,
        shared_covered_portals,
        shared_portals_report,
    ) = _portal_coverage_diagnostics(state)
    (
        fwd_one_edge_covered,
        bwd_one_edge_covered,
    ) = _one_edge_join_coverage_diagnostics(
        state,
        fwd_covered_portals,
        bwd_covered_portals,
    )
    logger.debug(
        "Bidirectional coverage diagnostics fwd_representative_portals=%s "
        "bwd_representative_portals=%s both_direction_representative_portals=%s "
        "shared_portals=%s one_edge_fwd_portals_to_bwd=%s "
        "one_edge_bwd_portals_from_fwd=%s",
        len(fwd_covered_portals),
        len(bwd_covered_portals),
        len(shared_covered_portals),
        shared_portals_report,
        fwd_one_edge_covered,
        bwd_one_edge_covered,
    )
    logger.debug(
        "Queue starvation diagnostics total_forward_pops=%s total_backward_pops=%s "
        "total_forward_children_generated=%s total_backward_children_generated=%s "
        "total_forward_children_accepted_as_representatives=%s "
        "total_backward_children_accepted_as_representatives=%s "
        "final_forward_queue_size=%s final_backward_queue_size=%s",
        state.audit.frontier_pops.get("fwd", 0),
        state.audit.frontier_pops.get("bwd", 0),
        state.audit.generated_child_labels_by_direction.get("fwd", 0),
        state.audit.generated_child_labels_by_direction.get("bwd", 0),
        state.audit.representative_child_accept_count.get("fwd", 0),
        state.audit.representative_child_accept_count.get("bwd", 0),
        len(state.frontier["fwd"]),
        len(state.frontier["bwd"]),
    )
    logger.debug(
        "Terminal anti-leak diagnostics leak_generated_by_direction=%s "
        "leak_rejected_by_direction=%s "
        "leak_with_same_portal_opposite_by_direction=%s "
        "leak_with_one_edge_opposite_by_direction=%s",
        dict(sorted(state.audit.terminal_leak_labels_generated_by_direction.items())),
        dict(sorted(state.audit.terminal_leak_labels_rejected_by_direction.items())),
        dict(sorted(state.audit.terminal_leak_same_portal_opposite_by_direction.items())),
        dict(sorted(state.audit.terminal_leak_one_edge_opposite_by_direction.items())),
    )
    logger.debug(
        "Backward starvation diagnostics top_rejection_reasons=%s "
        "rejected_length=%s rejected_elevation=%s rejected_visited_cells=%s "
        "rejected_terminal_anti_leak=%s rejected_representative_capacity=%s",
        _top_str_count_items(state.audit.backward_rejection_reason_count),
        state.audit.backward_rejected_length,
        state.audit.backward_rejected_elevation,
        state.audit.backward_rejected_visited_cells,
        state.audit.backward_rejected_terminal_anti_leak,
        state.audit.backward_rejected_representative_capacity,
    )
    logger.debug(
        "Backward expansion starvation summary %s",
        _backward_trace_summary(state),
    )
    logger.debug(
        "Backward dead portal repair counters attempted=%s inserted=%s "
        "children_generated=%s repaired_portals_by_cell=%s",
        state.audit.backward_dead_portal_repairs_attempted,
        state.audit.backward_dead_portal_repairs_inserted,
        state.audit.backward_dead_portal_repair_children_generated,
        dict(sorted(state.audit.repaired_portals_by_cell.items())),
    )
    logger.debug(
        "Backward cul-de-sac cell diagnostics "
        "backward_children_entering_dead_cell=%s "
        "backward_children_entering_cell_with_only_return_to_previous_cell=%s "
        "backward_dead_cells_by_id=%s cell93_entries=%s",
        state.audit.backward_children_entering_dead_cell,
        (
            state.audit
            .backward_children_entering_cell_with_only_return_to_previous_cell
        ),
        dict(sorted(state.audit.backward_dead_cells_by_id.items())),
        state.audit.backward_cell93_entry_diagnostics,
    )
    logger.debug(
        "Backward cul-de-sac prune counters children_pruned=%s "
        "cells_pruned_by_id=%s",
        state.audit.backward_culdesac_children_pruned,
        dict(sorted(state.audit.backward_culdesac_cells_pruned_by_id.items())),
    )
    logger.debug(
        "Backward directional portal refinement counters attempted=%s "
        "inserted=%s children_generated=%s candidates_seen=%s "
        "candidates_unvisited=%s cells=%s",
        state.audit.backward_directional_portal_refinements_attempted,
        state.audit.backward_directional_portal_refinements_inserted,
        state.audit.backward_directional_repair_children_generated,
        state.audit.backward_directional_repair_candidates_seen,
        state.audit.backward_directional_repair_candidates_unvisited,
        dict(sorted(state.audit.backward_directional_repair_cells.items())),
    )
    logger.debug(
        "Forward directional portal refinement counters attempted=%s "
        "inserted=%s children_generated=%s candidates_seen=%s "
        "candidates_unvisited=%s cells=%s",
        state.audit.forward_directional_portal_refinements_attempted,
        state.audit.forward_directional_portal_refinements_inserted,
        state.audit.forward_directional_repair_children_generated,
        state.audit.forward_directional_repair_candidates_seen,
        state.audit.forward_directional_repair_candidates_unvisited,
        dict(sorted(state.audit.forward_directional_repair_cells.items())),
    )
    logger.debug(
        "Bridge refinement counters attempted=%s edges_inserted=%s "
        "join_attempts=%s join_successes=%s cell_pairs=%s "
        "children_generated=%s complementary_sets=%s "
        "quality_distinct=%s quality_attempted=%s quality_inserted=%s "
        "quality_rejected_overlay_capacity=%s",
        state.audit.bridge_refinements_attempted,
        state.audit.bridge_edges_inserted,
        state.audit.bridge_join_attempts,
        state.audit.bridge_join_successes,
        dict(sorted(state.audit.bridge_refinement_cell_pairs.items())),
        state.audit.bridge_repair_children_generated,
        state.audit.complementary_connector_sets_considered,
        state.audit.complementary_quality_candidate_distinct,
        state.audit.complementary_quality_candidate_insert_attempted,
        state.audit.complementary_quality_candidate_inserted,
        state.audit.complementary_quality_candidate_rejected_by_overlay_capacity,
    )
    logger.debug(
        "Archive diversity counters rejected_exact=%s rejected_high_overlap=%s "
        "replaced_near_duplicate=%s routes_by_bridge_pair=%s "
        "routes_by_bridge_corridor=%s",
        state.audit.archive_rejected_exact_duplicate,
        state.audit.archive_rejected_high_overlap,
        state.audit.archive_replaced_near_duplicate,
        dict(sorted(state.audit.archive_routes_by_bridge_pair.items())),
        dict(sorted(state.audit.archive_routes_by_bridge_corridor.items())),
    )
    logger.debug(
        "Target-side backward coverage cells %s",
        _target_side_cell_diagnostics(state),
    )
    logger.debug(
        "Representative audit summary accepted=%s visited_cells_size_histogram=%s "
        "accept_reasons=%s cell_path_length_histogram=%s "
        "source_cell_far_from_source=%s target_cell_far_from_target=%s",
        dict(sorted(state.audit.representative_accept_count.items())),
        {
            direction: dict(sorted(histogram.items()))
            for direction, histogram
            in sorted(state.audit.representative_visited_cells_size_histogram.items())
        },
        dict(sorted(state.audit.representative_accept_reason_count.items())),
        {
            direction: dict(sorted(histogram.items()))
            for direction, histogram
            in sorted(state.audit.representative_cell_path_length_histogram.items())
        },
        dict(sorted(state.audit.representative_source_cell_far_from_source.items())),
        dict(sorted(state.audit.representative_target_cell_far_from_target.items())),
    )
    logger.debug(
        "One-edge join unexpected shared cells top=%s all=%s",
        _top_count_items(state.audit.one_edge_join_unexpected_shared_cells),
        dict(sorted(state.audit.one_edge_join_unexpected_shared_cells.items())),
    )
    logger.debug(
        "Local shortcuts discovered by cell=%s",
        dict(sorted(state.audit.local_shortcuts_discovered_by_cell.items())),
    )
    _log_overlay_cell_diagnostics(state)
    logger.debug(
        "Search audit counters feasibility_checked_on_combined_accumulator=%s "
        "rejected_length=%s rejected_elevation=%s rejected_avg_popularity=%s "
        "rejected_avg_width=%s rejected_by_score_prune=%s "
        "terminal_completion_attempts=%s "
        "terminal_completion_successes=%s "
        "terminal_leak_rejected_after_failed_completion=%s "
        "join_attempts_before_terminal_rejection=%s "
        "join_successes_before_terminal_rejection=%s "
        "generated_child_labels_total=%s "
        "generated_child_labels_with_same_portal_opposite=%s "
        "generated_child_labels_with_one_edge_opposite=%s "
        "terminal_leak_child_with_same_portal_opposite=%s "
        "terminal_leak_child_with_one_edge_opposite=%s "
        "rejected_backward_enter_source_cell=%s "
        "rejected_forward_enter_target_cell=%s "
        "same_portal_join_attempts=%s same_portal_join_successes=%s "
        "one_edge_join_attempts=%s one_edge_join_successes=%s "
        "one_edge_join_rejected_cell_conflict=%s "
        "one_edge_join_rejected_infeasible=%s "
        "one_edge_join_rejected_reconstruction=%s "
        "one_edge_join_shared_cells_count_histogram=%s "
        "one_edge_join_shared_only_p_cell=%s "
        "one_edge_join_shared_only_q_cell=%s "
        "one_edge_join_shared_only_edge_cells=%s "
        "one_edge_join_shared_multiple_cells=%s "
        "second_cell_visits_allowed=%s "
        "second_cell_visits_allowed_by_direction=%s "
        "rejected_third_cell_visit=%s "
        "rejected_third_cell_visit_by_direction=%s",
        state.audit.feasibility_checked_on_combined_accumulator,
        state.audit.rejected_length,
        state.audit.rejected_elevation,
        state.audit.rejected_avg_popularity,
        state.audit.rejected_avg_width,
        state.audit.rejected_by_score_prune,
        state.audit.terminal_completion_attempts,
        state.audit.terminal_completion_successes,
        state.audit.terminal_leak_rejected_after_failed_completion,
        state.audit.join_attempts_before_terminal_rejection,
        state.audit.join_successes_before_terminal_rejection,
        state.audit.generated_child_labels_total,
        state.audit.generated_child_labels_with_same_portal_opposite,
        state.audit.generated_child_labels_with_one_edge_opposite,
        state.audit.terminal_leak_child_with_same_portal_opposite,
        state.audit.terminal_leak_child_with_one_edge_opposite,
        state.audit.rejected_backward_enter_source_cell,
        state.audit.rejected_forward_enter_target_cell,
        state.audit.same_portal_join_attempts,
        state.audit.same_portal_join_successes,
        state.audit.one_edge_join_attempts,
        state.audit.one_edge_join_successes,
        state.audit.one_edge_join_rejected_cell_conflict,
        state.audit.one_edge_join_rejected_infeasible,
        state.audit.one_edge_join_rejected_reconstruction,
        dict(sorted(state.audit.one_edge_join_shared_cells_count_histogram.items())),
        state.audit.one_edge_join_shared_only_p_cell,
        state.audit.one_edge_join_shared_only_q_cell,
        state.audit.one_edge_join_shared_only_edge_cells,
        state.audit.one_edge_join_shared_multiple_cells,
        state.audit.second_cell_visits_allowed,
        dict(sorted(state.audit.second_cell_visits_allowed_by_direction.items())),
        state.audit.rejected_third_cell_visit,
        dict(sorted(state.audit.rejected_third_cell_visit_by_direction.items())),
    )
    archive_road_changes = [entry.road_changes for entry in state.archive.entries]
    archive_avg_road_changes = (
        sum(archive_road_changes) / len(archive_road_changes)
        if archive_road_changes
        else 0.0
    )
    logger.debug(
        "Archive road continuity road_changes=%s average_road_changes=%.2f",
        archive_road_changes,
        archive_avg_road_changes,
    )
    for idx, entry in enumerate(state.archive.entries, start=1):
        logger.debug(
            "Archive entry idx=%s join_portal=%s score=%.4f route=%s "
            "path_len=%s road_changes=%s",
            idx,
            entry.join_portal,
            entry.score,
            _fmt_route_vector(entry.metrics),
            len(entry.path_nodes),
            entry.road_changes,
        )

    return state
