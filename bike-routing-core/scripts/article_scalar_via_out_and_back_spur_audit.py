from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_feasibility_experiment as via
import article_scalar_via_revisit_structure_audit as revisit


DEFAULT_CANDIDATE_SOURCE = "tmp_scalar_via_recovered_non_elementary_walks.json"
DEFAULT_OUTPUT_JSON = "tmp_scalar_via_out_and_back_spur_audit.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_out_and_back_spur_audit.csv"
TARGET_QUERY_IDS = {
    "holdout_anchor_south_north_09_multi_tight",
    "holdout_anchor_south_north_28_quality_conflict",
    "paris_bures_02_multi_tight",
}


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


def _road_id(G: Any, edge_id: int) -> int | None:
    road_ids = getattr(G, "road_id", None)
    if road_ids is None:
        return None
    edge_id = int(edge_id)
    if edge_id < 0 or edge_id >= len(road_ids):
        return None
    road_id = int(road_ids[edge_id])
    return None if road_id < 0 else road_id


def _positions_by_vertex(nodes: Sequence[int]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        positions.setdefault(int(node), []).append(index)
    return positions


def _candidate_spans(
    G: Any,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for anchor, positions in sorted(_positions_by_vertex(nodes).items()):
        if len(positions) < 2:
            continue
        for left_index, left in enumerate(positions):
            for right in positions[left_index + 1 :]:
                total_edges = right - left
                if total_edges < 4 or total_edges % 2 != 0:
                    continue
                mid = left + total_edges // 2
                outbound = tuple(int(v) for v in nodes[left : mid + 1])
                if len(set(outbound)) != len(outbound):
                    continue
                if tuple(int(v) for v in nodes[mid : right + 1]) != tuple(reversed(outbound)):
                    continue

                reciprocal_failures: list[dict[str, Any]] = []
                road_id_discrepancies: list[dict[str, Any]] = []
                for offset in range(total_edges // 2):
                    out_edge = int(edge_ids[left + offset])
                    ret_edge = int(edge_ids[mid + (total_edges // 2 - 1 - offset)])
                    out_endpoint = revisit._endpoint_by_edge_id(G, out_edge)
                    ret_endpoint = revisit._endpoint_by_edge_id(G, ret_edge)
                    u = int(outbound[offset])
                    v = int(outbound[offset + 1])
                    if out_endpoint != (u, v) or ret_endpoint != (v, u):
                        reciprocal_failures.append(
                            {
                                "offset": offset,
                                "outbound_edge_id": out_edge,
                                "return_edge_id": ret_edge,
                                "expected_outbound": [u, v],
                                "actual_outbound": out_endpoint,
                                "expected_return": [v, u],
                                "actual_return": ret_endpoint,
                            }
                        )
                        continue
                    out_road_id = _road_id(G, out_edge)
                    ret_road_id = _road_id(G, ret_edge)
                    if out_road_id is not None and ret_road_id is not None and out_road_id != ret_road_id:
                        road_id_discrepancies.append(
                            {
                                "offset": offset,
                                "outbound_edge_id": out_edge,
                                "return_edge_id": ret_edge,
                                "outbound_road_id": out_road_id,
                                "return_road_id": ret_road_id,
                            }
                        )

                spans.append(
                    {
                        "anchor_vertex": anchor,
                        "turning_vertex": int(nodes[mid]),
                        "start_node_index": left,
                        "turn_node_index": mid,
                        "end_node_index": right,
                        "outbound_edge_count": total_edges // 2,
                        "total_spur_edge_count": total_edges,
                        "internal_spur_vertices": list(outbound[1:]),
                        "outbound_edge_ids": [int(e) for e in edge_ids[left:mid]],
                        "return_edge_ids": [int(e) for e in edge_ids[mid:right]],
                        "reciprocal_failures": reciprocal_failures,
                        "road_id_discrepancies": road_id_discrepancies,
                    }
                )
    return sorted(
        spans,
        key=lambda span: (
            span["start_node_index"],
            span["end_node_index"],
            span["anchor_vertex"],
            span["turning_vertex"],
        ),
    )


def _collapse_to_backbone(
    nodes: Sequence[int],
    span: dict[str, Any],
) -> tuple[int, ...]:
    left = int(span["start_node_index"])
    right = int(span["end_node_index"])
    anchor = int(span["anchor_vertex"])
    return tuple(int(v) for v in nodes[:left]) + (anchor,) + tuple(int(v) for v in nodes[right + 1 :])


def _span_has_internal_reuse_outside(
    nodes: Sequence[int],
    span: dict[str, Any],
) -> bool:
    left = int(span["start_node_index"])
    right = int(span["end_node_index"])
    outside = Counter(int(v) for v in tuple(nodes[:left]) + tuple(nodes[right + 1 :]))
    return any(outside[int(vertex)] > 0 for vertex in span["internal_spur_vertices"])


def _qualifying_spans(
    G: Any,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
    spans: Sequence[dict[str, Any]],
    repeated_directed_edge_count: int,
) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    if repeated_directed_edge_count > 0:
        return qualified
    for span in spans:
        backbone = _collapse_to_backbone(nodes, span)
        internal_reuse = _span_has_internal_reuse_outside(nodes, span)
        backbone_elementary = len(backbone) == len(set(backbone))
        if (
            not internal_reuse
            and backbone_elementary
            and not span["reciprocal_failures"]
            and not span["road_id_discrepancies"]
        ):
            qualified.append({**span, "backbone_node_ids": list(backbone)})
    return qualified


def _spans_overlap_or_nest(spans: Sequence[dict[str, Any]]) -> bool:
    intervals = [
        (int(span["start_node_index"]), int(span["end_node_index"]))
        for span in spans
    ]
    for index, (left_a, right_a) in enumerate(intervals):
        for left_b, right_b in intervals[index + 1 :]:
            if max(left_a, left_b) < min(right_a, right_b):
                return True
    return False


def _failure_reason(
    row: dict[str, Any],
    spans: Sequence[dict[str, Any]],
    qualified: Sequence[dict[str, Any]],
) -> str | None:
    if row["one_simple_spur"]:
        return None
    if len(qualified) > 1:
        return "multiple_spurs"
    if row["repeated_directed_edge_count"] > 0:
        return "repeated_directed_edges"
    if spans and _spans_overlap_or_nest(spans):
        return "nested/overlapping_repetition"
    if len(spans) > 1:
        return "multiple_spurs"
    if spans:
        for span in spans:
            backbone = _collapse_to_backbone(row["path_node_ids"], span)
            span_repeated_vertices = int(span["outbound_edge_count"])
            if (
                _span_has_internal_reuse_outside(row["path_node_ids"], span)
                or len(backbone) != len(set(backbone))
                or int(row["distinct_repeated_vertices"]) > span_repeated_vertices
            ):
                return "repeated_vertices_outside_a_candidate_spur"
        return "other"
    if row["distinct_repeated_vertices"] > 0:
        return "non-palindromic_repeated_corridor"
    return "other"


def _one_simple_spur(
    G: Any,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
) -> dict[str, Any]:
    stats = revisit._repetition_stats(G, nodes, edge_ids)
    spans = _candidate_spans(G, nodes, edge_ids)
    qualified = _qualifying_spans(
        G,
        nodes,
        edge_ids,
        spans,
        int(stats["repeated_directed_edge_count"]),
    )
    accepted = len(qualified) == 1
    chosen = qualified[0] if accepted else None
    spur_metrics = None
    if chosen is not None:
        left = int(chosen["start_node_index"])
        right = int(chosen["end_node_index"])
        spur_metrics = base._metrics_from_edge_ids(G, edge_ids[left:right])

    road_ids_available = getattr(G, "road_id", None) is not None
    road_discrepancies: list[dict[str, Any]] = []
    reciprocal_failures: list[dict[str, Any]] = []
    if chosen is not None:
        road_discrepancies = chosen["road_id_discrepancies"]
        reciprocal_failures = chosen["reciprocal_failures"]
    else:
        for span in spans:
            road_discrepancies.extend(span["road_id_discrepancies"])
            reciprocal_failures.extend(span["reciprocal_failures"])

    row: dict[str, Any] = {
        **stats,
        "path_node_ids": [int(v) for v in nodes],
        "one_simple_spur": accepted,
        "spur_anchor_vertex": None if chosen is None else chosen["anchor_vertex"],
        "turning_vertex": None if chosen is None else chosen["turning_vertex"],
        "outbound_edge_count": None if chosen is None else chosen["outbound_edge_count"],
        "total_spur_edge_count": None if chosen is None else chosen["total_spur_edge_count"],
        "spur_length": None if spur_metrics is None else spur_metrics.length,
        "spur_elevation_gain": None if spur_metrics is None else spur_metrics.elevation,
        "spur_popularity_length": None if spur_metrics is None else spur_metrics.popularity_length,
        "spur_width_length": None if spur_metrics is None else spur_metrics.width_length,
        "road_id_consistency_available": road_ids_available,
        "road_id_consistent_on_reciprocal_edges": (
            None if not road_ids_available else len(road_discrepancies) == 0
        ),
        "road_id_discrepancies": road_discrepancies,
        "reciprocal_edge_failures": reciprocal_failures,
        "candidate_spur_count": len(spans),
        "qualifying_spur_count": len(qualified),
        "candidate_spans": spans,
    }
    row["spur_failure_reason"] = _failure_reason(row, spans, qualified)
    return row


def _box_center_key(query: base.QueryBox, metrics: base.RouteMetrics) -> tuple[float, float]:
    return (query.normalized_violation_score(metrics), abs(metrics.length))


def _row_for_candidate(G: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    query = revisit._query_from_raw(candidate)
    nodes = tuple(int(v) for v in candidate["path_node_ids"])
    edge_ids = tuple(int(e) for e in candidate["csr_edge_ids"])
    metrics = base._metrics_from_edge_ids(G, edge_ids)
    result = via._make_path_result(0.0, nodes, edge_ids, metrics)
    validation = base._validate_path(G, result)
    feasible = validation.passed and query.is_feasible(metrics)
    shape = _one_simple_spur(G, nodes, edge_ids)
    row = {
        "status": "analyzed",
        "query_id": candidate["query_id"],
        "benchmark": candidate.get("benchmark"),
        "source_file": candidate["source_file"],
        "json_path": candidate["json_path"],
        "pair_type": candidate["pair_type"],
        "via_vertex": candidate["via_vertex"],
        "path_fingerprint": candidate.get("path_fingerprint"),
        "path_node_count": len(nodes),
        "edge_count": len(edge_ids),
        "csr_edge_ids": list(edge_ids),
        "total_L": metrics.length,
        "total_H": metrics.elevation,
        "total_P_length": metrics.popularity_length,
        "total_W_length": metrics.width_length,
        "total_avg_pop": metrics.avg_popularity,
        "total_avg_width": metrics.avg_width,
        "total_road_changes": metrics.road_changes,
        "box_feasible": feasible,
        "box_violations": query.violations(metrics),
        "box_center_score": query.normalized_violation_score(metrics),
        "validation_status": validation.as_dict(),
        **shape,
    }
    return row


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return {"median": None, "min": None, "max": None}
    return {
        "median": float(np.median(np.array(finite, dtype=np.float64))),
        "min": finite[0],
        "max": finite[-1],
    }


def _summary_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for query_id in sorted(TARGET_QUERY_IDS):
        subset = [row for row in rows if row["query_id"] == query_id]
        feasible = [row for row in subset if row["box_feasible"]]
        spur = [row for row in subset if row["one_simple_spur"]]
        spur_feasible = [row for row in spur if row["box_feasible"]]
        reasons = Counter(
            row["spur_failure_reason"]
            for row in subset
            if row["spur_failure_reason"] is not None
        )
        out[query_id] = {
            "total_unique_non_elementary_feasible_walks": len(feasible),
            "unique_recovered_walks": len(subset),
            "one_simple_spur_count": len(spur),
            "one_simple_spur_box_feasible_count": len(spur_feasible),
            "one_simple_spur_box_feasible_percentage": (
                None if not feasible else 100.0 * len(spur_feasible) / len(feasible)
            ),
            "spur_length_stats": _stats([row["spur_length"] for row in spur]),
            "spur_elevation_gain_stats": _stats([row["spur_elevation_gain"] for row in spur]),
            "failure_reason_counts": dict(sorted(reasons.items())),
        }
    return out


def _representative_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = [
        "query_id",
        "pair_type",
        "via_vertex",
        "path_fingerprint",
        "spur_anchor_vertex",
        "turning_vertex",
        "outbound_edge_count",
        "total_spur_edge_count",
        "spur_length",
        "spur_elevation_gain",
        "total_L",
        "total_H",
        "total_avg_pop",
        "total_avg_width",
        "box_center_score",
        "path_node_ids",
        "csr_edge_ids",
    ]
    return {key: row.get(key) for key in keys}


def _paris_representatives(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paris = [
        row
        for row in rows
        if row["query_id"] == "paris_bures_02_multi_tight" and row["one_simple_spur"]
    ]
    if not paris:
        return {
            "shortest_valid_one_simple_spur": None,
            "longest_valid_one_simple_spur": None,
            "highest_spur_elevation_gain": None,
            "best_box_centered_total_route": None,
        }
    return {
        "shortest_valid_one_simple_spur": _representative_payload(
            min(paris, key=lambda row: (float(row["spur_length"]), row["path_fingerprint"]))
        ),
        "longest_valid_one_simple_spur": _representative_payload(
            max(paris, key=lambda row: (float(row["spur_length"]), row["path_fingerprint"]))
        ),
        "highest_spur_elevation_gain": _representative_payload(
            max(paris, key=lambda row: (float(row["spur_elevation_gain"]), row["path_fingerprint"]))
        ),
        "best_box_centered_total_route": _representative_payload(
            min(
                paris,
                key=lambda row: (
                    float(row["box_center_score"]),
                    float(row["total_L"]),
                    row["path_fingerprint"],
                ),
            )
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-search out-and-back spur structure audit over recovered walks."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--candidate-source", default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = base._load_static_inputs(args)
    candidates, source_report = revisit._load_candidate_walks(args.candidate_source)

    rows: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in candidates:
        if candidate["query_id"] not in TARGET_QUERY_IDS:
            continue
        edge_key = (str(candidate["query_id"]), tuple(candidate["csr_edge_ids"]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        rows.append(_row_for_candidate(inputs.G, candidate))

    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "policy": "no Dijkstra and no new search; only recovered complete walk sequences are inspected",
            "candidate_source": args.candidate_source,
            "target_query_ids": sorted(TARGET_QUERY_IDS),
            "rule": "ONE_SIMPLE_SPUR",
        },
        "source_report": source_report,
        "rows": rows,
        "summary": {
            "status": "complete" if rows else "blocked_missing_recovered_walks",
            "candidate_walks_found": len(candidates),
            "unique_complete_edge_sequences": len(rows),
            "by_query": _summary_by_query(rows),
            "paris_bures_representatives": _paris_representatives(rows),
        },
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    print(
        f"Out-and-back spur audit status: {payload['summary']['status']}; "
        f"unique walks={payload['summary']['unique_complete_edge_sequences']}",
        flush=True,
    )
    for query_id, summary in payload["summary"]["by_query"].items():
        print(
            f"  {query_id}: feasible={summary['total_unique_non_elementary_feasible_walks']} "
            f"one_simple_spur={summary['one_simple_spur_count']} "
            f"spur_feasible={summary['one_simple_spur_box_feasible_count']}",
            flush=True,
        )
    print(f"Wrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
