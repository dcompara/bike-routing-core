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


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_revisit_structure_audit_from_recovered.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_revisit_structure_audit_from_recovered.csv"
DEFAULT_CANDIDATE_SOURCES = ("tmp_scalar_via_recovered_non_elementary_walks.json",)
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


def _walk_json(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        has_nodes = isinstance(value.get("path_node_ids"), list)
        has_edges = isinstance(value.get("csr_edge_ids"), list) or isinstance(value.get("edge_ids"), list)
        if has_nodes and has_edges:
            found.append((path or "/", value))
        for key, child in value.items():
            found.extend(_walk_json(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_json(child, f"{path}[{index}]"))
    return found


def _query_from_raw(raw: dict[str, Any]) -> base.QueryBox:
    query = raw.get("query")
    if not isinstance(query, dict):
        raise ValueError(f"candidate {raw.get('query_id')} has no embedded query")
    return base.QueryBox(
        name=str(query.get("name", raw["query_id"])),
        source=int(query["source"]),
        target=int(query["target"]),
        Lmin=float(query["Lmin"]),
        Lmax=float(query["Lmax"]),
        Hmin=float(query["Hmin"]),
        Hmax=float(query["Hmax"]),
        Pmin=float(query["Pmin"]),
        Wmax=float(query["Wmax"]),
        corridor_slack_m=int(query.get("corridor_slack_m", base.CORRIDOR_SLACK_M)),
        max_hops_from_boundary=int(
            query.get("max_hops_from_boundary", base.MAX_HOPS_FROM_BOUNDARY)
        ),
    )


def _load_candidate_walks(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = Path(path)
    if not source_path.exists():
        return [], {"path": path, "exists": False, "candidate_walks_found": 0}
    with open(source_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    candidates: list[dict[str, Any]] = []
    for json_path, obj in _walk_json(payload):
        query_id = obj.get("query_id") or obj.get("name")
        if query_id not in TARGET_QUERY_IDS:
            continue
        nodes = obj.get("path_node_ids")
        edges = obj.get("csr_edge_ids") or obj.get("edge_ids")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            continue
        candidates.append(
            {
                "source_file": path,
                "json_path": json_path,
                "query_id": str(query_id),
                "benchmark": obj.get("benchmark"),
                "query": obj.get("query"),
                "pair_type": obj.get("pair_type") or obj.get("original_pair_type"),
                "via_vertex": obj.get("via_vertex"),
                "path_fingerprint": obj.get("path_fingerprint"),
                "path_node_ids": [int(v) for v in nodes],
                "csr_edge_ids": [int(e) for e in edges],
            }
        )
    return candidates, {
        "path": path,
        "exists": True,
        "candidate_walks_found": len(candidates),
    }


def _endpoint_by_edge_id(G: Any, edge_id: int) -> tuple[int, int] | None:
    edge_id = int(edge_id)
    if edge_id < 0 or edge_id >= G.n_edges:
        return None
    u = int(np.searchsorted(G.offsets, edge_id, side="right") - 1)
    if u < 0 or u >= G.n_nodes:
        return None
    return u, int(G.to[edge_id])


def _repeated_undirected_edges(G: Any, edge_ids: Sequence[int]) -> int:
    undirected: list[tuple[int, int]] = []
    for edge_id in edge_ids:
        endpoint = _endpoint_by_edge_id(G, int(edge_id))
        if endpoint is None:
            continue
        u, v = endpoint
        undirected.append((min(u, v), max(u, v)))
    counts = Counter(undirected)
    return sum(count - 1 for count in counts.values() if count > 1)


def _repetition_stats(
    G: Any,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
) -> dict[str, Any]:
    node_counts = Counter(int(v) for v in nodes)
    repeated_vertices = {node: count for node, count in node_counts.items() if count > 1}
    edge_counts = Counter(int(edge_id) for edge_id in edge_ids)
    repeated_directed_edges = sum(count - 1 for count in edge_counts.values() if count > 1)
    return {
        "distinct_repeated_vertices": len(repeated_vertices),
        "maximum_vertex_multiplicity": max(node_counts.values()) if node_counts else 0,
        "total_repeated_occurrences": sum(count - 1 for count in repeated_vertices.values()),
        "repeated_directed_edge_count": repeated_directed_edges,
        "repeated_undirected_edge_count": _repeated_undirected_edges(G, edge_ids),
        "repeated_vertices": dict(sorted(repeated_vertices.items())),
    }


def _loop_spans(nodes: Sequence[int]) -> list[dict[str, Any]]:
    positions: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        positions.setdefault(int(node), []).append(index)
    spans: list[dict[str, Any]] = []
    for node, node_positions in sorted(positions.items()):
        if len(node_positions) < 2:
            continue
        for left, right in zip(node_positions, node_positions[1:]):
            spans.append(
                {
                    "anchor_vertex": node,
                    "start_node_index": left,
                    "end_node_index": right,
                    "edge_count": right - left,
                }
            )
    return sorted(spans, key=lambda item: (item["start_node_index"], item["end_node_index"], item["anchor_vertex"]))


def _is_two_edge_reciprocal_backtrack(
    G: Any,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
    left: int,
    right: int,
) -> bool:
    if right - left != 2:
        return False
    if len(edge_ids) <= left + 1:
        return False
    a = _endpoint_by_edge_id(G, int(edge_ids[left]))
    b = _endpoint_by_edge_id(G, int(edge_ids[left + 1]))
    if a is None or b is None:
        return False
    return a[0] == b[1] and a[1] == b[0] and int(nodes[left]) == int(nodes[right])


def _one_simple_loop(
    G: Any,
    query: base.QueryBox,
    nodes: Sequence[int],
    edge_ids: Sequence[int],
) -> dict[str, Any]:
    stats = _repetition_stats(G, nodes, edge_ids)
    repeated_vertices = stats["repeated_vertices"]
    accepted = (
        len(repeated_vertices) <= 1
        and all(count == 2 for count in repeated_vertices.values())
        and stats["repeated_directed_edge_count"] == 0
    )
    repeated_vertex = next(iter(repeated_vertices), None)
    loop_edge_count = None
    loop_metrics: base.RouteMetrics | None = None
    reciprocal_backtrack = False
    if accepted and repeated_vertex is not None:
        positions = [idx for idx, node in enumerate(nodes) if int(node) == int(repeated_vertex)]
        if len(positions) != 2:
            accepted = False
        else:
            left, right = positions
            loop_edge_count = right - left
            reciprocal_backtrack = _is_two_edge_reciprocal_backtrack(
                G,
                nodes,
                edge_ids,
                left,
                right,
            )
            if reciprocal_backtrack:
                accepted = False
            loop_metrics = base._metrics_from_edge_ids(G, edge_ids[left:right])
    elif accepted:
        loop_edge_count = 0
        loop_metrics = base.RouteMetrics(0.0, 0.0, 0.0, 0.0, 0)

    return {
        **stats,
        "one_simple_loop": accepted,
        "repeated_anchor_vertex": repeated_vertex,
        "loop_edge_count": loop_edge_count,
        "loop_length": None if loop_metrics is None else loop_metrics.length,
        "loop_elevation": None if loop_metrics is None else loop_metrics.elevation,
        "loop_popularity_length": None if loop_metrics is None else loop_metrics.popularity_length,
        "loop_width_length": None if loop_metrics is None else loop_metrics.width_length,
        "loop_avg_pop": None if loop_metrics is None else loop_metrics.avg_popularity,
        "loop_avg_width": None if loop_metrics is None else loop_metrics.avg_width,
        "two_edge_reciprocal_backtrack": reciprocal_backtrack,
        "loop_spans": _loop_spans(nodes),
    }


def _pattern_label(row: dict[str, Any]) -> str:
    if row["one_simple_loop"]:
        return "ONE_SIMPLE_LOOP"
    if row["repeated_directed_edge_count"] > 0:
        return "REPEATED_DIRECTED_EDGE"
    if row["maximum_vertex_multiplicity"] > 2:
        return "VERTEX_MULTIPLICITY_GT_2"
    if row["distinct_repeated_vertices"] > 1:
        return "MULTIPLE_REPEATED_VERTICES"
    if row.get("two_edge_reciprocal_backtrack"):
        return "TWO_EDGE_RECIPROCAL_BACKTRACK"
    return "OTHER"


def _row_for_candidate(G: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    query = _query_from_raw(candidate)
    nodes = tuple(int(v) for v in candidate["path_node_ids"])
    edge_ids = tuple(int(e) for e in candidate["csr_edge_ids"])
    metrics = base._metrics_from_edge_ids(G, edge_ids)
    result = via._make_path_result(0.0, nodes, edge_ids, metrics)
    validation = base._validate_path(G, result)
    shape = _one_simple_loop(G, query, nodes, edge_ids)
    feasible = validation.passed and query.is_feasible(metrics)
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
        "total_L": metrics.length,
        "total_H": metrics.elevation,
        "total_P_length": metrics.popularity_length,
        "total_W_length": metrics.width_length,
        "total_avg_pop": metrics.avg_popularity,
        "total_avg_width": metrics.avg_width,
        "total_road_changes": metrics.road_changes,
        "box_feasible": feasible,
        "box_violations": query.violations(metrics),
        "validation_status": validation.as_dict(),
        **shape,
    }
    row["pattern_label"] = _pattern_label(row)
    return row


def _summary_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for query_id in sorted(TARGET_QUERY_IDS):
        subset = [row for row in rows if row["query_id"] == query_id]
        patterns = Counter(row["pattern_label"] for row in subset if row["pattern_label"] != "ONE_SIMPLE_LOOP")
        out[query_id] = {
            "unique_non_elementary_walks": len(subset),
            "one_simple_loop_count": sum(1 for row in subset if row["one_simple_loop"]),
            "one_simple_loop_box_feasible_count": sum(
                1 for row in subset if row["one_simple_loop"] and row["box_feasible"]
            ),
            "other_repetition_pattern_counts": dict(sorted(patterns.items())),
        }
    return out


def _representatives(rows: Sequence[dict[str, Any]], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paris = [
        row
        for row in rows
        if row["query_id"] == "paris_bures_02_multi_tight" and row["one_simple_loop"]
    ]
    by_hash = {candidate.get("path_fingerprint"): candidate for candidate in candidates}
    reps: dict[str, Any] = {}
    selectors = {
        "shortest_loop": lambda row: (float(row["loop_length"]), row["path_fingerprint"]),
        "highest_elevation_loop": lambda row: (-float(row["loop_elevation"]), row["path_fingerprint"]),
        "best_box_centered_route": lambda row: (
            max(
                float(row["box_violations"]["length_low"]),
                float(row["box_violations"]["length_high"]),
                float(row["box_violations"]["elevation_low"]),
                float(row["box_violations"]["elevation_high"]),
                float(row["box_violations"]["popularity_low"]),
                float(row["box_violations"]["width_high"]),
            ),
            abs(float(row["total_L"])),
            row["path_fingerprint"],
        ),
    }
    for name, key in selectors.items():
        if not paris:
            reps[name] = None
            continue
        chosen = min(paris, key=key)
        raw = by_hash.get(chosen["path_fingerprint"])
        reps[name] = {
            "row": chosen,
            "path_node_ids": None if raw is None else raw["path_node_ids"],
            "csr_edge_ids": None if raw is None else raw["csr_edge_ids"],
        }
    return reps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-search revisit-structure audit over persisted recovered walks."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--candidate-source",
        action="append",
        default=None,
        help="JSON result file containing persisted non-elementary path_node_ids/csr_edge_ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = base._load_static_inputs(args)
    sources = args.candidate_source or list(DEFAULT_CANDIDATE_SOURCES)
    source_reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for source in sources:
        found, report = _load_candidate_walks(source)
        candidates.extend(found)
        source_reports.append(report)

    rows: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, tuple[int, ...]]] = set()
    unique_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        edge_key = (str(candidate["query_id"]), tuple(candidate["csr_edge_ids"]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        unique_candidates.append(candidate)
        rows.append(_row_for_candidate(inputs.G, candidate))

    blocked = not rows
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "policy": "no Dijkstra and no new search; only persisted recovered candidate walks are inspected",
            "target_query_ids": sorted(TARGET_QUERY_IDS),
            "candidate_sources": sources,
        },
        "source_reports": source_reports,
        "rows": rows,
        "summary": {
            "status": "blocked_missing_persisted_walk_sequences" if blocked else "complete",
            "candidate_walks_found": len(candidates),
            "unique_complete_edge_sequences": len(rows),
            "by_query": _summary_by_query(rows),
            "paris_bures_representatives": _representatives(rows, unique_candidates),
            "reason": None
            if not blocked
            else "No persisted recovered path_node_ids/csr_edge_ids were found.",
        },
    }
    if blocked:
        rows = [
            {
                "status": "blocked_missing_persisted_walk_sequences",
                "query_id": query_id,
                "candidate_walks_found": 0,
                "reason": payload["summary"]["reason"],
            }
            for query_id in sorted(TARGET_QUERY_IDS)
        ]
        payload["rows"] = rows

    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    print(
        f"Revisit-structure audit status: {payload['summary']['status']}; "
        f"unique walks={payload['summary']['unique_complete_edge_sequences']}",
        flush=True,
    )
    for query_id, summary in payload["summary"]["by_query"].items():
        print(
            f"  {query_id}: unique={summary['unique_non_elementary_walks']} "
            f"one_simple_loop={summary['one_simple_loop_count']} "
            f"feasible={summary['one_simple_loop_box_feasible_count']}",
            flush=True,
        )
    print(f"Wrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
