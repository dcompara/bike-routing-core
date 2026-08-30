from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_benchmark as bench
import article_scalar_via_feasibility_experiment as via
import article_scalar_via_mixed_benchmark as mixed


DEFAULT_OUTPUT_JSON = "tmp_scalar_via_recovered_non_elementary_walks.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_via_recovered_non_elementary_walks.csv"
DEFAULT_HOLDOUT_BOXES_JSON = mixed.DEFAULT_HOLDOUT_BOXES_JSON
DEFAULT_DEVELOPMENT_BOXES_JSON = mixed.DEFAULT_DEVELOPMENT_BOXES_JSON

TARGETS = {
    "holdout": {
        "path": DEFAULT_HOLDOUT_BOXES_JSON,
        "query_ids": {
            "holdout_anchor_south_north_09_multi_tight",
            "holdout_anchor_south_north_28_quality_conflict",
        },
    },
    "development": {
        "path": DEFAULT_DEVELOPMENT_BOXES_JSON,
        "query_ids": {"paris_bures_02_multi_tight"},
    },
}
EXPECTED_NON_ELEMENTARY_COUNTS = {
    "holdout_anchor_south_north_09_multi_tight": 2,
    "holdout_anchor_south_north_28_quality_conflict": 114,
    "paris_bures_02_multi_tight": 160,
}


def _json_default(value: Any) -> Any:
    return mixed._json_default(value)


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


def _path_fingerprint(edge_ids: Sequence[int]) -> str:
    raw = ",".join(str(int(edge_id)) for edge_id in edge_ids).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _load_target_items(
    inputs: base.StaticInputs,
    holdout_path: str,
    development_path: str,
) -> dict[str, list[bench.BenchmarkItem]]:
    paths = {"holdout": holdout_path, "development": development_path}
    out: dict[str, list[bench.BenchmarkItem]] = {}
    for label, spec in TARGETS.items():
        items, _ = bench._load_items_from_boxes_json(inputs, paths[label])
        by_name = {item.query.name: item for item in items}
        missing = sorted(spec["query_ids"] - set(by_name))
        if missing:
            raise ValueError(f"{label} target boxes missing: {missing}")
        out[label] = [by_name[name] for name in sorted(spec["query_ids"])]
    return out


def _build_trees(
    prepared: bench.PreparedGraph,
    query: base.QueryBox,
) -> tuple[dict[str, dict[str, via.TreeResult]], float]:
    specs = mixed._specs(query)
    reverse_adj = via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)
    trees: dict[str, dict[str, via.TreeResult]] = {}
    start = time.perf_counter()
    for key in ("P", "S"):
        trees[key] = {
            "forward": via._run_scalar_tree(
                prepared.G,
                query,
                prepared.context,
                specs[key],
                reverse=False,
            ),
            "backward": via._run_scalar_tree(
                prepared.G,
                query,
                prepared.context,
                specs[key],
                reverse=True,
                reverse_adj=reverse_adj,
            ),
        }
    return trees, time.perf_counter() - start


def _recover_for_item(
    prepared: bench.PreparedGraph,
    original_G: Any,
    benchmark_label: str,
    item: bench.BenchmarkItem,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = item.query
    trees, tree_s = _build_trees(prepared, query)
    raw_candidates: list[dict[str, Any]] = []
    profile_counts = {pair_type: 0 for pair_type in mixed.PAIR_TYPES}
    non_elementary_counts = {pair_type: 0 for pair_type in mixed.PAIR_TYPES}
    validation_failures = 0
    profile_scan_s = 0.0
    reconstruction_s = 0.0
    for pair_type in mixed.PAIR_TYPES:
        forward_key, backward_key = mixed._tree_key(pair_type)
        profiles, _, _, scan_s = mixed._scan_pair_profiles(
            query,
            pair_type,
            trees[forward_key]["forward"],
            trees[backward_key]["backward"],
        )
        profile_counts[pair_type] = len(profiles)
        profile_scan_s += scan_s
        recon_start = time.perf_counter()
        for profile in profiles:
            candidate, reason, validation, repeated = mixed._reconstruct_candidate(
                prepared,
                original_G,
                query,
                pair_type,
                trees[forward_key]["forward"],
                trees[backward_key]["backward"],
                profile,
            )
            if candidate is None:
                validation_failures += 1
                continue
            if candidate.elementary:
                continue
            non_elementary_counts[pair_type] += 1
            raw_candidates.append(
                {
                    "query_id": query.name,
                    "benchmark": benchmark_label,
                    "query": query.as_dict(),
                    "pair_type": pair_type,
                    "via_vertex": candidate.via_vertex,
                    "path_node_ids": list(candidate.path_nodes),
                    "csr_edge_ids": list(candidate.edge_ids),
                    "path_fingerprint": _path_fingerprint(candidate.edge_ids),
                    "L": candidate.metrics.length,
                    "H": candidate.metrics.elevation,
                    "P_length": candidate.metrics.popularity_length,
                    "W_length": candidate.metrics.width_length,
                    "avg_pop": candidate.metrics.avg_popularity,
                    "avg_width": candidate.metrics.avg_width,
                    "road_changes": candidate.metrics.road_changes,
                    "repeated_vertex_count": candidate.repeated_vertex_count,
                    "validation": candidate.validation.as_dict(),
                    "validation_status": candidate.validation.as_dict(),
                    "profile_metrics": candidate.profile_metrics.as_dict(),
                    "profile_exact_deltas": candidate.profile_exact_deltas,
                }
            )
        reconstruction_s += time.perf_counter() - recon_start

    unique: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, ...]] = set()
    for candidate in sorted(
        raw_candidates,
        key=lambda c: (
            c["pair_type"],
            int(c["via_vertex"]),
            c["path_fingerprint"],
        ),
    ):
        edge_key = tuple(int(edge_id) for edge_id in candidate["csr_edge_ids"])
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        unique.append(candidate)

    expected = EXPECTED_NON_ELEMENTARY_COUNTS[query.name]
    summary = {
        "benchmark": benchmark_label,
        "query_id": query.name,
        "profile_feasible_by_pair": profile_counts,
        "non_elementary_by_pair": non_elementary_counts,
        "raw_non_elementary_count": len(raw_candidates),
        "unique_non_elementary_count": len(unique),
        "expected_non_elementary_count": expected,
        "count_matches_expected": len(raw_candidates) == expected,
        "unique_count_matches_expected": len(unique) == expected,
        "validation_failures": validation_failures,
        "tree_computation_s": tree_s,
        "profile_scan_s": profile_scan_s,
        "reconstruction_s": reconstruction_s,
    }
    return unique, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing profile-feasible non-elementary mixed-via walks."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--holdout-boxes-json", default=DEFAULT_HOLDOUT_BOXES_JSON)
    parser.add_argument("--development-boxes-json", default=DEFAULT_DEVELOPMENT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = bench._compute_global_metric_constants(inputs)
    prepared = mixed._full_prepared(inputs, global_constants)
    items = _load_target_items(inputs, args.holdout_boxes_json, args.development_boxes_json)
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, label_items in items.items():
        for item in label_items:
            print(f"[recover {label}] {item.query.name}", flush=True)
            recovered, summary = _recover_for_item(prepared, inputs.G, label, item)
            candidates.extend(recovered)
            summaries.append(summary)

    discrepancies = [
        summary
        for summary in summaries
        if not summary["count_matches_expected"]
    ]
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "purpose": "data recovery for revisit-structure audit, not new performance evidence",
            "graph_mode": "full",
            "holdout_boxes_json": args.holdout_boxes_json,
            "development_boxes_json": args.development_boxes_json,
            "pair_types": list(mixed.PAIR_TYPES),
            "P": mixed.P_NAME,
            "S": mixed.S_NAME,
            "frozen_policy": "no boxes, scalar formulas, candidate ordering, feasibility rules, or elementarity rules are changed",
            "rho_H_global": rho_info,
        },
        "status": "count_discrepancy" if discrepancies else "complete",
        "summaries": summaries,
        "discrepancies": discrepancies,
        "candidates": candidates,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, candidates)
    print(
        f"Recovery status: {payload['status']}; candidates={len(candidates)}",
        flush=True,
    )
    for summary in summaries:
        print(
            f"  {summary['query_id']}: raw={summary['raw_non_elementary_count']} "
            f"unique={summary['unique_non_elementary_count']} "
            f"expected={summary['expected_non_elementary_count']}",
            flush=True,
        )
    print(f"Wrote {args.output_json} and {args.output_csv}", flush=True)
    if discrepancies:
        sys.exit(2)


if __name__ == "__main__":
    main()
