from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_benchmark as bench
import article_scalar_via_feasibility_experiment as via
import article_scalar_via_mixed_benchmark as mixed


DEFAULT_OUTPUT_JSON = "tmp_scalar_portfolio_coverage_audit_results.json"
DEFAULT_OUTPUT_CSV = "tmp_scalar_portfolio_coverage_audit_results.csv"
DEFAULT_VALIDATION_BOXES_JSON = mixed.DEFAULT_HOLDOUT_BOXES_JSON
DEFAULT_DEVELOPMENT_BOXES_JSON = mixed.DEFAULT_DEVELOPMENT_BOXES_JSON

BASELINE_PORTFOLIO = (bench.SCALAR_REFERENCE, bench.SCALAR_SLOPE)
CURRENT_FAILURE_IDS = (
    "holdout_anchor_east_west_09_multi_tight",
    "holdout_anchor_south_north_09_multi_tight",
    "holdout_anchor_south_north_28_quality_conflict",
)
DEVELOPMENT_FAILURE_IDS = ("paris_bures_02_multi_tight",)


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


def _load_items(
    inputs: base.StaticInputs,
    validation_path: str,
    development_path: str,
) -> dict[str, list[bench.BenchmarkItem]]:
    validation_items, _ = bench._load_items_from_boxes_json(inputs, validation_path)
    development_items, _ = bench._load_items_from_boxes_json(inputs, development_path)
    return {"validation": validation_items, "development": development_items}


def _raw_witnesses(path: str) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    boxes = payload["boxes"] if isinstance(payload, dict) and "boxes" in payload else payload
    out: dict[str, dict[str, Any]] = {}
    for raw in boxes:
        query = raw.get("query", raw)
        query_id = str(query.get("name", query.get("query_id")))
        witness = raw.get("witness", raw)
        out[query_id] = {
            "route_id": witness.get("route_id"),
            "scalar_name": witness.get("scalar_name"),
            "via_vertex": witness.get("via_vertex"),
            "metrics": witness.get("metrics"),
            "generator": witness.get("generator"),
            "witness_source": raw.get("witness_source") or witness.get("witness_source"),
            "witness_method": raw.get("witness_method") or witness.get("witness_method"),
        }
    return out


def _portfolio_specs(query: base.QueryBox) -> list[base.ScalarizationSpec]:
    return base._fixed_portfolio(query)


def _portfolio_names() -> list[str]:
    dummy = base.QueryBox("dummy", 0, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 99.0)
    return [spec.name for spec in _portfolio_specs(dummy)]


def _run_same_scalar_via(
    prepared: bench.PreparedGraph,
    original_G: Any,
    query: base.QueryBox,
    spec: base.ScalarizationSpec,
    reverse_adj: list[list[tuple[int, int]]],
) -> dict[str, Any]:
    forward_start = time.perf_counter()
    forward = via._run_scalar_tree(
        prepared.G,
        query,
        prepared.context,
        spec,
        reverse=False,
    )
    forward_s = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    backward = via._run_scalar_tree(
        prepared.G,
        query,
        prepared.context,
        spec,
        reverse=True,
        reverse_adj=reverse_adj,
    )
    backward_s = time.perf_counter() - backward_start
    profiles, scanned, nearest_profile, scan_s = mixed._scan_pair_profiles(
        query,
        spec.name,
        forward,
        backward,
    )
    (
        first_hit,
        reconstructed_first,
        first_non_elementary,
        first_validation,
        first_exact_box,
        first_recon_s,
    ) = mixed._reconstruct_first_hit(
        prepared,
        original_G,
        query,
        spec.name,
        forward,
        backward,
        profiles,
    )
    (
        exact,
        nearest_exact,
        reconstructed_exhaustive,
        non_elementary,
        validation,
        exact_box,
        exhaustive_recon_s,
    ) = mixed._reconstruct_exhaustive(
        prepared,
        original_G,
        query,
        spec.name,
        forward,
        backward,
        profiles,
    )
    tree_s = forward_s + backward_s
    time_to_first = None
    if first_hit is not None:
        time_to_first = tree_s + scan_s + first_recon_s
    exhaustive_total = tree_s + scan_s + exhaustive_recon_s
    best = None
    if exact:
        best = sorted(
            exact,
            key=lambda c: (c.box_score, c.metrics.length, c.pair_type, c.via_vertex),
        )[0]
    return {
        "scalar_name": spec.name,
        "family": spec.family,
        "parameters": spec.parameters,
        "forward_tree_time_s": forward_s,
        "backward_tree_time_s": backward_s,
        "tree_time_s": tree_s,
        "profile_scan_s": scan_s,
        "profile_feasible_count": len(profiles),
        "via_vertices_scanned": scanned,
        "exact_elementary_feasible_count": len(exact),
        "time_to_first_exact_feasible_s": time_to_first,
        "exhaustive_time_s": exhaustive_total,
        "exhaustive_reconstruction_s": exhaustive_recon_s,
        "reconstructed_before_first_hit": None if first_hit is None else reconstructed_first,
        "first_hit_non_elementary_rejections": first_non_elementary,
        "first_hit_validation_rejections": first_validation,
        "first_hit_exact_box_rejections": first_exact_box,
        "reconstructed_exhaustive": reconstructed_exhaustive,
        "non_elementary_count": non_elementary,
        "validation_failure_count": validation,
        "exact_box_rejection_count": exact_box,
        "feasible": bool(exact),
        "first_hit": None if first_hit is None else first_hit.as_dict(),
        "best_exact_route": None if best is None else best.as_dict(),
        "nearest_profile": nearest_profile,
        "nearest_exact": nearest_exact,
    }


def _result_rows_by_key(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["benchmark"]), str(row["query_id"]), str(row["scalar_name"])): row
        for row in rows
        if row.get("row_type") == "scalar_query"
    }


def _solved_sets(
    rows: Sequence[dict[str, Any]],
    scalar_names: Sequence[str],
) -> dict[str, dict[str, set[str]]]:
    out = {
        "validation": {name: set() for name in scalar_names},
        "development": {name: set() for name in scalar_names},
    }
    for row in rows:
        if row.get("row_type") != "scalar_query" or not bool(row["feasible"]):
            continue
        out[str(row["benchmark"])][str(row["scalar_name"])].add(str(row["query_id"]))
    return out


def _coverage_for_subset(
    subset: Sequence[str],
    solved_by_scalar: dict[str, set[str]],
) -> set[str]:
    solved: set[str] = set()
    for scalar_name in subset:
        solved |= solved_by_scalar[scalar_name]
    return solved


def _subset_runtime_score(
    subset: Sequence[str],
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    query_ids_by_benchmark: dict[str, list[str]],
) -> float:
    total = 0.0
    for benchmark, query_ids in query_ids_by_benchmark.items():
        for query_id in query_ids:
            for scalar_name in subset:
                row = rows_by_key[(benchmark, query_id, scalar_name)]
                if bool(row["feasible"]):
                    total += float(row["time_to_first_exact_feasible_s"])
                    break
                total += float(row["exhaustive_time_s"])
    return total


def _marginal_newly_solved(
    subset: Sequence[str],
    solved_by_scalar: dict[str, set[str]],
) -> list[dict[str, Any]]:
    covered: set[str] = set()
    out: list[dict[str, Any]] = []
    for scalar_name in subset:
        newly = sorted(solved_by_scalar[scalar_name] - covered)
        out.append(
            {
                "scalar_name": scalar_name,
                "newly_solved_count": len(newly),
                "newly_solved_queries": newly,
            }
        )
        covered |= solved_by_scalar[scalar_name]
    return out


def _best_subsets_by_k(
    scalar_names: Sequence[str],
    solved: dict[str, dict[str, set[str]]],
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    query_ids_by_benchmark: dict[str, list[str]],
    max_k: int,
) -> dict[str, Any]:
    validation_universe = set(query_ids_by_benchmark["validation"])
    best_by_k: dict[int, dict[str, Any]] = {}
    smallest_full: dict[str, Any] | None = None
    for k in range(1, max_k + 1):
        best: dict[str, Any] | None = None
        for combo in itertools.combinations(scalar_names, k):
            validation_solved = _coverage_for_subset(combo, solved["validation"])
            development_solved = _coverage_for_subset(combo, solved["development"])
            runtime_score = _subset_runtime_score(combo, rows_by_key, query_ids_by_benchmark)
            key = (
                len(validation_solved),
                len(development_solved),
                -runtime_score,
                tuple(reversed(combo)),
            )
            if best is None or key > best["_key"]:
                best = {
                    "_key": key,
                    "K": k,
                    "scalars": list(combo),
                    "validation_solved": len(validation_solved),
                    "validation_total": len(validation_universe),
                    "validation_queries": sorted(validation_solved),
                    "development_solved": len(development_solved),
                    "development_total": len(query_ids_by_benchmark["development"]),
                    "development_queries": sorted(development_solved),
                    "runtime_score_s": runtime_score,
                }
        assert best is not None
        best.pop("_key", None)
        best["marginal_validation"] = _marginal_newly_solved(
            best["scalars"],
            solved["validation"],
        )
        best["marginal_development"] = _marginal_newly_solved(
            best["scalars"],
            solved["development"],
        )
        best_by_k[k] = best
        if smallest_full is None and best["validation_solved"] == len(validation_universe):
            smallest_full = best
    max_validation = max(item["validation_solved"] for item in best_by_k.values())
    smallest_max = next(
        item for k, item in sorted(best_by_k.items()) if item["validation_solved"] == max_validation
    )
    return {
        "smallest_full_validation_cover": smallest_full,
        "smallest_subset_at_max_validation_coverage": smallest_max,
        "best_by_k": {str(k): value for k, value in best_by_k.items()},
    }


def _choose_order(
    subset: Sequence[str],
    solved: dict[str, dict[str, set[str]]],
    rows: Sequence[dict[str, Any]],
) -> list[str]:
    row_subset = [
        row
        for row in rows
        if row.get("row_type") == "scalar_query" and row["scalar_name"] in subset
    ]
    runtime_by_scalar: dict[str, float] = {}
    for scalar_name in subset:
        times = [
            float(row["time_to_first_exact_feasible_s"])
            for row in row_subset
            if row["scalar_name"] == scalar_name
            and bool(row["feasible"])
            and row["time_to_first_exact_feasible_s"] not in (None, "")
        ]
        runtime_by_scalar[scalar_name] = statistics.median(times) if times else float("inf")
    return sorted(
        subset,
        key=lambda name: (
            -(len(solved["validation"][name]) + len(solved["development"][name])),
            runtime_by_scalar[name],
            -len(solved["validation"][name]),
            name,
        ),
    )


def _portfolio_sequential_runtime(
    subset: Sequence[str],
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    query_ids_by_benchmark: dict[str, list[str]],
    order: Sequence[str],
) -> dict[str, Any]:
    del subset
    times: list[float] = []
    evaluated_counts: list[int] = []
    solved_count = 0
    total_count = 0
    by_query: list[dict[str, Any]] = []
    for benchmark, query_ids in query_ids_by_benchmark.items():
        for query_id in query_ids:
            total_count += 1
            elapsed = 0.0
            evaluated = 0
            hit_scalar: str | None = None
            for scalar_name in order:
                evaluated += 1
                row = rows_by_key[(benchmark, query_id, scalar_name)]
                if bool(row["feasible"]):
                    elapsed += float(row["time_to_first_exact_feasible_s"])
                    hit_scalar = scalar_name
                    solved_count += 1
                    break
                elapsed += float(row["exhaustive_time_s"])
            times.append(elapsed)
            evaluated_counts.append(evaluated)
            by_query.append(
                {
                    "benchmark": benchmark,
                    "query_id": query_id,
                    "solved": hit_scalar is not None,
                    "hit_scalar": hit_scalar,
                    "elapsed_s": elapsed,
                    "scalar_tree_pairs_evaluated": evaluated,
                }
            )
    ordered = sorted(times)
    p90_idx = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "order": list(order),
        "solved": solved_count,
        "attempted": total_count,
        "median_time_to_first_feasible_s": statistics.median(times),
        "p90_time_to_first_feasible_s": ordered[p90_idx],
        "max_time_to_first_feasible_s": max(times),
        "mean_scalar_tree_pairs_evaluated_before_success": statistics.mean(evaluated_counts),
        "by_query": by_query,
    }


def _baseline_summary(
    solved: dict[str, dict[str, set[str]]],
    query_ids_by_benchmark: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "scalars": list(BASELINE_PORTFOLIO),
        "validation_solved": len(_coverage_for_subset(BASELINE_PORTFOLIO, solved["validation"])),
        "validation_total": len(query_ids_by_benchmark["validation"]),
        "development_solved": len(_coverage_for_subset(BASELINE_PORTFOLIO, solved["development"])),
        "development_total": len(query_ids_by_benchmark["development"]),
        "marginal_validation": _marginal_newly_solved(BASELINE_PORTFOLIO, solved["validation"]),
        "marginal_development": _marginal_newly_solved(BASELINE_PORTFOLIO, solved["development"]),
    }


def _failure_provenance(
    validation_witnesses: dict[str, dict[str, Any]],
    development_witnesses: dict[str, dict[str, Any]],
    solved: dict[str, dict[str, set[str]]],
    scalar_names: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for query_id in CURRENT_FAILURE_IDS:
        out[query_id] = {
            "benchmark": "validation",
            "witness": validation_witnesses[query_id],
            "solving_scalarizations": [
                scalar_name
                for scalar_name in scalar_names
                if query_id in solved["validation"][scalar_name]
            ],
        }
    for query_id in DEVELOPMENT_FAILURE_IDS:
        out[query_id] = {
            "benchmark": "development",
            "witness": development_witnesses[query_id],
            "solving_scalarizations": [
                scalar_name
                for scalar_name in scalar_names
                if query_id in solved["development"][scalar_name]
            ],
        }
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal existing scalar-portfolio coverage audit."
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--validation-boxes-json", default=DEFAULT_VALIDATION_BOXES_JSON)
    parser.add_argument("--development-boxes-json", default=DEFAULT_DEVELOPMENT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-cover-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = bench._compute_global_metric_constants(inputs)
    prepared = mixed._full_prepared(inputs, global_constants)
    reverse_adj = via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)
    items_by_benchmark = _load_items(
        inputs,
        args.validation_boxes_json,
        args.development_boxes_json,
    )
    scalar_names = _portfolio_names()
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for benchmark_label, items in items_by_benchmark.items():
        for item_index, item in enumerate(items, start=1):
            specs_by_name = {spec.name: spec for spec in _portfolio_specs(item.query)}
            for scalar_index, scalar_name in enumerate(scalar_names, start=1):
                print(
                    f"[{benchmark_label} {item_index}/{len(items)} "
                    f"scalar {scalar_index}/{len(scalar_names)}] "
                    f"{item.query.name} :: {scalar_name}",
                    flush=True,
                )
                result = _run_same_scalar_via(
                    prepared,
                    inputs.G,
                    item.query,
                    specs_by_name[scalar_name],
                    reverse_adj,
                )
                row = {
                    "row_type": "scalar_query",
                    "benchmark": benchmark_label,
                    "query_id": item.query.name,
                    "source": item.query.source,
                    "target": item.query.target,
                    **{
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "first_hit",
                            "best_exact_route",
                            "nearest_profile",
                            "nearest_exact",
                        }
                    },
                    "first_hit": result["first_hit"],
                    "best_exact_route": result["best_exact_route"],
                    "nearest_profile": result["nearest_profile"],
                    "nearest_exact": result["nearest_exact"],
                }
                rows.append(row)
                artifacts.append(
                    {
                        "benchmark": benchmark_label,
                        "query_id": item.query.name,
                        "scalar_name": scalar_name,
                        "result": result,
                    }
                )

    rows_by_key = _result_rows_by_key(rows)
    query_ids_by_benchmark = {
        label: [item.query.name for item in items]
        for label, items in items_by_benchmark.items()
    }
    solved = _solved_sets(rows, scalar_names)
    cover = _best_subsets_by_k(
        scalar_names,
        solved,
        rows_by_key,
        query_ids_by_benchmark,
        args.max_cover_k,
    )
    baseline = _baseline_summary(solved, query_ids_by_benchmark)
    validation_witnesses = _raw_witnesses(args.validation_boxes_json)
    development_witnesses = _raw_witnesses(args.development_boxes_json)
    provenance = _failure_provenance(
        validation_witnesses,
        development_witnesses,
        solved,
        scalar_names,
    )
    chosen_portfolios = [baseline]
    for key in sorted(cover["best_by_k"], key=lambda raw: int(raw)):
        chosen_portfolios.append(cover["best_by_k"][key])
    if cover["smallest_full_validation_cover"] is not None:
        chosen_portfolios.append(cover["smallest_full_validation_cover"])

    seen_portfolios: set[tuple[str, ...]] = set()
    runtime_summaries: list[dict[str, Any]] = []
    for portfolio in chosen_portfolios:
        subset = tuple(portfolio["scalars"])
        if subset in seen_portfolios:
            continue
        seen_portfolios.add(subset)
        order = _choose_order(subset, solved, rows)
        runtime = _portfolio_sequential_runtime(
            subset,
            rows_by_key,
            query_ids_by_benchmark,
            order,
        )
        runtime["scalars"] = list(subset)
        runtime["validation_solved"] = len(_coverage_for_subset(subset, solved["validation"]))
        runtime["development_solved"] = len(_coverage_for_subset(subset, solved["development"]))
        runtime_summaries.append(runtime)

    summary = {
        "scalar_count": len(scalar_names),
        "scalar_names": scalar_names,
        "baseline": baseline,
        "cover_analysis": cover,
        "failure_witness_provenance": provenance,
        "sequential_runtime": runtime_summaries,
    }
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_mode": "full",
            "validation_boxes_json": args.validation_boxes_json,
            "development_boxes_json": args.development_boxes_json,
            "frozen_boxes_policy": "boxes are loaded from existing JSON files; thresholds and witnesses are not modified",
            "method": "same-scalar via scan/reconstruction for every scalar already present in base._fixed_portfolio",
            "validation_set_note": "the 12-box set is used for architecture selection here, not final generalization evidence",
            "rho_H_global": rho_info,
        },
        "rows": rows,
        "artifacts": artifacts,
        "summary": summary,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)

    print("\nScalar portfolio coverage audit", flush=True)
    print(
        f"  baseline {BASELINE_PORTFOLIO}: validation "
        f"{baseline['validation_solved']}/{baseline['validation_total']}, "
        f"development {baseline['development_solved']}/{baseline['development_total']}",
        flush=True,
    )
    for key, value in cover["best_by_k"].items():
        print(
            f"  K={key}: validation {value['validation_solved']}/"
            f"{value['validation_total']}, development {value['development_solved']}/"
            f"{value['development_total']} :: {value['scalars']}",
            flush=True,
        )
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
