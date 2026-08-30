from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_portfolio_coverage_audit as portfolio
import article_scalar_via_benchmark as bench
import article_scalar_via_mixed_benchmark as mixed


DEFAULT_CONFIG_JSON = "tmp_scalar_via_final_holdout_frozen_config.json"
DEFAULT_BOXES_JSON = "tmp_scalar_via_final_holdout_boxes.json"
DEFAULT_RESULTS_JSON = "tmp_scalar_via_final_holdout_results.json"
DEFAULT_RESULTS_CSV = "tmp_scalar_via_final_holdout_results.csv"
DEFAULT_DEVELOPMENT_BOXES_JSON = "tmp_scalar_via_benchmark_boxes.json"
DEFAULT_VALIDATION_BOXES_JSON = "tmp_scalar_via_holdout_boxes.json"

FINAL_SCALAR_1 = "hinge_width_strong"
FINAL_SCALAR_2 = "hinge_low_length_pressure"
CONTROL_PHYSICAL = "physical_length"
BASELINE_UNION_2 = "VIA-UNION-2"
FINAL_K2 = "FROZEN-K2-SEQUENTIAL"
HWS_METHOD = "hinge_width_strong"
RANDOM_COST_SOURCE_A = "SOURCE_A_RANDOM_POSITIVE_EDGE_COST_PATH"
RANDOM_COST_SOURCE_B = "SOURCE_B_RANDOM_WAYPOINT_RANDOM_COST_ROUTE"


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


def _sha256_jsonable(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_by_name(query: base.QueryBox, name: str) -> base.ScalarizationSpec:
    if name == CONTROL_PHYSICAL:
        return bench._physical_spec()
    for spec in base._fixed_portfolio(query):
        if spec.name == name:
            return spec
    raise ValueError(f"scalarization {name!r} is not present in the fixed portfolio")


def _frozen_config(seed: int) -> dict[str, Any]:
    dummy = base.QueryBox("formula_reference", 0, 1, 0.0, 1.0, 0.0, 1.0, 150.0, 15.0)
    scalar_1 = _spec_by_name(dummy, FINAL_SCALAR_1)
    scalar_2 = _spec_by_name(dummy, FINAL_SCALAR_2)
    audited_names = [spec.name for spec in base._fixed_portfolio(dummy)]
    config = {
        "scientific_status": {
            "24_original_boxes": "DEVELOPMENT SET; not independent generalization evidence",
            "12_former_holdout_boxes": "VALIDATION / ARCHITECTURE-SELECTION SET; not independent generalization evidence",
            "new_final_boxes": "FINAL INDEPENDENT HOLDOUT",
        },
        "seed": seed,
        "candidate_architecture": FINAL_K2,
        "scalar_execution_order": [FINAL_SCALAR_1, FINAL_SCALAR_2],
        "second_scalar_policy": f"{FINAL_SCALAR_2} is evaluated if and only if {FINAL_SCALAR_1} has no exact elementary feasible route",
        "strict_elementarity_rule": "path_node_ids must contain no repeated vertex",
        "rejected_relaxed_revisit_rules": ["ONE_SIMPLE_LOOP", "ONE_SIMPLE_SPUR"],
        "road_id_policy": (
            "road_id is used only by the existing deterministic shortest-tree tie behavior; "
            "it is not an objective or post-hoc tuning signal"
        ),
        "same_scalar_via_pipeline": [
            "forward shortest-path tree",
            "backward shortest-path tree",
            "same-scalar via profile scan using exact additive L,H,P_length,W_length",
            "profile box filtering",
            "candidate reconstruction",
            "directed CSR validation",
            "strict elementary-path validation",
        ],
        "hinge_formula": (
            "c(e)=alpha_L*l(e)+l(e)*(alpha_w*max(0,width(e)-Wref)/max(Wscale,EPS)"
            "+alpha_p*max(0,Pref-popularity(e))/max(Pscale,EPS))"
        ),
        "query_dependent_references": {
            "Wref": "query.Wmax",
            "Pref": "query.Pmin",
            "Wscale": "safe_scale(Wref)=max(abs(Wref),1.0) unless explicitly supplied",
            "Pscale": "safe_scale(Pref)=max(abs(Pref),1.0) unless explicitly supplied",
            "EPS": base.EPS,
        },
        "scalar_1": {
            "name": scalar_1.name,
            "family": scalar_1.family,
            "parameters": scalar_1.parameters,
            "effective_references": {"Wref": "query.Wmax", "Pref": "query.Pmin"},
        },
        "scalar_2": {
            "name": scalar_2.name,
            "family": scalar_2.family,
            "parameters": scalar_2.parameters,
            "effective_references": {"Wref": "query.Wmax", "Pref": "query.Pmin"},
        },
        "forbidden_witness_generators": {
            "audited_fixed_portfolio_scalarizations": audited_names,
            "additional_explicit_forbidden": [
                FINAL_SCALAR_1,
                FINAL_SCALAR_2,
                "pop_width_reference",
                "slope_exp_beta_150_width",
                "physical_length same-scalar via",
            ],
            "witness_generation_policy": (
                "final holdout witnesses are generated only by independent random positive edge-cost paths "
                "or independent random waypoint plus random-cost routes"
            ),
        },
    }
    config["frozen_configuration_sha256"] = _sha256_jsonable(config)
    return config


def _valid_nodes(inputs: base.StaticInputs) -> list[int]:
    G = inputs.G
    out_degree = np.diff(G.offsets) > 0
    in_degree = np.bincount(G.to.astype(np.int64), minlength=G.n_nodes) > 0
    return [int(node) for node in np.flatnonzero(out_degree & in_degree)]


def _load_existing_od_pairs(paths: Sequence[str]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        boxes = payload["boxes"] if isinstance(payload, dict) and "boxes" in payload else payload
        for raw in boxes:
            query = raw.get("query", raw)
            pairs.add((int(query["source"]), int(query["target"])))
    return pairs


def _full_prepared(inputs: base.StaticInputs) -> bench.PreparedGraph:
    constants, _ = bench._compute_global_metric_constants(inputs)
    return mixed._full_prepared(inputs, constants)


def _random_costs(G: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    u = rng.random(G.n_edges, dtype=np.float64)
    lengths = G.w[:, 0].astype(np.float64)
    return lengths * (0.2 + 3.0 * u)


def _dijkstra_fixed_edge_cost(
    G: Any,
    source: int,
    target: int,
    edge_cost: np.ndarray,
) -> base.ScalarPathResult:
    start = time.perf_counter()
    dist = np.full(G.n_nodes, float("inf"), dtype=np.float64)
    hops = np.full(G.n_nodes, 2**31 - 1, dtype=np.int32)
    parent_node = np.full(G.n_nodes, -1, dtype=np.int32)
    parent_edge = np.full(G.n_nodes, -1, dtype=np.int32)
    dist[source] = 0.0
    hops[source] = 0
    heap: list[tuple[float, int, int, int]] = [(0.0, 0, -1, int(source))]
    heap_pops = 0
    expanded_nodes = 0
    edge_scans = 0
    raw_edge_rows_checked = 0

    while heap:
        current, hop_count, last_edge, node = heapq.heappop(heap)
        heap_pops += 1
        if abs(current - float(dist[node])) > base.EPS or hop_count != int(hops[node]):
            continue
        expanded_nodes += 1
        if node == int(target):
            break
        start_idx = int(G.offsets[node])
        end_idx = int(G.offsets[node + 1])
        raw_edge_rows_checked += end_idx - start_idx
        for edge_id in range(start_idx, end_idx):
            step = float(edge_cost[edge_id])
            if not math.isfinite(step) or step <= 0.0:
                raise ValueError(f"random edge cost is not strictly positive on edge {edge_id}")
            edge_scans += 1
            nxt = int(G.to[edge_id])
            candidate = current + step
            next_hops = hop_count + 1
            old = float(dist[nxt])
            improve = candidate < old - base.EPS
            tie = abs(candidate - old) <= base.EPS and (
                next_hops < int(hops[nxt])
                or (next_hops == int(hops[nxt]) and edge_id < int(parent_edge[nxt]))
            )
            if improve or tie:
                dist[nxt] = candidate
                hops[nxt] = next_hops
                parent_node[nxt] = node
                parent_edge[nxt] = edge_id
                heapq.heappush(heap, (candidate, next_hops, edge_id, nxt))

    stats = base.DijkstraStats(
        heap_pops=heap_pops,
        expanded_nodes=expanded_nodes,
        edge_scans=edge_scans,
        raw_edge_rows_checked=raw_edge_rows_checked,
        elapsed_s=time.perf_counter() - start,
    )
    if not math.isfinite(float(dist[target])):
        return base.ScalarPathResult(False, float("inf"), (), (), None, stats)
    nodes: list[int] = []
    edges: list[int] = []
    cur = int(target)
    while cur != int(source):
        edge_id = int(parent_edge[cur])
        prev = int(parent_node[cur])
        if edge_id < 0 or prev < 0:
            return base.ScalarPathResult(False, float("inf"), (), (), None, stats)
        nodes.append(cur)
        edges.append(edge_id)
        cur = prev
    nodes.append(int(source))
    nodes.reverse()
    edges.reverse()
    edge_tuple = tuple(int(edge_id) for edge_id in edges)
    metrics = base._metrics_from_edge_ids(G, edge_tuple)
    return base.ScalarPathResult(
        True,
        float(dist[target]),
        tuple(int(node) for node in nodes),
        edge_tuple,
        metrics,
        stats,
    )


def _path_fingerprint(edge_ids: Sequence[int]) -> str:
    raw = ",".join(str(int(edge_id)) for edge_id in edge_ids).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _elementary_valid_result(G: Any, result: base.ScalarPathResult) -> tuple[bool, base.RouteValidation, int]:
    validation = base._validate_path(G, result)
    elementary, repeated = base._elementary_status(result.path_nodes)
    return bool(validation.passed and elementary), validation, repeated


def _physical_shortest(
    inputs: base.StaticInputs,
    prepared: bench.PreparedGraph,
    source: int,
    target: int,
    pair_id: str,
) -> bench.WitnessRoute | None:
    return bench._direct_witness_for_pair(
        inputs,
        prepared,
        pair_id,
        int(source),
        int(target),
        bench._physical_spec(),
        route_id_suffix="ordinary_physical_shortest_diagnostic",
    )


def _make_random_witness(
    inputs: base.StaticInputs,
    pair_id: str,
    route_id: str,
    source: int,
    target: int,
    result: base.ScalarPathResult,
    *,
    generator: str,
    provenance: dict[str, Any],
) -> dict[str, Any] | None:
    if not result.route_found or result.metrics is None:
        return None
    ok, validation, repeated = _elementary_valid_result(inputs.G, result)
    if not ok:
        return None
    metrics = base._metrics_from_edge_ids(inputs.G, result.edge_ids)
    return {
        "route_id": route_id,
        "pair_id": pair_id,
        "generator": generator,
        "witness_generation_source": generator,
        "scalar_name": "independent_random_positive_edge_cost",
        "via_vertex": None,
        "source": int(source),
        "target": int(target),
        "path_nodes": len(result.path_nodes),
        "edges": len(result.edge_ids),
        "path_node_ids": list(result.path_nodes),
        "csr_edge_ids": list(result.edge_ids),
        "path_fingerprint": _path_fingerprint(result.edge_ids),
        "metrics": metrics.as_dict(),
        "L": metrics.length,
        "H": metrics.elevation,
        "P_length": metrics.popularity_length,
        "W_length": metrics.width_length,
        "avg_pop": metrics.avg_popularity,
        "avg_width": metrics.avg_width,
        "road_changes": metrics.road_changes,
        "validation": validation.as_dict(),
        "directed_validation": validation.as_dict(),
        "elementary": True,
        "strict_elementarity": True,
        "repeated_vertex_count": repeated,
        "provenance": provenance,
    }


def _source_a_witness(
    inputs: base.StaticInputs,
    pair_id: str,
    source: int,
    target: int,
    seed: int,
    index: int,
) -> dict[str, Any] | None:
    costs = _random_costs(inputs.G, seed)
    result = _dijkstra_fixed_edge_cost(inputs.G, source, target, costs)
    return _make_random_witness(
        inputs,
        pair_id,
        f"{pair_id}:source_a:{index:02d}:seed_{seed}",
        source,
        target,
        result,
        generator=RANDOM_COST_SOURCE_A,
        provenance={
            "source": "A",
            "random_seed": seed,
            "edge_cost_formula": "c_random(e)=l(e)*(0.2+3.0*U_e), U_e~Uniform(0,1)",
            "independence_note": "random field independent of elevation, popularity, width, Pmin, and Wmax",
        },
    )


def _concatenate_segments(
    inputs: base.StaticInputs,
    segment_results: Sequence[base.ScalarPathResult],
) -> base.ScalarPathResult | None:
    if not segment_results or any(not r.route_found or r.metrics is None for r in segment_results):
        return None
    nodes: list[int] = []
    edges: list[int] = []
    scalar_cost = 0.0
    for idx, result in enumerate(segment_results):
        if idx == 0:
            nodes.extend(int(v) for v in result.path_nodes)
        else:
            if nodes[-1] != int(result.path_nodes[0]):
                return None
            nodes.extend(int(v) for v in result.path_nodes[1:])
        edges.extend(int(edge_id) for edge_id in result.edge_ids)
        scalar_cost += float(result.scalar_cost)
    edge_tuple = tuple(edges)
    metrics = base._metrics_from_edge_ids(inputs.G, edge_tuple)
    return base.ScalarPathResult(
        True,
        scalar_cost,
        tuple(nodes),
        edge_tuple,
        metrics,
        base.DijkstraStats(0, 0, 0, 0, sum(float(r.stats.elapsed_s) for r in segment_results)),
    )


def _waypoint_candidates(
    inputs: base.StaticInputs,
    source: int,
    target: int,
    rng: random.Random,
    valid_nodes: Sequence[int],
    count: int,
) -> list[int]:
    xy = inputs.xy_int.astype(np.float64)
    sxy = xy[source]
    txy = xy[target]
    direction = txy - sxy
    length2 = float(np.dot(direction, direction))
    if length2 <= 0.0:
        return rng.sample(list(valid_nodes), min(count, len(valid_nodes)))
    candidates: list[tuple[float, int]] = []
    for node in rng.sample(list(valid_nodes), min(2000, len(valid_nodes))):
        if node in (source, target):
            continue
        rel = xy[node] - sxy
        frac = float(np.dot(rel, direction) / length2)
        if not (0.08 <= frac <= 0.92):
            continue
        projected = sxy + frac * direction
        lateral = float(np.linalg.norm(xy[node] - projected))
        candidates.append((lateral + rng.random() * 500.0, int(node)))
    if len(candidates) < count:
        extra = [node for node in valid_nodes if node not in (source, target)]
        rng.shuffle(extra)
        return [int(node) for _, node in sorted(candidates)[:count]] + extra[: count - len(candidates)]
    return [int(node) for _, node in sorted(candidates)[:count]]


def _source_b_witness(
    inputs: base.StaticInputs,
    valid_nodes: Sequence[int],
    pair_id: str,
    source: int,
    target: int,
    waypoint_seed: int,
    segment_seed_base: int,
    index: int,
) -> dict[str, Any] | None:
    rng = random.Random(int(waypoint_seed))
    waypoint_count = 1 + (index % 3)
    waypoints = _waypoint_candidates(inputs, source, target, rng, valid_nodes, waypoint_count)
    stops = [int(source), *waypoints, int(target)]
    segment_results: list[base.ScalarPathResult] = []
    segment_seeds: list[int] = []
    for segment_index, (a, b) in enumerate(zip(stops, stops[1:])):
        seed = int(segment_seed_base + 7919 * (segment_index + 1))
        segment_seeds.append(seed)
        costs = _random_costs(inputs.G, seed)
        segment_results.append(_dijkstra_fixed_edge_cost(inputs.G, a, b, costs))
    result = _concatenate_segments(inputs, segment_results)
    if result is None:
        return None
    return _make_random_witness(
        inputs,
        pair_id,
        f"{pair_id}:source_b:{index:02d}:wayseed_{waypoint_seed}:segbase_{segment_seed_base}",
        source,
        target,
        result,
        generator=RANDOM_COST_SOURCE_B,
        provenance={
            "source": "B",
            "waypoint_seed": waypoint_seed,
            "waypoint_count": waypoint_count,
            "waypoint_node_ids": waypoints,
            "segment_random_seeds": segment_seeds,
            "edge_cost_formula": "c_random(e)=l(e)*(0.2+3.0*U_e), U_e~Uniform(0,1)",
            "independence_note": "waypoints and segment costs are independent of all audited scalarizations",
        },
    )


def _make_box_query(
    pair_id: str,
    box_index: int,
    source: int,
    target: int,
    witness_metrics: base.RouteMetrics,
    shortest_metrics: base.RouteMetrics,
    category: str,
) -> tuple[base.QueryBox, list[str]]:
    L = witness_metrics.length
    H = witness_metrics.elevation
    avg_pop = witness_metrics.avg_popularity
    avg_width = witness_metrics.avg_width
    tags = [category]
    Lmin = max(0.0, 0.72 * L)
    Lmax = max(L + 1.0, 1.28 * L)
    Hmin = max(0.0, 0.55 * H)
    Hmax = max(H + 10.0, 1.38 * H + 5.0)
    Pmin = max(0.0, 0.82 * avg_pop)
    Wmax = max(avg_width + 0.1, 1.22 * avg_width)

    if category == "LMIN_ACTIVE":
        Lmin = max(0.0, 0.985 * L)
        if shortest_metrics.length < L:
            Lmin = max(Lmin, min(L - 1.0, shortest_metrics.length + 1.0))
    elif category == "LMAX_ACTIVE":
        Lmax = 1.015 * L
    elif category == "HMIN_ACTIVE":
        Hmin = max(0.0, 0.965 * H)
        if shortest_metrics.elevation < H:
            Hmin = max(Hmin, min(H - 0.1, shortest_metrics.elevation + 0.1))
    elif category == "HMAX_ACTIVE":
        Hmax = max(H + 0.2, 1.025 * H)
    elif category == "POPULARITY_ACTIVE":
        Pmin = 0.985 * avg_pop
        if shortest_metrics.avg_popularity < avg_pop:
            Pmin = max(Pmin, min(avg_pop - 1e-3, shortest_metrics.avg_popularity + 0.05))
    elif category == "WIDTH_ACTIVE":
        Wmax = max(avg_width + 1e-3, 1.015 * avg_width)
        if shortest_metrics.avg_width > avg_width:
            Wmax = min(Wmax, max(avg_width + 1e-3, shortest_metrics.avg_width - 0.05))
    elif category == "MULTI_TIGHT":
        Lmin = max(0.0, 0.96 * L)
        Lmax = 1.035 * L
        Hmin = max(0.0, 0.94 * H)
        Pmin = 0.975 * avg_pop
        Wmax = max(avg_width + 1e-3, 1.025 * avg_width)
        tags.extend(["LMIN_ACTIVE", "LMAX_ACTIVE", "HMIN_ACTIVE", "POPULARITY_ACTIVE", "WIDTH_ACTIVE"])
    elif category == "QUALITY_CONFLICT":
        Lmax = 1.08 * L
        Pmin = 0.99 * avg_pop
        Wmax = max(avg_width + 1e-3, 1.01 * avg_width)
        tags.extend(["POPULARITY_ACTIVE", "WIDTH_ACTIVE"])

    query = base.QueryBox(
        name=f"{pair_id}_{box_index:02d}_{category.lower()}",
        source=int(source),
        target=int(target),
        Lmin=float(Lmin),
        Lmax=float(max(Lmax, L + 1e-6)),
        Hmin=float(min(Hmin, H)),
        Hmax=float(max(Hmax, H + 1e-6)),
        Pmin=float(min(Pmin, avg_pop)),
        Wmax=float(max(Wmax, avg_width + 1e-6)),
    )
    return query, sorted(set(tags))


def _box_from_witness(
    pair_id: str,
    box_index: int,
    source: int,
    target: int,
    witness: dict[str, Any],
    physical_shortest: bench.WitnessRoute,
    category: str,
) -> dict[str, Any] | None:
    wm = base.RouteMetrics(
        float(witness["L"]),
        float(witness["H"]),
        float(witness["P_length"]),
        float(witness["W_length"]),
        int(witness["road_changes"]),
    )
    query, tags = _make_box_query(
        pair_id,
        box_index,
        source,
        target,
        wm,
        physical_shortest.metrics,
        category,
    )
    if not query.is_feasible(wm):
        return None
    shortest_score = query.normalized_violation_score(physical_shortest.metrics)
    shortest_violations = query.violations(physical_shortest.metrics)
    shortest_violated = [key for key, value in shortest_violations.items() if value > 1e-6]
    return {
        "query": query.as_dict(),
        "query_id": query.name,
        "pair_id": pair_id,
        "category": category,
        "tightness": "final_independent",
        "tags": tags,
        "witness": witness,
        "witness_source": witness["witness_generation_source"],
        "witness_independence": "independent random positive edge-cost source, not any audited scalarization",
        "physical_shortest_diagnostic": physical_shortest.as_dict(include_paths=True),
        "shortest_violation_score": shortest_score,
        "shortest_violations": shortest_violations,
        "shortest_violated_constraints": shortest_violated,
        "adversarial": shortest_score > 1e-6,
        "deliberate_adversarial": category in {"LMIN_ACTIVE", "HMIN_ACTIVE", "POPULARITY_ACTIVE", "WIDTH_ACTIVE", "MULTI_TIGHT", "QUALITY_CONFLICT"},
    }


def _generate_final_holdout(
    inputs: base.StaticInputs,
    prepared: bench.PreparedGraph,
    *,
    seed: int,
    pair_count: int,
    boxes_per_pair: int,
    development_boxes_json: str,
    validation_boxes_json: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_pairs = _load_existing_od_pairs([development_boxes_json, validation_boxes_json])
    valid_nodes = _valid_nodes(inputs)
    xy = inputs.xy_int.astype(np.float64)
    rng = random.Random(seed)
    categories = [
        "LMIN_ACTIVE",
        "LMAX_ACTIVE",
        "HMIN_ACTIVE",
        "HMAX_ACTIVE",
        "POPULARITY_ACTIVE",
        "WIDTH_ACTIVE",
        "MULTI_TIGHT",
        "QUALITY_CONFLICT",
    ]
    boxes: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    witness_seen: set[str] = set()
    direction_counts: Counter[str] = Counter()
    attempts = 0
    max_attempts = 900
    while len(selected_pairs) < pair_count and attempts < max_attempts:
        attempts += 1
        source = int(rng.choice(valid_nodes))
        candidates = []
        sxy = xy[source]
        for target in rng.sample(valid_nodes, min(900, len(valid_nodes))):
            target = int(target)
            if target == source or (source, target) in existing_pairs:
                continue
            euclid = float(np.linalg.norm(xy[target] - sxy))
            if 7000.0 <= euclid <= 24000.0:
                candidates.append((euclid, target))
        if not candidates:
            continue
        rng.shuffle(candidates)
        target = int(candidates[0][1])
        dx, dy = xy[target] - sxy
        direction = (
            "east_west" if abs(dx) >= 1.35 * abs(dy) else
            "south_north" if abs(dy) >= 1.35 * abs(dx) else
            "diagonal"
        )
        if direction_counts[direction] >= max(4, math.ceil(pair_count / 2)):
            continue
        pair_id = f"final_pair_{len(selected_pairs):02d}_{direction}_{source}_{target}"
        shortest = _physical_shortest(inputs, prepared, source, target, pair_id)
        if shortest is None:
            continue
        if not (10000.0 <= shortest.metrics.length <= 55000.0):
            continue

        witnesses: list[dict[str, Any]] = []
        witness_attempts = 0
        while len(witnesses) < boxes_per_pair and witness_attempts < 90:
            witness_attempts += 1
            global_index = len(selected_pairs) * boxes_per_pair + len(witnesses)
            if witness_attempts % 2 == 1:
                w_seed = seed * 1000003 + attempts * 101 + witness_attempts * 17
                witness = _source_a_witness(inputs, pair_id, source, target, w_seed, global_index)
            else:
                waypoint_seed = seed * 1000033 + attempts * 211 + witness_attempts * 19
                segment_seed_base = seed * 1000037 + attempts * 307 + witness_attempts * 23
                witness = _source_b_witness(
                    inputs,
                    valid_nodes,
                    pair_id,
                    source,
                    target,
                    waypoint_seed,
                    segment_seed_base,
                    global_index,
                )
            if witness is None:
                continue
            if witness["path_fingerprint"] in witness_seen:
                continue
            if not (shortest.metrics.length * 0.75 <= float(witness["L"]) <= 85000.0):
                continue
            witness_seen.add(witness["path_fingerprint"])
            witnesses.append(witness)
        if len(witnesses) < boxes_per_pair:
            continue

        pair_boxes: list[dict[str, Any]] = []
        for local_index, witness in enumerate(witnesses):
            category = categories[(len(selected_pairs) * boxes_per_pair + local_index) % len(categories)]
            box = _box_from_witness(
                pair_id,
                local_index,
                source,
                target,
                witness,
                shortest,
                category,
            )
            if box is None:
                break
            pair_boxes.append(box)
        if len(pair_boxes) != boxes_per_pair:
            continue

        selected_pairs.append(
            {
                "pair_id": pair_id,
                "source": source,
                "target": target,
                "direction_bucket": direction,
                "physical_shortest_metrics": shortest.metrics.as_dict(),
                "box_count": len(pair_boxes),
            }
        )
        direction_counts[direction] += 1
        boxes.extend(pair_boxes)
        existing_pairs.add((source, target))

    if len(selected_pairs) < pair_count:
        raise RuntimeError(
            f"only generated {len(selected_pairs)} OD pairs after {attempts} attempts; "
            "increase max attempts or relax distance filters"
        )
    generation = {
        "pair_selection_procedure": (
            "deterministic pseudo-random source/target sampling from nodes with in/out degree; "
            "OD pairs already present in development/validation are excluded; Euclidean and ordinary "
            "physical-shortest distance filters enforce the useful routing range; frozen K2 is not run"
        ),
        "seed": seed,
        "pair_count_target": pair_count,
        "boxes_per_pair": boxes_per_pair,
        "attempts": attempts,
        "existing_od_pairs_excluded": sorted([list(pair) for pair in existing_pairs]),
        "selected_pairs": selected_pairs,
        "direction_counts": dict(direction_counts),
        "witness_sources_used": [RANDOM_COST_SOURCE_A, RANDOM_COST_SOURCE_B],
        "source_c_used": False,
    }
    return boxes, generation


def _validate_frozen_boxes(
    inputs: base.StaticInputs,
    boxes_path: str,
) -> tuple[list[bench.BenchmarkItem], dict[str, Any]]:
    items, load_report = bench._load_items_from_boxes_json(inputs, boxes_path)
    failures = []
    for item in items:
        if not item.witness.validation.passed or not item.witness.elementary or not item.query.is_feasible(item.witness.metrics):
            failures.append(item.query.name)
    if failures:
        raise ValueError(f"frozen final holdout witness validation failed: {failures}")
    return items, load_report


def _row_metrics(prefix: str, route: dict[str, Any] | None) -> dict[str, Any]:
    metrics = None if route is None else route.get("metrics")
    return {
        f"{prefix}L": None if metrics is None else metrics.get("length"),
        f"{prefix}H": None if metrics is None else metrics.get("elevation"),
        f"{prefix}avg_pop": None if metrics is None else metrics.get("avg_pop"),
        f"{prefix}avg_width": None if metrics is None else metrics.get("avg_width"),
        f"{prefix}road_changes": None if metrics is None else metrics.get("road_changes"),
    }


def _method_row_from_scalar(
    item: bench.BenchmarkItem,
    method: str,
    scalar_row: dict[str, Any],
) -> dict[str, Any]:
    first = scalar_row.get("first_hit")
    return {
        "query_id": item.query.name,
        "pair_id": item.pair_id,
        "method": method,
        "solved": bool(scalar_row["feasible"]),
        "scalar_that_solved": scalar_row["scalar_name"] if scalar_row["feasible"] else None,
        "tree_pairs_evaluated": 1,
        "time_to_first_feasible_s": scalar_row.get("time_to_first_exact_feasible_s"),
        "exhaustive_time_s": scalar_row.get("exhaustive_time_s"),
        "profile_feasible_count": scalar_row.get("profile_feasible_count"),
        "exact_elementary_feasible_count": scalar_row.get("exact_elementary_feasible_count"),
        "non_elementary_rejection_count": scalar_row.get("non_elementary_count"),
        "validation_failure_count": scalar_row.get("validation_failure_count"),
        **_row_metrics("", first),
        "repeated_vertex_count": None if first is None else first.get("repeated_vertex_count"),
        "directed_validation_status": None if first is None else first.get("validation"),
        "nearest_profile": scalar_row.get("nearest_profile"),
        "nearest_exact_reconstructed_candidate": scalar_row.get("nearest_exact"),
        "normalized_violation": 0.0 if scalar_row["feasible"] else _nearest_score(scalar_row),
        "violated_constraints": "" if scalar_row["feasible"] else "|".join(_nearest_violated(scalar_row)),
        "witness_source": item.witness.generator,
        "category": item.category,
        "tags": "|".join(item.tags),
        "adversarial": item.adversarial,
    }


def _nearest_score(row: dict[str, Any]) -> float | None:
    nearest = row.get("nearest_exact") or row.get("nearest_profile")
    if nearest is None:
        return None
    return nearest.get("normalized_violation_score")


def _nearest_violated(row: dict[str, Any]) -> list[str]:
    nearest = row.get("nearest_exact") or row.get("nearest_profile")
    violations = None if nearest is None else nearest.get("violations")
    if not isinstance(violations, dict):
        return []
    return [key for key, value in violations.items() if float(value) > 1e-6]


def _union2_row(
    item: bench.BenchmarkItem,
    p_row: dict[str, Any],
    s_row: dict[str, Any],
) -> dict[str, Any]:
    members = [p_row, s_row]
    solved_members = [row for row in members if row["feasible"]]
    winner = None
    if solved_members:
        winner = min(
            solved_members,
            key=lambda row: (
                float(row.get("time_to_first_exact_feasible_s") or float("inf")),
                row["scalar_name"],
            ),
        )
    first = None if winner is None else winner.get("first_hit")
    nearest_rows = [row for row in members if row.get("nearest_exact") or row.get("nearest_profile")]
    nearest_row = None
    if nearest_rows:
        nearest_row = min(
            nearest_rows,
            key=lambda row: float(_nearest_score(row) if _nearest_score(row) is not None else float("inf")),
        )
    return {
        "query_id": item.query.name,
        "pair_id": item.pair_id,
        "method": BASELINE_UNION_2,
        "solved": winner is not None,
        "scalar_that_solved": None if winner is None else winner["scalar_name"],
        "tree_pairs_evaluated": 2,
        "time_to_first_feasible_s": None if winner is None else winner.get("time_to_first_exact_feasible_s"),
        "exhaustive_time_s": sum(float(row.get("exhaustive_time_s") or 0.0) for row in members),
        "profile_feasible_count": sum(int(row.get("profile_feasible_count") or 0) for row in members),
        "exact_elementary_feasible_count": sum(int(row.get("exact_elementary_feasible_count") or 0) for row in members),
        "non_elementary_rejection_count": sum(int(row.get("non_elementary_count") or 0) for row in members),
        "validation_failure_count": sum(int(row.get("validation_failure_count") or 0) for row in members),
        **_row_metrics("", first),
        "repeated_vertex_count": None if first is None else first.get("repeated_vertex_count"),
        "directed_validation_status": None if first is None else first.get("validation"),
        "nearest_profile": None if nearest_row is None else nearest_row.get("nearest_profile"),
        "nearest_exact_reconstructed_candidate": None if nearest_row is None else nearest_row.get("nearest_exact"),
        "normalized_violation": 0.0 if winner is not None else (None if nearest_row is None else _nearest_score(nearest_row)),
        "violated_constraints": "" if winner is not None or nearest_row is None else "|".join(_nearest_violated(nearest_row)),
        "witness_source": item.witness.generator,
        "category": item.category,
        "tags": "|".join(item.tags),
        "adversarial": item.adversarial,
    }


def _k2_row(
    item: bench.BenchmarkItem,
    hws_row: dict[str, Any],
    hllp_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if hws_row["feasible"]:
        executed = [hws_row]
        winner = hws_row
        first_time = hws_row.get("time_to_first_exact_feasible_s")
    else:
        assert hllp_row is not None
        executed = [hws_row, hllp_row]
        winner = hllp_row if hllp_row["feasible"] else None
        first_time = None
        if winner is not None:
            first_time = float(hws_row.get("exhaustive_time_s") or 0.0) + float(
                hllp_row.get("time_to_first_exact_feasible_s") or 0.0
            )
    first = None if winner is None else winner.get("first_hit")
    nearest_rows = [row for row in executed if row.get("nearest_exact") or row.get("nearest_profile")]
    nearest_row = None
    if nearest_rows:
        nearest_row = min(
            nearest_rows,
            key=lambda row: float(_nearest_score(row) if _nearest_score(row) is not None else float("inf")),
        )
    return {
        "query_id": item.query.name,
        "pair_id": item.pair_id,
        "method": FINAL_K2,
        "solved": winner is not None,
        "scalar_that_solved": None if winner is None else winner["scalar_name"],
        "tree_pairs_evaluated": len(executed),
        "time_to_first_feasible_s": first_time,
        "exhaustive_time_s": sum(float(row.get("exhaustive_time_s") or 0.0) for row in executed),
        "profile_feasible_count": sum(int(row.get("profile_feasible_count") or 0) for row in executed),
        "exact_elementary_feasible_count": sum(int(row.get("exact_elementary_feasible_count") or 0) for row in executed),
        "non_elementary_rejection_count": sum(int(row.get("non_elementary_count") or 0) for row in executed),
        "validation_failure_count": sum(int(row.get("validation_failure_count") or 0) for row in executed),
        **_row_metrics("", first),
        "repeated_vertex_count": None if first is None else first.get("repeated_vertex_count"),
        "directed_validation_status": None if first is None else first.get("validation"),
        "nearest_profile": None if nearest_row is None else nearest_row.get("nearest_profile"),
        "nearest_exact_reconstructed_candidate": None if nearest_row is None else nearest_row.get("nearest_exact"),
        "normalized_violation": 0.0 if winner is not None else (None if nearest_row is None else _nearest_score(nearest_row)),
        "violated_constraints": "" if winner is not None or nearest_row is None else "|".join(_nearest_violated(nearest_row)),
        "witness_source": item.witness.generator,
        "category": item.category,
        "tags": "|".join(item.tags),
        "adversarial": item.adversarial,
    }


def _evaluate_final_holdout(
    inputs: base.StaticInputs,
    prepared: bench.PreparedGraph,
    items: Sequence[bench.BenchmarkItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    reverse_adj = None
    for index, item in enumerate(items):
        print(f"[evaluate] {index + 1}/{len(items)} {item.query.name}", flush=True)
        if reverse_adj is None:
            reverse_adj = bench.via._build_reverse_edge_adjacency(prepared.G, prepared.context.edge_mask)  # type: ignore[attr-defined]
        scalar_cache: dict[str, dict[str, Any]] = {}
        for scalar_name in (CONTROL_PHYSICAL, bench.SCALAR_REFERENCE, bench.SCALAR_SLOPE, FINAL_SCALAR_1):
            spec = _spec_by_name(item.query, scalar_name)
            row = portfolio._run_same_scalar_via(prepared, inputs.G, item.query, spec, reverse_adj)
            scalar_cache[scalar_name] = row
            scalar_rows.append({"query_id": item.query.name, **row})

        rows.append(_method_row_from_scalar(item, CONTROL_PHYSICAL, scalar_cache[CONTROL_PHYSICAL]))
        rows.append(_union2_row(item, scalar_cache[bench.SCALAR_REFERENCE], scalar_cache[bench.SCALAR_SLOPE]))
        rows.append(_method_row_from_scalar(item, HWS_METHOD, scalar_cache[FINAL_SCALAR_1]))

        hllp_row = None
        if not scalar_cache[FINAL_SCALAR_1]["feasible"]:
            spec = _spec_by_name(item.query, FINAL_SCALAR_2)
            hllp_row = portfolio._run_same_scalar_via(prepared, inputs.G, item.query, spec, reverse_adj)
            scalar_rows.append({"query_id": item.query.name, **hllp_row})
        rows.append(_k2_row(item, scalar_cache[FINAL_SCALAR_1], hllp_row))
    return rows, scalar_rows


def _rates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    methods = sorted({row["method"] for row in rows})
    total = len({row["query_id"] for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        solved = sum(1 for row in subset if row["solved"])
        out[method] = {
            "solved": solved,
            "total": total,
            "percentage": 100.0 * solved / total if total else None,
        }
    return out


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    arr = np.array(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, q))


def _stratified(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            groups[str(row.get(field))].append(row)
        out[method] = {
            key: {
                "solved": sum(1 for row in group if row["solved"]),
                "total": len(group),
                "percentage": 100.0 * sum(1 for row in group if row["solved"]) / len(group),
            }
            for key, group in sorted(groups.items())
        }
    return out


def _final_report(
    boxes: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    generation: dict[str, Any],
    existing_original_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    pair_ids = sorted({box["pair_id"] for box in boxes})
    od_pairs = sorted({(int(box["query"]["source"]), int(box["query"]["target"])) for box in boxes})
    witness_sources = Counter(str(box["witness_source"]) for box in boxes)
    k2 = [row for row in rows if row["method"] == FINAL_K2]
    k2_times = [float(row["time_to_first_feasible_s"]) for row in k2 if row["solved"] and row["time_to_first_feasible_s"] is not None]
    hws_solved = sum(1 for row in k2 if row["scalar_that_solved"] == FINAL_SCALAR_1)
    hllp_solved = sum(1 for row in k2 if row["scalar_that_solved"] == FINAL_SCALAR_2)
    failures = []
    by_query_box = {box["query_id"]: box for box in boxes}
    for row in k2:
        if row["solved"]:
            continue
        box = by_query_box[row["query_id"]]
        failures.append(
            {
                "query_id": row["query_id"],
                "pair_id": row["pair_id"],
                "witness_metrics": box["witness"]["metrics"],
                "witness_provenance": box["witness"]["provenance"],
                "nearest_profile": row["nearest_profile"],
                "nearest_exact_reconstructed_candidate": row["nearest_exact_reconstructed_candidate"],
                "violated_constraints": row["violated_constraints"],
                "normalized_violation": row["normalized_violation"],
                "profile_feasible_count": row["profile_feasible_count"],
            }
        )
    return {
        "final_holdout_size": {
            "total_boxes": len(boxes),
            "distinct_od_pairs": len(od_pairs),
            "od_pairs": [list(pair) for pair in od_pairs],
            "all_od_pairs_new_vs_development_and_validation": all(pair not in existing_original_pairs for pair in od_pairs),
        },
        "scientific_status": {
            "24_original_boxes": "DEVELOPMENT SET; not independent generalization evidence",
            "12_former_holdout_boxes": "VALIDATION / ARCHITECTURE-SELECTION SET; not independent generalization evidence",
            "new_boxes": "FINAL INDEPENDENT HOLDOUT",
        },
        "witness_independence": {
            "counts_by_source": dict(witness_sources),
            "random_seed": generation["seed"],
            "source_c_used": generation.get("source_c_used", False),
            "validation_status": "all frozen witnesses directed-valid, strictly elementary, and exact-box-feasible",
            "no_audited_scalarization_generated_any_witness": True,
        },
        "success_rates": _rates(rows),
        "frozen_k2_sequential_details": {
            "solved_by_hinge_width_strong": hws_solved,
            "requiring_hinge_low_length_pressure_and_solved": hllp_solved,
            "still_unsolved": sum(1 for row in k2 if not row["solved"]),
            "median_time_to_first_feasible_s": statistics.median(k2_times) if k2_times else None,
            "p90_time_to_first_feasible_s": _percentile(k2_times, 90),
            "max_time_to_first_feasible_s": max(k2_times) if k2_times else None,
            "mean_tree_pairs_evaluated_before_success": (
                statistics.mean(float(row["tree_pairs_evaluated"]) for row in k2 if row["solved"])
                if any(row["solved"] for row in k2)
                else None
            ),
        },
        "stratified_results": {
            "by_witness_generation_source": _stratified(rows, "witness_source"),
            "by_constraint_category": _stratified(rows, "category"),
            "by_od_pair": _stratified(rows, "pair_id"),
        },
        "frozen_k2_failures": failures,
        "pair_selection": generation,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final independent holdout for frozen scalar-via architecture.")
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument("--development-boxes-json", default=DEFAULT_DEVELOPMENT_BOXES_JSON)
    parser.add_argument("--validation-boxes-json", default=DEFAULT_VALIDATION_BOXES_JSON)
    parser.add_argument("--config-json", default=DEFAULT_CONFIG_JSON)
    parser.add_argument("--boxes-json", default=DEFAULT_BOXES_JSON)
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--pair-count", type=int, default=10)
    parser.add_argument("--boxes-per-pair", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    config = _frozen_config(args.seed)
    _write_json(args.config_json, config)
    print(
        f"[phase 0] wrote frozen config {args.config_json} sha256={config['frozen_configuration_sha256']}",
        flush=True,
    )

    inputs = base._load_static_inputs(args)
    prepared = _full_prepared(inputs)
    existing_pairs_before = _load_existing_od_pairs([args.development_boxes_json, args.validation_boxes_json])
    boxes, generation = _generate_final_holdout(
        inputs,
        prepared,
        seed=args.seed,
        pair_count=args.pair_count,
        boxes_per_pair=args.boxes_per_pair,
        development_boxes_json=args.development_boxes_json,
        validation_boxes_json=args.validation_boxes_json,
    )
    boxes_payload = {
        "metadata": {
            "script": Path(__file__).name,
            "status": "FINAL INDEPENDENT HOLDOUT",
            "frozen_configuration_sha256": config["frozen_configuration_sha256"],
            "frozen_configuration_path": args.config_json,
            "generation_seed": args.seed,
            "policy": "complete holdout JSON is written before any candidate evaluation",
            "development_boxes_json": args.development_boxes_json,
            "validation_boxes_json": args.validation_boxes_json,
        },
        "generation": generation,
        "boxes": boxes,
    }
    _write_json(args.boxes_json, boxes_payload)
    content_hash_before_hash_metadata = _file_sha256(args.boxes_json)
    boxes_payload["metadata"]["content_sha256_before_hash_metadata"] = content_hash_before_hash_metadata
    _write_json(args.boxes_json, boxes_payload)
    final_boxes_hash = _file_sha256(args.boxes_json)
    print(f"[phase 3] wrote frozen final holdout {args.boxes_json} sha256={final_boxes_hash}", flush=True)

    items, load_report = _validate_frozen_boxes(inputs, args.boxes_json)
    rows, scalar_rows = _evaluate_final_holdout(inputs, prepared, items)
    report = _final_report(boxes, rows, generation, existing_pairs_before)
    results_payload = {
        "metadata": {
            "script": Path(__file__).name,
            "elapsed_s": time.perf_counter() - start,
            "graph_mode": "full",
            "boxes_json": args.boxes_json,
            "boxes_json_sha256": _file_sha256(args.boxes_json),
            "frozen_configuration": config,
            "load_report": load_report,
            "method_policy": "predeclared methods only; no tuning after holdout freeze",
        },
        "final_report": report,
        "method_query_rows": rows,
        "scalar_query_rows": scalar_rows,
    }
    _write_json(args.results_json, results_payload)
    _write_csv(args.results_csv, rows)
    print(f"[phase 5] wrote {args.results_json} and {args.results_csv}", flush=True)
    for method, rate in report["success_rates"].items():
        print(
            f"  {method}: {rate['solved']}/{rate['total']} ({rate['percentage']:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
