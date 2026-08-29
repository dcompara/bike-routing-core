from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import article_scalar_feasibility_experiment as base
import article_scalar_via_benchmark as bench
import article_scalar_via_feasibility_experiment as via


DEFAULT_HOLDOUT_BOXES_JSON = "tmp_scalar_via_holdout_boxes.json"
DEFAULT_HOLDOUT_RESULTS_JSON = "tmp_scalar_via_holdout_results.json"
DEFAULT_HOLDOUT_RESULTS_CSV = "tmp_scalar_via_holdout_results.csv"
DEFAULT_DEVELOPMENT_RESULTS_JSON = "tmp_scalar_via_benchmark.json"

HOLDOUT_LABEL = "independent_holdout"
FROZEN_EVALUATED_SCALARS = set(bench.BENCHMARK_SCALAR_NAMES)
FROZEN_EVALUATED_SPEC_NAMES = {
    bench.SCALAR_PHYSICAL,
    "shortest_length",
    bench.SCALAR_REFERENCE,
    bench.SCALAR_SLOPE,
}
FROZEN_METHODS = (
    "physical_length direct",
    "physical_length via",
    "pop_width_reference direct",
    "pop_width_reference via",
    "slope_exp_beta_150_width direct",
    "slope_exp_beta_150_width via",
    bench.UNION_2_NAME,
    bench.UNION_3_NAME,
)

INDEPENDENT_SPEC_NAMES = (
    "hinge_width_strong",
    "hinge_width_very_strong",
    "hinge_pop_strong",
    "hinge_pop_very_strong",
    "hinge_balanced_strong",
    "hinge_low_length_pressure",
    "hinge_medium_length_pressure",
    "width_linear_mild",
    "width_linear_strong",
    "pop_complement_mild",
    "pop_complement_strong",
    "low_elevation_mild",
    "low_elevation_strong",
    "low_elevation_with_width_pop",
    "high_elevation_width",
    "high_elevation_pop",
    "high_elevation_width_pop",
    "slope_inverse_beta_5",
    "slope_inverse_beta_12_width_pop",
    "slope_exp_beta_250_width_pop_light",
    "slope_exp_beta_8",
    "slope_exp_beta_400_width_light",
    "slope_exp_beta_16_width_pop",
)

TARGET_CATEGORY_SEQUENCE = (
    "L_LOW_AND_H_LOW",
    "MULTI_TIGHT",
    "POP_ACTIVE",
    "WIDTH_ACTIVE",
    "H_HIGH_ACTIVE",
    "QUALITY_CONFLICT",
    "LOOSE_CONTROL",
    "L_LOW_AND_H_LOW",
    "MULTI_TIGHT",
    "POP_ACTIVE",
    "WIDTH_ACTIVE",
    "LOOSE_CONTROL",
    "H_HIGH_ACTIVE",
    "QUALITY_CONFLICT",
    "MULTI_TIGHT",
)


@dataclass(frozen=True)
class HoldoutCandidate:
    item: bench.BenchmarkItem
    witness_source: str
    witness_method: str
    witness_creation_time: str
    witness_historical_file: str | None
    independence_note: str
    box_template: str

    def as_box_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        out = self.item.as_dict(include_paths=include_paths)
        out["benchmark_label"] = HOLDOUT_LABEL
        out["category"] = self.item.category
        out["tightness"] = self.item.tightness
        out["witness_source"] = self.witness_source
        out["witness_method"] = self.witness_method
        out["witness_creation_time"] = self.witness_creation_time
        out["witness_historical_file"] = self.witness_historical_file
        out["independence_note"] = self.independence_note
        out["box_template"] = self.box_template
        witness = dict(out["witness"])
        witness["witness_source"] = self.witness_source
        witness["witness_method"] = self.witness_method
        witness["witness_creation_time"] = self.witness_creation_time
        witness["witness_historical_file"] = self.witness_historical_file
        witness["independence_note"] = self.independence_note
        out["witness"] = witness
        return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _full_prepared(
    inputs: base.StaticInputs,
    global_constants: base.MetricConstants,
) -> bench.PreparedGraph:
    full_mask = np.ones(inputs.G.n_edges, dtype=bool)
    return bench.PreparedGraph(
        mode="full",
        G=inputs.G,
        context=base.GraphContext(
            mode="full",
            edge_mask=full_mask,
            node_count=inputs.G.n_nodes,
            edge_count=inputs.G.n_edges,
            constants=global_constants,
            metadata={"rho_H_policy": "global_full_graph_reused_across_modes"},
        ),
        edge_id_to_original=np.arange(inputs.G.n_edges, dtype=np.int32),
        graph_prep_s=0.0,
        corridor_construction_s=0.0,
        compaction_s=0.0,
        metadata={"graph_storage": "native_full_csr"},
    )


def _is_independent_spec(spec: base.ScalarizationSpec) -> bool:
    return (
        spec.family != "physical_length"
        and spec.name not in FROZEN_EVALUATED_SPEC_NAMES
        and spec.name in INDEPENDENT_SPEC_NAMES
    )


def _independent_specs(query: base.QueryBox) -> list[base.ScalarizationSpec]:
    by_name = {spec.name: spec for spec in base._fixed_portfolio(query)}
    specs = [
        by_name[name]
        for name in INDEPENDENT_SPEC_NAMES
        if name in by_name and _is_independent_spec(by_name[name])
    ]
    if not specs:
        raise RuntimeError("no independent non-frozen scalarizations are available")
    return specs


def _holdout_probe_query(pair: bench.PairSpec) -> base.QueryBox:
    shortest = pair.shortest.metrics
    return base.QueryBox(
        name=f"{pair.pair_id}_holdout_probe",
        source=pair.source,
        target=pair.target,
        Lmin=0.0,
        Lmax=max(65000.0, shortest.length * 1.75 + 6000.0),
        Hmin=0.0,
        Hmax=max(950.0, shortest.elevation * 2.8 + 250.0),
        Pmin=175.0,
        Wmax=14.5,
    )


def _edge_ids_from_node_path(
    G: base.CompactDiGraph | Any,
    nodes: Sequence[int],
) -> tuple[int, ...] | None:
    if len(nodes) < 2:
        return ()
    edge_ids: list[int] = []
    for u_raw, v_raw in zip(nodes, nodes[1:]):
        u = int(u_raw)
        v = int(v_raw)
        if not (0 <= u < G.n_nodes):
            return None
        start = int(G.offsets[u])
        end = int(G.offsets[u + 1])
        candidates = [
            edge_id
            for edge_id in range(start, end)
            if int(G.to[edge_id]) == v
        ]
        if not candidates:
            return None
        edge_ids.append(
            min(candidates, key=lambda edge_id: (float(G.w[edge_id, 0]), edge_id))
        )
    return tuple(edge_ids)


def _historical_paris_witnesses(inputs: base.StaticInputs) -> list[HoldoutCandidate]:
    path = Path("tmp_paris_bures_join_audit.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    entries: list[tuple[str, dict[str, Any]]] = []
    for run_index, run in enumerate(payload.get("runs", [])):
        for closest_index, entry in enumerate(run.get("closest", [])):
            if isinstance(entry.get("path_nodes"), list):
                entries.append((f"runs[{run_index}].closest[{closest_index}]", entry))

    out: list[HoldoutCandidate] = []
    seen_paths: set[tuple[int, ...]] = set()
    for ref, entry in entries:
        nodes = tuple(int(node) for node in entry["path_nodes"])
        if not nodes or nodes[0] != base.PARIS_BURES_SOURCE or nodes[-1] != base.PARIS_BURES_TARGET:
            continue
        if nodes in seen_paths:
            continue
        seen_paths.add(nodes)
        edge_ids = _edge_ids_from_node_path(inputs.G, nodes)
        if edge_ids is None:
            continue
        result = bench._result_from_original_edges(
            inputs.G,
            base.PARIS_BURES_SOURCE,
            base.PARIS_BURES_TARGET,
            edge_ids,
        )
        if result is None:
            continue
        witness = bench._make_witness_from_result(
            route_id=f"historical_join_audit:{ref}",
            pair_id="paris_bures",
            generator="historical_portal_join_audit_node_path",
            scalar_name="historical_non_frozen",
            via_vertex=None,
            original_G=inputs.G,
            result=result,
        )
        if witness is None:
            continue
        pair = _historical_paris_pair(inputs)
        # Candidate boxes are filled later; this placeholder is not emitted.
        query, category, tightness, tags, template = _holdout_query_from_template(
            pair=pair,
            witness=witness,
            template="QUALITY_CONFLICT",
            index=len(out) + 1,
        )
        item = _make_holdout_item(
            inputs=inputs,
            pair=pair,
            witness=witness,
            query=query,
            category=category,
            tightness=tightness,
            tags=tags,
            deliberate_adversarial=False,
        )
        if item is None:
            continue
        out.append(
            HoldoutCandidate(
                item=item,
                witness_source="historical_experiment_output",
                witness_method="portal_join_audit_node_path_reconstructed_on_base_csr",
                witness_creation_time="historical_file_timestamp_unknown",
                witness_historical_file=f"{path}:{ref}",
                independence_note=(
                    "Historical route path from join-audit output, not generated "
                    "by the frozen three direct/same-scalar-via benchmark methods."
                ),
                box_template=template,
            )
        )
        if len(out) >= 6:
            break
    return out


def _historical_paris_pair(inputs: base.StaticInputs) -> bench.PairSpec:
    full = _full_prepared(inputs, base._metric_constants(inputs.G, np.ones(inputs.G.n_edges, dtype=bool)))
    shortest = bench._direct_witness_for_pair(
        inputs,
        full,
        "paris_bures",
        base.PARIS_BURES_SOURCE,
        base.PARIS_BURES_TARGET,
        bench._physical_spec(),
        route_id_suffix="ordinary_shortest_for_holdout_diagnostic",
    )
    if shortest is None:
        raise RuntimeError("Paris-Bures ordinary shortest path not found")
    return bench.PairSpec(
        pair_id="paris_bures",
        source=base.PARIS_BURES_SOURCE,
        target=base.PARIS_BURES_TARGET,
        shortest=shortest,
        selection_note="historical Paris-Bures pair",
    )


def _collect_non_frozen_witnesses(
    inputs: base.StaticInputs,
    full_prepared: bench.PreparedGraph,
    pair: bench.PairSpec,
    *,
    max_profiles_per_scalar: int,
    max_witnesses_per_pair: int,
) -> tuple[list[bench.WitnessRoute], dict[str, Any]]:
    probe = _holdout_probe_query(pair)
    witnesses: dict[tuple[int, ...], bench.WitnessRoute] = {}
    rejected: list[dict[str, Any]] = []
    generation_runs: list[dict[str, Any]] = []

    specs = _independent_specs(probe)
    for spec in specs:
        direct_start = time.perf_counter()
        result = base._dijkstra_scalar_path(full_prepared.G, probe, full_prepared.context, spec)
        original_result = bench._map_result_to_original(full_prepared, inputs.G, result)
        witness = bench._make_witness_from_result(
            route_id=f"{pair.pair_id}:holdout_non_frozen_direct:{spec.name}",
            pair_id=pair.pair_id,
            generator="non_frozen_scalar_direct",
            scalar_name=spec.name,
            via_vertex=None,
            original_G=inputs.G,
            result=original_result,
        )
        if witness is None:
            rejected.append(
                {
                    "pair_id": pair.pair_id,
                    "source": "non_frozen_scalar_direct",
                    "scalar_name": spec.name,
                    "reason": "no_valid_elementary_route",
                }
            )
        elif witness.edge_ids == pair.shortest.edge_ids:
            rejected.append(
                {
                    "pair_id": pair.pair_id,
                    "source": "non_frozen_scalar_direct",
                    "scalar_name": spec.name,
                    "reason": "identical_to_ordinary_physical_shortest",
                }
            )
        else:
            witnesses.setdefault(witness.edge_ids, witness)
        generation_runs.append(
            {
                "pair_id": pair.pair_id,
                "source": "non_frozen_scalar_direct",
                "scalar_name": spec.name,
                "elapsed_s": time.perf_counter() - direct_start,
                "accepted_so_far": len(witnesses),
            }
        )

    reverse_adj = via._build_reverse_edge_adjacency(
        full_prepared.G,
        full_prepared.context.edge_mask,
    )
    for spec in specs:
        if len(witnesses) >= max_witnesses_per_pair:
            break
        start = time.perf_counter()
        forward = via._run_scalar_tree(
            full_prepared.G,
            probe,
            full_prepared.context,
            spec,
            reverse=False,
        )
        backward = via._run_scalar_tree(
            full_prepared.G,
            probe,
            full_prepared.context,
            spec,
            reverse=True,
            reverse_adj=reverse_adj,
        )
        profiles: list[via.ProfileCandidate] = []
        for via_vertex in range(full_prepared.G.n_nodes):
            metrics = via._combined_profile(probe, forward, backward, via_vertex)
            if metrics is None:
                continue
            if metrics.length <= probe.Lmax + 1e-6 and metrics.length >= 0.75 * pair.shortest.metrics.length:
                profiles.append(via.ProfileCandidate(via_vertex, metrics))
        selected: dict[int, via.ProfileCandidate] = {}

        def add(chosen: Iterable[via.ProfileCandidate], limit: int) -> None:
            added = 0
            for profile in chosen:
                if len(selected) >= max_profiles_per_scalar:
                    return
                before = len(selected)
                selected.setdefault(int(profile.via_vertex), profile)
                if len(selected) > before:
                    added += 1
                    if added >= limit:
                        return

        if profiles:
            add(sorted(profiles, key=lambda p: (-p.metrics.elevation, p.metrics.length)), 8)
            add(sorted(profiles, key=lambda p: (p.metrics.avg_width, p.metrics.length)), 8)
            add(sorted(profiles, key=lambda p: (-p.metrics.avg_popularity, p.metrics.length)), 8)
            add(
                sorted(
                    profiles,
                    key=lambda p: (
                        -bench._route_difference_score(pair.shortest.metrics, p.metrics),
                        p.metrics.length,
                    ),
                ),
                10,
            )
            add(
                sorted(
                    profiles,
                    key=lambda p: (
                        abs(p.metrics.length - 1.25 * pair.shortest.metrics.length),
                        -p.metrics.elevation,
                        p.metrics.avg_width,
                    ),
                ),
                8,
            )
        reconstructed = 0
        accepted = 0
        for profile in selected.values():
            if len(witnesses) >= max_witnesses_per_pair:
                break
            reconstructed += 1
            attempt = bench._reconstruct_via_candidate(
                full_prepared,
                inputs.G,
                probe,
                forward,
                backward,
                profile,
            )
            if attempt.candidate is None or not attempt.candidate.elementary:
                continue
            result = via._make_path_result(
                attempt.candidate.scalar_cost,
                attempt.candidate.path_nodes,
                attempt.candidate.edge_ids,
                attempt.candidate.metrics,
            )
            witness = bench._make_witness_from_result(
                route_id=(
                    f"{pair.pair_id}:holdout_non_frozen_via:{spec.name}:"
                    f"via{attempt.candidate.via_vertex}"
                ),
                pair_id=pair.pair_id,
                generator="non_frozen_same_scalar_via",
                scalar_name=spec.name,
                via_vertex=attempt.candidate.via_vertex,
                original_G=inputs.G,
                result=result,
            )
            if witness is None:
                continue
            if witness.edge_ids == pair.shortest.edge_ids:
                continue
            if witness.edge_ids not in witnesses:
                witnesses[witness.edge_ids] = witness
                accepted += 1
        generation_runs.append(
            {
                "pair_id": pair.pair_id,
                "source": "non_frozen_same_scalar_via",
                "scalar_name": spec.name,
                "profiles_considered": len(profiles),
                "profiles_selected_for_reconstruction": len(selected),
                "profiles_reconstructed": reconstructed,
                "witnesses_accepted": accepted,
                "elapsed_s": time.perf_counter() - start,
                **bench._tie_totals(forward, backward),
            }
        )

    out = list(witnesses.values())
    out.sort(
        key=lambda witness: (
            -bench._route_difference_score(pair.shortest.metrics, witness.metrics),
            witness.generator,
            witness.scalar_name,
            witness.route_id,
        )
    )
    return out, {
        "probe_query": probe.as_dict(),
        "independent_scalar_names": [spec.name for spec in specs],
        "generation_runs": generation_runs,
        "rejections": rejected,
        "witness_count": len(out),
    }


def _clamped_box(
    witness: base.RouteMetrics,
    *,
    Lmin: float,
    Lmax: float,
    Hmin: float,
    Hmax: float,
    Pmin: float,
    Wmax: float,
) -> tuple[float, float, float, float, float, float]:
    Lmin = min(float(Lmin), witness.length - 1e-3)
    Lmax = max(float(Lmax), witness.length + 1e-3)
    Hmin = min(float(Hmin), witness.elevation - 1e-3)
    Hmax = max(float(Hmax), witness.elevation + 1e-3)
    Pmin = min(max(0.0, float(Pmin)), witness.avg_popularity - 1e-6)
    Wmax = max(float(Wmax), witness.avg_width + 1e-6)
    return max(0.0, Lmin), Lmax, max(0.0, Hmin), Hmax, Pmin, Wmax


def _holdout_query_from_template(
    *,
    pair: bench.PairSpec,
    witness: bench.WitnessRoute,
    template: str,
    index: int,
) -> tuple[base.QueryBox, str, str, tuple[str, ...], str]:
    m = witness.metrics
    if template == "LOOSE_CONTROL":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1800.0, 0.12 * m.length),
            Lmax=m.length + max(3000.0, 0.16 * m.length),
            Hmin=m.elevation - max(110.0, 0.35 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(160.0, 0.38 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 35.0,
            Wmax=m.avg_width + 3.5,
        )
        category, tightness, tags = "LOOSE_CONTROL", "loose", ("loose",)
    elif template == "L_LOW_AND_H_LOW":
        values = _clamped_box(
            m,
            Lmin=m.length - max(120.0, 0.012 * m.length),
            Lmax=m.length + max(1800.0, 0.10 * m.length),
            Hmin=m.elevation - max(5.0, 0.020 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(85.0, 0.22 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 22.0,
            Wmax=m.avg_width + 2.1,
        )
        category = "L_LOW_AND_H_LOW"
        tightness = "tight"
        tags = ("L_LOW_ACTIVE", "H_LOW_ACTIVE", "L_LOW_AND_H_LOW", "tight")
    elif template == "MULTI_TIGHT":
        values = _clamped_box(
            m,
            Lmin=m.length - max(220.0, 0.035 * m.length),
            Lmax=m.length + max(250.0, 0.042 * m.length),
            Hmin=m.elevation - max(18.0, 0.070 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(20.0, 0.080 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 5.0,
            Wmax=m.avg_width + 0.65,
        )
        category = "MULTI_TIGHT"
        tightness = "tight"
        tags = ("MULTI_TIGHT", "POP_ACTIVE", "WIDTH_ACTIVE", "tight")
    elif template == "POP_ACTIVE":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1400.0, 0.085 * m.length),
            Lmax=m.length + max(2800.0, 0.14 * m.length),
            Hmin=m.elevation - max(90.0, 0.26 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(150.0, 0.35 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 2.5,
            Wmax=m.avg_width + 2.4,
        )
        category = "POP_ACTIVE"
        tightness = "tight"
        tags = ("POP_ACTIVE", "tight")
    elif template == "WIDTH_ACTIVE":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1400.0, 0.085 * m.length),
            Lmax=m.length + max(2800.0, 0.14 * m.length),
            Hmin=m.elevation - max(90.0, 0.26 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(150.0, 0.35 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 25.0,
            Wmax=m.avg_width + 0.35,
        )
        category = "WIDTH_ACTIVE"
        tightness = "tight"
        tags = ("WIDTH_ACTIVE", "tight")
    elif template == "H_HIGH_ACTIVE":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1300.0, 0.085 * m.length),
            Lmax=m.length + max(2600.0, 0.14 * m.length),
            Hmin=m.elevation - max(160.0, 0.45 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(7.0, 0.024 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 24.0,
            Wmax=m.avg_width + 2.2,
        )
        category = "H_HIGH_ACTIVE"
        tightness = "tight"
        tags = ("H_HIGH_ACTIVE", "tight")
    elif template == "QUALITY_CONFLICT":
        values = _clamped_box(
            m,
            Lmin=m.length - max(1100.0, 0.070 * m.length),
            Lmax=m.length + max(2300.0, 0.12 * m.length),
            Hmin=m.elevation - max(75.0, 0.20 * max(m.elevation, 1.0)),
            Hmax=m.elevation + max(130.0, 0.30 * max(m.elevation, 1.0)),
            Pmin=m.avg_popularity - 3.5,
            Wmax=m.avg_width + 0.50,
        )
        category = "QUALITY_CONFLICT"
        tightness = "tight"
        tags = ("QUALITY_CONFLICT", "POP_ACTIVE", "WIDTH_ACTIVE", "tight")
    else:
        raise ValueError(template)

    Lmin, Lmax, Hmin, Hmax, Pmin, Wmax = values
    query = base.QueryBox(
        name=f"holdout_{pair.pair_id}_{index:02d}_{template.lower()}",
        source=pair.source,
        target=pair.target,
        Lmin=Lmin,
        Lmax=Lmax,
        Hmin=Hmin,
        Hmax=Hmax,
        Pmin=Pmin,
        Wmax=Wmax,
    )
    return query, category, tightness, tags, template


def _make_holdout_item(
    *,
    inputs: base.StaticInputs,
    pair: bench.PairSpec,
    witness: bench.WitnessRoute,
    query: base.QueryBox,
    category: str,
    tightness: str,
    tags: tuple[str, ...],
    deliberate_adversarial: bool,
) -> bench.BenchmarkItem | None:
    item = bench.BenchmarkItem(
        query=query,
        pair_id=pair.pair_id,
        category=category,
        tightness=tightness,
        tags=tags,
        witness=witness,
        physical_shortest=pair.shortest,
        shortest_violation_score=0.0,
        shortest_violations={},
        shortest_violated_constraints=(),
        adversarial=False,
        deliberate_adversarial=deliberate_adversarial,
    )
    try:
        return bench._validate_witness_for_query(inputs, item)
    except ValueError:
        return None


def _holdout_candidates_for_pair(
    inputs: base.StaticInputs,
    pair: bench.PairSpec,
    witnesses: Sequence[bench.WitnessRoute],
    *,
    creation_time: str,
    witness_source: str,
    witness_method_prefix: str,
    historical_file: str | None = None,
) -> tuple[list[HoldoutCandidate], list[dict[str, Any]]]:
    candidates: list[HoldoutCandidate] = []
    rejections: list[dict[str, Any]] = []
    index = 0
    for witness in witnesses:
        if witness.scalar_name in FROZEN_EVALUATED_SCALARS:
            rejections.append(
                {
                    "route_id": witness.route_id,
                    "reason": "frozen_evaluated_scalar_name",
                    "scalar_name": witness.scalar_name,
                }
            )
            continue
        if witness.generator in {"direct_scalar_path", "same_scalar_via_profile_probe"}:
            rejections.append(
                {
                    "route_id": witness.route_id,
                    "reason": "development_generator_not_allowed_for_holdout",
                    "generator": witness.generator,
                }
            )
            continue
        for template in TARGET_CATEGORY_SEQUENCE[:7]:
            index += 1
            query, category, tightness, tags, template_name = _holdout_query_from_template(
                pair=pair,
                witness=witness,
                template=template,
                index=index,
            )
            item = _make_holdout_item(
                inputs=inputs,
                pair=pair,
                witness=witness,
                query=query,
                category=category,
                tightness=tightness,
                tags=tags,
                deliberate_adversarial=(
                    template_name in {"L_LOW_AND_H_LOW", "MULTI_TIGHT"}
                ),
            )
            if item is None:
                rejections.append(
                    {
                        "route_id": witness.route_id,
                        "query_id": query.name,
                        "reason": "witness_failed_box_validation",
                    }
                )
                continue
            candidates.append(
                HoldoutCandidate(
                    item=item,
                    witness_source=witness_source,
                    witness_method=f"{witness_method_prefix}:{witness.generator}:{witness.scalar_name}",
                    witness_creation_time=creation_time,
                    witness_historical_file=historical_file,
                    independence_note=(
                        "Witness route was generated or recovered without using "
                        "physical_length, pop_width_reference, or "
                        "slope_exp_beta_150_width in direct or same-scalar via mode."
                    ),
                    box_template=template_name,
                )
            )
    return candidates, rejections


def _candidate_sort_key(
    candidate: HoldoutCandidate,
    selected_by_pair: dict[str, int],
    selected_witness_counts: dict[str, int],
) -> tuple[Any, ...]:
    item = candidate.item
    return (
        selected_by_pair.get(item.pair_id, 0),
        selected_witness_counts.get(item.witness.route_id, 0),
        item.pair_id == "paris_bures",
        item.category != "LOOSE_CONTROL",
        -item.shortest_violation_score,
        item.query.name,
    )


def _select_holdout_candidates(
    candidates: Sequence[HoldoutCandidate],
    *,
    holdout_size: int,
    max_paris_items: int,
    max_boxes_per_witness: int,
) -> list[HoldoutCandidate]:
    selected: list[HoldoutCandidate] = []
    selected_names: set[str] = set()
    selected_by_pair: dict[str, int] = {}
    selected_witness_counts: dict[str, int] = {}

    def can_add(candidate: HoldoutCandidate) -> bool:
        if candidate.item.query.name in selected_names:
            return False
        if selected_witness_counts.get(candidate.item.witness.route_id, 0) >= max_boxes_per_witness:
            return False
        if candidate.item.pair_id == "paris_bures" and selected_by_pair.get("paris_bures", 0) >= max_paris_items:
            return False
        return True

    def add(candidate: HoldoutCandidate) -> bool:
        if not can_add(candidate):
            return False
        selected.append(candidate)
        selected_names.add(candidate.item.query.name)
        selected_by_pair[candidate.item.pair_id] = selected_by_pair.get(candidate.item.pair_id, 0) + 1
        selected_witness_counts[candidate.item.witness.route_id] = selected_witness_counts.get(candidate.item.witness.route_id, 0) + 1
        return True

    category_targets = TARGET_CATEGORY_SEQUENCE[:holdout_size]
    for category in category_targets:
        options = [
            candidate
            for candidate in candidates
            if candidate.item.category == category and can_add(candidate)
        ]
        if not options:
            continue
        add(
            min(
                options,
                key=lambda c: _candidate_sort_key(
                    c,
                    selected_by_pair,
                    selected_witness_counts,
                ),
            )
        )
        if len(selected) >= holdout_size:
            return selected

    ordered = sorted(
        candidates,
        key=lambda c: _candidate_sort_key(c, selected_by_pair, selected_witness_counts),
    )
    for candidate in ordered:
        if len(selected) >= holdout_size:
            break
        add(candidate)
    return selected


def _load_development_comparison(path: str) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    full = payload.get("summary", {}).get("success_rate_by_method", {}).get("full")
    if not isinstance(full, dict):
        return None
    return {
        "label": "development benchmark",
        "path": path,
        "success_rate_by_method_full": {
            key: value for key, value in full.items() if key in FROZEN_METHODS
        },
    }


def _category_distribution(candidates: Sequence[HoldoutCandidate]) -> dict[str, Any]:
    out: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        item = candidate.item
        for tag in item.tags:
            out.setdefault(tag, {"count": 0})["count"] += 1
        out.setdefault(item.category, {"count": 0})["count"] += 1
        out.setdefault(item.tightness, {"count": 0})["count"] += 1
    return out


def _generate_holdout_boxes(
    inputs: base.StaticInputs,
    global_constants: base.MetricConstants,
    args: argparse.Namespace,
) -> dict[str, Any]:
    creation_time = _utc_now()
    full_prepared = _full_prepared(inputs, global_constants)
    pair_specs = bench._select_pair_specs(
        inputs,
        full_prepared,
        pair_count=args.pair_count,
        max_candidate_pairs=args.max_candidate_pairs,
        seed=args.seed,
        min_shortest_m=args.min_shortest_m,
        max_shortest_m=args.max_shortest_m,
    )

    all_candidates: list[HoldoutCandidate] = []
    rejections: list[dict[str, Any]] = []
    generation_by_pair: list[dict[str, Any]] = []

    historical_candidates = _historical_paris_witnesses(inputs)
    all_candidates.extend(historical_candidates)
    if historical_candidates:
        generation_by_pair.append(
            {
                "pair_id": "paris_bures",
                "source": "historical_experiment_output",
                "candidate_count": len(historical_candidates),
            }
        )

    for pair in pair_specs:
        witnesses, generation = _collect_non_frozen_witnesses(
            inputs,
            full_prepared,
            pair,
            max_profiles_per_scalar=args.max_holdout_profiles_per_scalar,
            max_witnesses_per_pair=args.max_witnesses_per_pair,
        )
        candidates, pair_rejections = _holdout_candidates_for_pair(
            inputs,
            pair,
            witnesses,
            creation_time=creation_time,
            witness_source="non_frozen_scalar_portfolio",
            witness_method_prefix="non_frozen_portfolio",
        )
        all_candidates.extend(candidates)
        rejections.extend(pair_rejections)
        generation_by_pair.append(
            {
                "pair": {
                    "pair_id": pair.pair_id,
                    "source": pair.source,
                    "target": pair.target,
                    "selection_note": pair.selection_note,
                    "ordinary_shortest": pair.shortest.as_dict(include_paths=False),
                },
                "generation": generation,
                "candidate_box_count": len(candidates),
                "candidate_rejection_count": len(pair_rejections),
            }
        )
        if len({candidate.item.pair_id for candidate in all_candidates}) >= args.pair_count and len(all_candidates) >= args.holdout_size * 5:
            break

    selected = _select_holdout_candidates(
        all_candidates,
        holdout_size=args.holdout_size,
        max_paris_items=args.max_paris_items,
        max_boxes_per_witness=2,
    )
    distinct_pairs = {candidate.item.pair_id for candidate in selected}
    if len(selected) < args.min_holdout_size:
        raise RuntimeError(
            f"only selected {len(selected)} holdout boxes; need at least {args.min_holdout_size}"
        )
    if len(distinct_pairs) < 3:
        raise RuntimeError(
            f"holdout has only {len(distinct_pairs)} distinct pairs; need at least 3"
        )
    if sum(1 for c in selected if c.item.pair_id == "paris_bures") > args.max_paris_items:
        raise RuntimeError("Paris-Bures is not a minority in selected holdout")
    if not any(
        c.item.pair_id != "paris_bures"
        and c.item.deliberate_adversarial
        and c.item.shortest_violation_score > args.adversarial_violation_threshold
        for c in selected
    ):
        raise RuntimeError("no non-Paris deliberately adversarial holdout item selected")

    accepted_boxes = [candidate.as_box_dict(include_paths=True) for candidate in selected]
    rejected_count = len(all_candidates) - len(selected) + len(rejections)
    return {
        "benchmark_label": HOLDOUT_LABEL,
        "frozen": True,
        "freeze_workflow": {
            "phase_a": "generate independent witnesses, validate, construct boxes, write this JSON",
            "phase_b": "evaluation script reloads this frozen JSON from disk",
            "do_not_modify_after_evaluation": True,
        },
        "created_at_utc": creation_time,
        "holdout_items_attempted": len(all_candidates) + len(rejections),
        "holdout_items_accepted": len(selected),
        "holdout_items_rejected": rejected_count,
        "rejection_reasons": rejections[:200],
        "independence_policy": {
            "evaluated_scalars_excluded_from_witness_generation": sorted(
                FROZEN_EVALUATED_SCALARS
            ),
            "evaluated_spec_names_excluded_from_witness_generation": sorted(
                FROZEN_EVALUATED_SPEC_NAMES
            ),
            "allowed_witness_sources_used": sorted(
                {candidate.witness_source for candidate in selected}
            ),
            "non_frozen_scalar_specs_available": list(INDEPENDENT_SPEC_NAMES),
        },
        "category_distribution": _category_distribution(selected),
        "distinct_pairs": sorted(distinct_pairs),
        "generation_by_pair": generation_by_pair,
        "boxes": accepted_boxes,
    }


def _classify_union2_failures(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("graph_mode") != "full"
            or row.get("method") != "via_union"
            or row.get("scalar_name") != bench.UNION_2_NAME
            or bool(row.get("feasible"))
        ):
            continue
        members = [
            member
            for member in rows
            if member.get("graph_mode") == "full"
            and member.get("method") == "via"
            and member.get("query_id") == row.get("query_id")
            and member.get("scalar_name") in {bench.SCALAR_REFERENCE, bench.SCALAR_SLOPE}
        ]
        profile_count = sum(int(member.get("profile_feasible_count") or 0) for member in members)
        exact_count = sum(int(member.get("exact_feasible_count") or 0) for member in members)
        non_elementary = sum(int(member.get("non_elementary_count") or 0) for member in members)
        validation_rejects = sum(int(member.get("rejected_validation_count") or 0) for member in members)
        if validation_rejects:
            failure_class = "VALIDATION_FAILURE"
        elif profile_count == 0:
            failure_class = "NO_PROFILE_FEASIBLE"
        elif exact_count == 0 and non_elementary >= profile_count:
            failure_class = "PROFILE_FEASIBLE_BUT_NON_ELEMENTARY"
        else:
            failure_class = "OTHER"
        nearest = row.get("nearest_exact_candidate") or row.get("nearest_profile")
        failures.append(
            {
                "query_id": row.get("query_id"),
                "failure_class": failure_class,
                "profile_feasible_count": profile_count,
                "exact_elementary_feasible_count": exact_count,
                "non_elementary_count": non_elementary,
                "validation_reject_count": validation_rejects,
                "best_normalized_violation": row.get("normalized_violation_score"),
                "violated_constraints": row.get("violated_constraints"),
                "nearest_candidate_or_profile": nearest,
            }
        )
    return failures


def _evaluate_frozen_holdout(
    inputs: base.StaticInputs,
    global_constants: base.MetricConstants,
    rho_info: dict[str, Any],
    boxes_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    items, load_meta = bench._load_items_from_boxes_json(inputs, boxes_path)
    rows, graph_comparisons = bench._run_benchmark(inputs, items, global_constants, args)
    summary = bench._build_summary(
        rows,
        items,
        tie_threshold=args.tie_substantial_threshold,
    )
    with open(boxes_path, encoding="utf-8") as fh:
        frozen_boxes = json.load(fh)
    holdout_failures = _classify_union2_failures(rows)
    summary["holdout_union2_failure_classification"] = holdout_failures
    summary["development_comparison"] = _load_development_comparison(
        args.development_results_json
    )
    return {
        "metadata": {
            "script": Path(__file__).name,
            "benchmark_label": HOLDOUT_LABEL,
            "boxes_json": boxes_path,
            "graph_path": args.graph_path,
            "phase": "evaluate_frozen_holdout",
            "evaluated_methods": list(FROZEN_METHODS),
            "scalarizations": list(bench.BENCHMARK_SCALAR_NAMES),
            "rho_H_global": rho_info,
            "freeze_confirmation": (
                "Holdout boxes were loaded from disk before evaluation; this "
                "evaluation code does not alter thresholds or witness paths."
            ),
        },
        "frozen_holdout": frozen_boxes,
        "load_meta": load_meta,
        "queries": [item.as_dict(include_paths=False) for item in items],
        "graph_comparisons": graph_comparisons,
        "rows": rows,
        "summary": summary,
    }


def _print_holdout_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    full = summary["success_rate_by_method"].get("full", {})
    print("\nHoldout summary", flush=True)
    print(
        f"  boxes={summary['box_count']} "
        f"pairs={summary['distinct_pair_count']} "
        f"pair_ids={', '.join(summary['distinct_pairs'])}",
        flush=True,
    )
    for key in FROZEN_METHODS:
        item = full.get(key)
        if item is None:
            continue
        print(
            f"  full {key}: {item['solved']}/{item['attempted']} "
            f"({item['percent']:.1f}%)",
            flush=True,
        )
    failures = summary.get("holdout_union2_failure_classification", [])
    print(f"  full {bench.UNION_2_NAME} failures={len(failures)}", flush=True)
    for failure in failures:
        print(
            f"    {failure['query_id']}: {failure['failure_class']} "
            f"profiles={failure['profile_feasible_count']} "
            f"non_elementary={failure['non_elementary_count']}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and evaluate an independent holdout benchmark for the "
            "three frozen scalar/direct-via methods."
        )
    )
    parser.add_argument("--graph-path", default=base.GRAPH_PATH)
    parser.add_argument("--seeds-path", default=base.SEEDS_PATH)
    parser.add_argument("--partition-path", default=base.PARTITION_PATH)
    parser.add_argument("--boundary-nodes-path", default=base.BOUNDARY_NODES_PATH)
    parser.add_argument(
        "--phase",
        choices=("generate", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--holdout-boxes-json", default=DEFAULT_HOLDOUT_BOXES_JSON)
    parser.add_argument("--output-json", default=DEFAULT_HOLDOUT_RESULTS_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_HOLDOUT_RESULTS_CSV)
    parser.add_argument("--development-results-json", default=DEFAULT_DEVELOPMENT_RESULTS_JSON)
    parser.add_argument("--holdout-size", type=int, default=12)
    parser.add_argument("--min-holdout-size", type=int, default=10)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--max-candidate-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--min-shortest-m", type=float, default=6000.0)
    parser.add_argument("--max-shortest-m", type=float, default=52000.0)
    parser.add_argument("--max-paris-items", type=int, default=2)
    parser.add_argument("--max-witnesses-per-pair", type=int, default=35)
    parser.add_argument("--max-holdout-profiles-per-scalar", type=int, default=45)
    parser.add_argument("--adversarial-violation-threshold", type=float, default=0.25)
    parser.add_argument(
        "--graph-modes",
        default="full",
        help="Passed through to article_scalar_via_benchmark evaluation.",
    )
    parser.add_argument(
        "--include-geometric-paris",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tie-substantial-threshold", type=int, default=10)
    parser.add_argument(
        "--overwrite-holdout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow replacing the frozen holdout boxes file during generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    inputs = base._load_static_inputs(args)
    global_constants, rho_info = bench._compute_global_metric_constants(inputs)
    boxes_path = Path(args.holdout_boxes_json)

    if args.phase in {"generate", "all"}:
        if boxes_path.exists() and not args.overwrite_holdout:
            print(
                f"Frozen holdout already exists at {boxes_path}; loading it without regeneration.",
                flush=True,
            )
        else:
            payload = _generate_holdout_boxes(inputs, global_constants, args)
            bench._write_json(str(boxes_path), payload)
            print(
                f"Froze {payload['holdout_items_accepted']} holdout boxes at {boxes_path}",
                flush=True,
            )
        if args.phase == "generate":
            print("Generation phase complete; no evaluation was run.", flush=True)
            return

    if not boxes_path.exists():
        raise FileNotFoundError(
            f"{boxes_path} does not exist; run with --phase generate first"
        )

    result_payload = _evaluate_frozen_holdout(
        inputs,
        global_constants,
        rho_info,
        str(boxes_path),
        args,
    )
    result_payload["metadata"]["elapsed_s"] = time.perf_counter() - start
    bench._write_json(args.output_json, result_payload)
    bench._write_csv(args.output_csv, result_payload["rows"])
    _print_holdout_summary(result_payload)
    print(f"\nWrote {args.output_json} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
