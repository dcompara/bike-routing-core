import math
import ast
from collections import defaultdict


def parse_osmid(x):
    """Return list[int] from edge attribute 'osmid' (int/list/str)."""
    if x is None:
        return []
    if isinstance(x, int):
        return [x]
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, int):
                return [v]
            if isinstance(v, (list, tuple, set)):
                out = []
                for t in v:
                    try:
                        out.append(int(t))
                    except Exception:
                        pass
                return out
        except Exception:
            pass
        # fallback separators
        for sep in [";", ",", " "]:
            if sep in s:
                parts = [
                    p
                    for p in s.replace("[", "").replace("]", "").split(sep)
                    if p.strip()
                ]
                out = []
                for p in parts:
                    try:
                        out.append(int(p))
                    except Exception:
                        pass
                return out
        try:
            return [int(s)]
        except Exception:
            return []
    return []


def norm_tag(x):
    """Normalize tag that can be str or list of str."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return str(x[0]).strip()
    return str(x).strip()


def road_key(data):
    """
    Refinement #1: prefer ref over name.
    Add highway to reduce accidental merges.
    Returns a hashable road-group key.
    """
    ref = norm_tag(data.get("ref"))
    name = norm_tag(data.get("name"))
    highway = norm_tag(data.get("highway"))

    label = ref if ref else (name if name else "__noname__")
    return (label, highway)


def export_graph_to_xy(
    G,
    output_file_path,
    *,
    road_key_fn=road_key,
    parse_osmid_fn=parse_osmid,
    return_mapping: bool = True,
):
    """
    Export a NetworkX graph to a Warthog-compatible .xy file.

    Node IDs are remapped to 0..N-1 (zero-indexed).
    """

    # Node mapping: old NetworkX node id -> new contiguous id (0..N-1)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}

    # store node attributes indexed by NEW ids
    node_dict = {
        node_mapping[node]: (
            data["x"],
            data["y"],
            int(float(data["elevation"])),
        )
        for node, data in G.nodes(data=True)
    }

    # PASS 1: length-weighted osmid vote per road group
    weight_by_group = defaultdict(lambda: defaultdict(float))
    for _, _, data in G.edges(data=True):
        key = road_key_fn(data)
        L = float(data.get("length", 0.0))
        osmids = parse_osmid_fn(data.get("osmid"))
        for oid in osmids:
            weight_by_group[key][oid] += L

    # canonical way osmid per group
    canonical_way_osmid = {}
    for key, wmap in weight_by_group.items():
        if not wmap:
            canonical_way_osmid[key] = 0
            continue
        max_w = max(wmap.values())
        best = min(oid for oid, w in wmap.items() if w == max_w)
        canonical_way_osmid[key] = best

    # Renumber roads (0..R-1)
    all_keys = sorted(canonical_way_osmid.keys())
    road_id32 = {key: i for i, key in enumerate(all_keys)}

    # Export
    output_lines = []
    output_lines.append("# warthog xy Energy graph \n")
    output_lines.append("# this file is formatted as follows: [header data] [node data] [edge data] \n")
    output_lines.append("# header format: nodes [number of nodes] edges [number of edges]  \n")
    output_lines.append("# node data format: v [id] [x=lon*1e6] [y=lat*1e6] [elevation] \n")
    output_lines.append("# edge data format: e [from] [to] [length] [10 x height_gain] [popularity] [street_width] [road_id32] \n")
    output_lines.append("# \n")
    output_lines.append("# 316bit integer values are used throughout excpet for road_id. \n")
    output_lines.append("# Identifiers are all zero indexed. \n")
    output_lines.append("# \n")
    output_lines.append(f"nodes {len(G.nodes)} edges {len(G.edges)} attributes 5 (4 plus the road_id) \n")

    # nodes: write in ascending new id order (important!)
    for node_id in range(len(G.nodes)):
        x, y, elevation = node_dict[node_id]
        output_lines.append(
            f"v {node_id} {int(float(x) * 1e6)} {int(float(y) * 1e6)} {elevation} \n"
        )

    # edges: use new ids
    for from_node, to_node, data in G.edges(data=True):
        length = math.ceil(float(data.get("length", 0)))
        height_gain = int(max(0, 10 * float(data.get("height_gain", 0))))
        popularity = math.ceil(float(data.get("popularity", 0)))
        street_width = math.ceil(10 * float(data.get("street_width", 0)))

        key = road_key_fn(data)
        rid = road_id32.get(key, 0)

        new_from = node_mapping[from_node]
        new_to = node_mapping[to_node]

        output_lines.append(
            f"e {new_from} {new_to} {length} {height_gain} {popularity} {street_width} {rid} \n"
        )

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"Graph saved to {output_file_path}")

    if return_mapping:
        return node_mapping, road_id32
