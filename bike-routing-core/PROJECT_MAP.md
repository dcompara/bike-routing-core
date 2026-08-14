# PROJECT_MAP.md

## Project name
BIKE-ROUTING-CORE

## Project goal

This project aims to build a **multi-objective bike routing algorithm** that generates a small set of candidate routes between a source and a target while taking into account several criteria simultaneously.

Current objectives include, depending on availability in the graph:
- path length / distance
- elevation gain
- road size / comfort
- popularity / cycling usage proxies (e.g. Strava heatmap-derived data)
- possibly other edge attributes later (a road_id is already there to follow a given street name)

The long-term goal is **not** only shortest path routing, but a route generator able to return a few meaningful trade-off solutions subject to lower and upper bounds on several objectives.

---

## High-level architecture

The project is split conceptually into two phases:

### 1. Offline preprocessing phase
This phase is mostly already done.

It includes:
- loading and processing an OpenStreetMap graph
- assigning compact node identifiers
- exporting the graph to a compact `.xy` format
- generating graph partition data
- generating seeds
- generating Voronoi / hop-based cells
- generating boundary / border node files

This phase is currently **not the main focus**.

### 2. Online routing phase
This is the current focus of the project.

It includes:
- loading the compact graph and partition files
- reducing the search space using source-target geometry and partition information
- implementing the actual multi-objective routing algorithm
- finding a few feasible candidate routes satisfying objective bounds
- scoring / ranking / diversifying the returned routes

---

## Current project priority

**Main priority:** implement and stabilize the routing algorithm.

**Not a priority right now:**
- redesigning the OSM preprocessing pipeline
- reorganizing old preprocessing code unless strictly needed
- changing file formats unless necessary for the routing core

In particular, `src/brOSM/` should mostly be treated as:
- already useful,
- mostly stable,
- reference / preprocessing code,
- low-priority for refactoring.

Codex should focus primarily on:
- `src/brcore/algo/`
- `src/brcore/graph/`
- `src/brcore/io/`
- the interfaces needed for the future routing core

---

## Current directory structure

```text
BIKE-ROUTING-CORE/

├── data/
│   ├── graph_Paris_south_4_objectives.xy
│   ├── paris_voronoi_boundaries.txt
│   ├── paris_voronoi_boundary_nodes.txt
│   ├── paris_voronoi_cells.txt
│   ├── paris_voronoi_nodes.txt
│   ├── seeds_latlon.txt
│   ├── seeds.txt
│   └── South_graph_with_road_ids.graphml
│
├── scripts/
│   └── main.py
│
├── src/
│   ├── brcore/
│   │   ├── algo/
│   │   │   ├── coords.py
│   │   │   ├── heuristic.py
│   │   │   ├── heuristics_int.py
│   │   │   ├── params.py
│   │   │   └── search_space_reduction.py
│   │   │
│   │   ├── graph/
│   │   │   └── compact.py
│   │   │
│   │   ├── io/
│   │   │   ├── load_plot_xy.py
│   │   │   └── loaders.py
│   │   │
│   │   └── __init__.py
│   │
│   └── brOSM/
│       ├── Graph_OSM.py
│       ├── partition.py
│       └── save_graph_xy.py
│
├── CODE_ChatGPT.docx
├── Notes.txt
└── pyproject.toml
