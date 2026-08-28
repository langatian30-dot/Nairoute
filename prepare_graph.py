import os
import osmnx as ox

from utils import (
    GRAPH_FILE,
    DEFAULT_CITY,
    enrich_graph_with_ai,
    RISK_FIELDS
)

print("========================================")
print("SafeRoute Graph Preparation")
print("========================================")

print()
print("GRAPH_FILE:")
print(os.path.abspath(GRAPH_FILE))

# ------------------------------------------------------
# LOAD GRAPH
# ------------------------------------------------------

if os.path.exists(GRAPH_FILE):

    print()
    print("Loading existing graph...")

    G = ox.load_graphml(GRAPH_FILE)

else:

    print()
    print("Downloading road network...")

    G = ox.graph_from_place(
        DEFAULT_CITY,
        network_type="drive"
    )

print()
print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

# ------------------------------------------------------
# CHECK CURRENT RISK FIELDS
# ------------------------------------------------------

u, v, key, data = next(
    iter(
        G.edges(
            keys=True,
            data=True
        )
    )
)

print()
print("Risk fields BEFORE enrichment:")

for field in RISK_FIELDS.values():
    print(
        field,
        "=",
        data.get(field, "MISSING")
    )

# ------------------------------------------------------
# ENRICH
# ------------------------------------------------------

print()
print("Starting AI risk enrichment...")

G = enrich_graph_with_ai(G)

# ------------------------------------------------------
# CHECK IN MEMORY
# ------------------------------------------------------

u, v, key, data = next(
    iter(
        G.edges(
            keys=True,
            data=True
        )
    )
)

print()
print("Risk fields AFTER enrichment:")

for field in RISK_FIELDS.values():

    print(
        field,
        "=",
        data.get(field, "MISSING")
    )

# ------------------------------------------------------
# SAVE
# ------------------------------------------------------

print()
print("Saving enriched graph to:")

absolute_path = os.path.abspath(GRAPH_FILE)

print(absolute_path)

ox.save_graphml(
    G,
    absolute_path
)

print()
print("Graph saved.")

# ------------------------------------------------------
# VERIFY SAVED FILE
# ------------------------------------------------------

print()
print("Reloading saved graph for verification...")

G_check = ox.load_graphml(
    absolute_path
)

u, v, key, data = next(
    iter(
        G_check.edges(
            keys=True,
            data=True
        )
    )
)

print()
print("Risk fields AFTER RELOADING FILE:")

for field in RISK_FIELDS.values():

    print(
        field,
        "=",
        data.get(field, "MISSING")
    )

print()
print("========================================")

if all(
    field in data
    for field in RISK_FIELDS.values()
):

    print("✅ ENRICHED GRAPH VERIFIED")

else:

    print("❌ RISK FIELDS ARE STILL MISSING")

print("========================================")