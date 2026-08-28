import osmnx as ox

from utils import (
    GRAPH_FILE,
    enrich_graph_with_ai
)

print("Loading graph...")

G = ox.load_graphml(GRAPH_FILE)

print(
    f"Graph contains {G.number_of_edges():,} edges."
)

# ------------------------------------------------------
# SMALL TEST GRAPH
# ------------------------------------------------------

edges = list(
    G.edges(
        keys=True
    )
)[:100]

test_graph = G.edge_subgraph(
    edges
).copy()

print(
    f"Testing enrichment on "
    f"{test_graph.number_of_edges():,} edges..."
)

# ------------------------------------------------------
# ENRICH
# ------------------------------------------------------

test_graph = enrich_graph_with_ai(
    test_graph
)

# ------------------------------------------------------
# CHECK RESULTS
# ------------------------------------------------------

u, v, key, data = next(
    iter(
        test_graph.edges(
            keys=True,
            data=True
        )
    )
)

print()
print("Sample enriched edge:")
print(data)

print()
print("Risk fields:")

for field in RISK_FIELDS.values():

    print(
        field,
        "=",
        data.get(field)
    )