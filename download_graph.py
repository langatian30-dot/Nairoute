import osmnx as ox

print("Downloading Nairobi road network...")

G = ox.graph_from_place(
    "Nairobi, Kenya",
    network_type="drive"
)

ox.save_graphml(
    G,
    "road_network.graphml"
)

print("Done! road_network.graphml has been created.")