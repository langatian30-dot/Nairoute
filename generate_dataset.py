import osmnx as ox
import pandas as pd
from geopy.distance import geodesic

from crime_data import CRIME_HOTSPOTS

# ==========================================================
# LOAD NAIROBI ROAD NETWORK
# ==========================================================

print("Loading Nairobi road network...")

G = ox.graph_from_place(
    "Nairobi, Kenya",
    network_type="drive"
)

print("Road network loaded.")

# ==========================================================
# CLEAN FUNCTIONS
# ==========================================================

def clean_highway(highway):

    if isinstance(highway, list):
        return highway[0]

    if highway is None:
        return "residential"

    return str(highway)


def clean_speed(speed):

    if speed is None:
        return 50

    if isinstance(speed, list):
        speed = speed[0]

    try:

        return float(
            str(speed)
            .replace(" km/h", "")
            .replace(" mph", "")
        )

    except:

        return 50


def clean_lanes(lanes):

    if lanes is None:
        return 1

    if isinstance(lanes, list):
        lanes = lanes[0]

    try:

        return int(lanes)

    except:

        return 1


# ==========================================================
# CRIME SCORE
# ==========================================================

def crime_score(lat, lon):

    score = 0

    for hotspot in CRIME_HOTSPOTS:

        d = geodesic(

            (lat, lon),

            (
                hotspot["lat"],
                hotspot["lon"]
            )

        ).km

        if d < 0.5:

            score += hotspot["risk"]

        elif d < 1:

            score += hotspot["risk"] * 0.7

        elif d < 2:

            score += hotspot["risk"] * 0.4

    return round(score, 2)


# ==========================================================
# TIME PERIOD
# ==========================================================

# ==========================================================
# TIME PERIODS
# ==========================================================

TIME_PERIODS = {

    "Morning": 1.00,

    "Afternoon": 1.05,

    "Evening": 1.20,

    "Night": 1.50

}


# ==========================================================
# CREATE DATASET
# ==========================================================

rows = []

print("Extracting road features...")
for u, v, key, data in G.edges(keys=True, data=True):

    node = G.nodes[u]

    highway = clean_highway(
        data.get("highway")
    )

    length = float(
        data.get("length", 100)
    )

    maxspeed = clean_speed(
        data.get("maxspeed")
    )

    lanes = clean_lanes(
        data.get("lanes")
    )

    crime = crime_score(
        node["y"],
        node["x"]
    )

    # THIS LOOP MUST BE INSIDE THE FIRST LOOP
    for time_period, multiplier in TIME_PERIODS.items():

        risk = (

            crime * 0.6

            + length / 1000 * 0.2

            + lanes * 0.1

            + maxspeed / 100 * 0.1

        )

        risk *= multiplier

        rows.append({

            "highway": highway,

            "length": length,

            "maxspeed": maxspeed,

            "lanes": lanes,

            "crime_score": crime,

            "time_period": time_period,

            "risk": round(risk, 2)

        })
# ==========================================================
# SAVE DATASET
# ==========================================================

dataset = pd.DataFrame(rows)

dataset.to_csv(
    "road_risk_dataset.csv",
    index=False
)

print()

print(dataset.head())

print()

print(f"Dataset Size: {dataset.shape}")

print()

print("road_risk_dataset.csv created successfully.")