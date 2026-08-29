# ==========================================================
# SAFEROUTE V4.0
# utils.py
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium
import streamlit as st

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from crime_data import CRIME_HOTSPOTS


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================

DEFAULT_CITY = "Nairobi, Kenya"

GRAPH_FILE = "nairobi_graph.graphml"
MODEL_FILE = "risk_model.joblib"

GEOCODER = Nominatim(
    user_agent="NaiRouteAI",
    timeout=10
)

AVERAGE_SPEED = {
    "Morning": 40,
    "Afternoon": 45,
    "Evening": 35,
    "Night": 50
}

RISK_FIELDS = {
    "Morning": "risk_morning",
    "Afternoon": "risk_afternoon",
    "Evening": "risk_evening",
    "Night": "risk_night"
}


# ==========================================================
# EDGE ATTRIBUTE CLEANING (for popups / display)
# ==========================================================

def clean_highway_label(highway):
    """
    Normalizes the raw OSM 'highway' tag into a
    human-readable road type label.
    """

    if isinstance(highway, list):
        highway = highway[0]

    if highway is None:
        highway = "residential"

    highway = str(highway).replace("_", " ").title()

    return highway


def clean_maxspeed_label(maxspeed):
    """
    Normalizes the raw OSM 'maxspeed' tag into a
    display string, e.g. '50 km/h'.
    """

    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]

    if maxspeed is None:
        return "Unknown"

    try:

        value = float(
            str(maxspeed)
            .replace("km/h", "")
            .replace("mph", "")
            .strip()
        )

        return f"{value:.0f} km/h"

    except (TypeError, ValueError):

        return "Unknown"


def clean_lanes_label(lanes):
    """
    Normalizes the raw OSM 'lanes' tag into a
    display-friendly value.
    """

    if isinstance(lanes, list):
        lanes = lanes[0]

    if lanes is None:
        return "Unknown"

    try:

        return str(int(lanes))

    except (TypeError, ValueError):

        return "Unknown"


# ==========================================================
# HUGGING FACE HUB CONFIG
# ==========================================================
# The road graph and trained model are too large for a normal
# git repository (the model alone exceeds GitHub's 100MB limit).
# They're hosted on Hugging Face Hub instead and downloaded
# automatically the first time the app runs, if not already
# present locally. This keeps the git repo lightweight while
# still working seamlessly for local development (where the
# files already exist and are used as-is, no download needed).

HF_REPO_ID = "Ianoo412/Nairoute-data"


def _ensure_file_available(local_path, hf_filename):
    """
    Returns a usable local path to the given file, downloading
    it from Hugging Face Hub first if it isn't already present
    on disk (e.g. on a fresh deployment).
    """

    if os.path.exists(local_path):
        return local_path

    try:

        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            repo_type="dataset",
        )

        return downloaded_path

    except Exception as e:

        raise FileNotFoundError(
            f"Could not find '{local_path}' locally, and "
            f"downloading it from Hugging Face Hub also "
            f"failed: {e}"
        )


# ==========================================================
# LOAD AI MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_path = _ensure_file_available(
        MODEL_FILE,
        MODEL_FILE,
    )

    return joblib.load(model_path)


# ==========================================================
# LOAD ROAD NETWORK
# ==========================================================

@st.cache_resource
def load_graph():

    graph_path = _ensure_file_available(
        GRAPH_FILE,
        GRAPH_FILE,
    )

    return ox.load_graphml(graph_path)


# ==========================================================
# GEOCODING
# ==========================================================

def geocode_location(location):
    """
    Convert a user-entered location into latitude and longitude.

    The query is restricted to Kenya and explicitly targets Nairobi.
    """

    if not location or not location.strip():
        raise ValueError("Location cannot be empty.")

    query = location.strip()

    if "nairobi" not in query.lower():
        query = f"{query}, Nairobi, Kenya"

    place = GEOCODER.geocode(
        query,
        country_codes="ke",
        addressdetails=True,
        exactly_one=True
    )

    if place is None:
        raise ValueError(
            f"Could not find '{location}' in Nairobi, Kenya."
        )

    return (
        float(place.latitude),
        float(place.longitude)
    )


MAX_LOCATION_DISTANCE_KM = 20


def validate_location_within_range(
    label,
    user_lat,
    user_lon,
    node_lat,
    node_lon,
    max_km=MAX_LOCATION_DISTANCE_KM
):
    """
    Raises ValueError if the road node a location snapped to
    is unreasonably far from where the location actually
    geocoded to (e.g. a location outside Nairobi entirely).

    Pulled out as its own function, taking plain coordinates,
    specifically so this logic can be unit-tested directly
    without depending on geocoding behavior.
    """

    distance_km = geodesic(
        (user_lat, user_lon),
        (node_lat, node_lon),
    ).km

    if distance_km > max_km:

        raise ValueError(
            f"{label} appears to be outside the NaiRoute "
            f"Nairobi road network (~{distance_km:.0f} km "
            "away). Please enter a location within Nairobi."
        )

    return distance_km


# ==========================================================
# FIND NEAREST ROAD NODES
# ==========================================================

def get_nearest_nodes(G, start_location, destination):
    """
    Convert user locations to coordinates and snap them
    to the nearest nodes in the road graph.

    Handles both:
    - unprojected graphs (latitude/longitude)
    - projected graphs (meter-based CRS)
    """

    # ------------------------------------------------------
    # GEOCODE USER LOCATIONS
    # ------------------------------------------------------

    start_lat, start_lon = geocode_location(
        start_location
    )

    end_lat, end_lon = geocode_location(
        destination
    )

    # ------------------------------------------------------
    # CONVERT COORDINATES TO GRAPH CRS
    # ------------------------------------------------------

    graph_crs = G.graph.get("crs")

    try:

        from pyproj import Transformer

        if graph_crs:

            graph_crs_string = str(
                graph_crs
            )

            if graph_crs_string.upper() not in (
                "EPSG:4326",
                "EPSG:4326.0"
            ):

                transformer = Transformer.from_crs(
                    "EPSG:4326",
                    graph_crs,
                    always_xy=True
                )

                start_x, start_y = transformer.transform(
                    start_lon,
                    start_lat
                )

                end_x, end_y = transformer.transform(
                    end_lon,
                    end_lat
                )

            else:

                start_x = start_lon
                start_y = start_lat

                end_x = end_lon
                end_y = end_lat

        else:

            # Assume standard WGS84 coordinates.
            start_x = start_lon
            start_y = start_lat

            end_x = end_lon
            end_y = end_lat

    except Exception:

        # Safe fallback for normal WGS84 graphs.
        start_x = start_lon
        start_y = start_lat

        end_x = end_lon
        end_y = end_lat

    # ------------------------------------------------------
    # FIND NEAREST GRAPH NODES
    # ------------------------------------------------------

    start_node, start_snap_distance = (
        ox.distance.nearest_nodes(
            G,
            X=start_x,
            Y=start_y,
            return_dist=True
        )
    )

    end_node, end_snap_distance = (
        ox.distance.nearest_nodes(
            G,
            X=end_x,
            Y=end_y,
            return_dist=True
        )
    )

    # ------------------------------------------------------
    # VALIDATE SNAP DISTANCES
    # ------------------------------------------------------

    try:

        start_snap_distance = float(
            start_snap_distance
        )

        end_snap_distance = float(
            end_snap_distance
        )

    except Exception:

        start_snap_distance = 0
        end_snap_distance = 0

    graph_is_geographic = True

    if graph_crs:

        crs_string = str(graph_crs).upper()

        graph_is_geographic = (
            "4326" in crs_string
        )

    if graph_is_geographic:

        # For lat/lon (EPSG:4326) graphs, the snap distance
        # returned by nearest_nodes is in degrees, which isn't
        # meaningful to threshold directly. Instead, measure
        # the real-world distance between where the user's
        # location actually geocoded to and the road node it
        # snapped onto — this correctly catches locations far
        # outside Nairobi regardless of the graph's coordinate
        # system.

        validate_location_within_range(
            f"Start location '{start_location}'",
            start_lat,
            start_lon,
            float(G.nodes[start_node]["y"]),
            float(G.nodes[start_node]["x"]),
        )

        validate_location_within_range(
            f"Destination '{destination}'",
            end_lat,
            end_lon,
            float(G.nodes[end_node]["y"]),
            float(G.nodes[end_node]["x"]),
        )

    else:

        # Projected (meter-based) graph: the snap distance
        # from nearest_nodes is already in meters.

        if start_snap_distance > MAX_LOCATION_DISTANCE_KM * 1000:

            raise ValueError(
                "Start location is too far from the "
                "NaiRoute road network. Please enter a "
                "more specific location within Nairobi."
            )

        if end_snap_distance > MAX_LOCATION_DISTANCE_KM * 1000:

            raise ValueError(
                "Destination is too far from the "
                "NaiRoute road network. Please enter a "
                "more specific location within Nairobi."
            )

    return start_node, end_node


# ==========================================================
# AI RISK PREDICTION
# ==========================================================

def predict_risk(
    length,
    maxspeed,
    lanes,
    highway,
    crime_score,
    time_period
):

    model = load_model()

    data = pd.DataFrame({
        "length": [length],
        "maxspeed": [maxspeed],
        "lanes": [lanes],
        "highway": [highway],
        "crime_score": [crime_score],
        "time_period": [time_period]
    })

    prediction = model.predict(data)

    return float(prediction[0])


# ==========================================================
# CRIME HOTSPOT SCORE
# ==========================================================

def get_crime_score(
    lat,
    lon
):

    nearest = 0

    for hotspot in CRIME_HOTSPOTS:

        distance = (
            (lat - hotspot["lat"]) ** 2
            +
            (lon - hotspot["lon"]) ** 2
        ) ** 0.5

        if distance < 0.01:

            nearest = max(
                nearest,
                hotspot["risk"]
            )

    return nearest


# ==========================================================
# RISK COLOUR
# ==========================================================

def get_risk_color(risk):

    if risk < 2:
        return "green"

    elif risk < 4:
        return "yellow"

    elif risk < 6:
        return "orange"

    return "red"


# ==========================================================
# CALCULATE ROUTE RISK
# ==========================================================

def calculate_route_risk(
    G,
    route,
    time_period
):

    risk_field = RISK_FIELDS[
        time_period
    ]

    risks = []

    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        edge_data = G.get_edge_data(
            u,
            v
        )

        if not edge_data:
            continue

        data = next(
            iter(edge_data.values())
        )

        risk = data.get(
            risk_field,
            0
        )

        try:
            risk = float(risk)

        except (
            TypeError,
            ValueError
        ):
            risk = 0

        risks.append(risk)

    if not risks:
        return 0

    return round(
        float(np.mean(risks)),
        2
    )


# ==========================================================
# ROUTE RISK BREAKDOWN
# ==========================================================

def calculate_route_risk_breakdown(
    G,
    route,
    time_period="Morning"
):

    risk_field = RISK_FIELDS.get(
        time_period,
        "risk_morning"
    )

    low = 0
    medium = 0
    high = 0
    total = 0

    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        edge_data = G.get_edge_data(
            u,
            v
        )

        if not edge_data:
            continue

        data = next(
            iter(edge_data.values())
        )

        risk = data.get(
            risk_field
        )

        if risk is None:
            continue

        try:
            risk = float(risk)

        except (
            TypeError,
            ValueError
        ):
            continue

        total += 1

        if risk < 0.35:

            low += 1

        elif risk < 0.70:

            medium += 1

        else:

            high += 1

    if total == 0:

        return {
            "low": 0,
            "medium": 0,
            "high": 0
        }

    return {
        "low": (low / total) * 100,
        "medium": (medium / total) * 100,
        "high": (high / total) * 100
    }


# ==========================================================
# ENRICH GRAPH WITH AI RISK
# ==========================================================

def enrich_graph_with_ai(G):

    total = G.number_of_edges()

    print(
        f"Starting risk enrichment "
        f"for {total:,} road edges..."
    )

    crime_cache = {}

    for i, (
        u,
        v,
        key,
        data
    ) in enumerate(
        G.edges(
            keys=True,
            data=True
        )
    ):

        if i % 1000 == 0:

            print(
                f"Processing "
                f"{i:,}/{total:,} "
                f"({(i / total) * 100:.1f}%)"
            )

        lat = float(
            G.nodes[u]["y"]
        )

        lon = float(
            G.nodes[u]["x"]
        )

        location_key = (
            round(lat, 4),
            round(lon, 4)
        )

        if location_key not in crime_cache:

            crime_cache[
                location_key
            ] = get_crime_score(
                lat,
                lon
            )

        crime = crime_cache[
            location_key
        ]

        length = float(
            data.get(
                "length",
                100
            )
        )

        maxspeed = data.get(
            "maxspeed",
            50
        )

        if isinstance(
            maxspeed,
            list
        ):

            maxspeed = maxspeed[0]

        try:

            maxspeed = float(
                str(
                    maxspeed
                ).split()[0]
            )

        except Exception:

            maxspeed = 50

        lanes = data.get(
            "lanes",
            1
        )

        if isinstance(
            lanes,
            list
        ):

            lanes = lanes[0]

        try:

            lanes = int(lanes)

        except Exception:

            lanes = 1

        highway = data.get(
            "highway",
            "residential"
        )

        if isinstance(
            highway,
            list
        ):

            highway = highway[0]

        for period, field in RISK_FIELDS.items():

            data[field] = predict_risk(
                length,
                maxspeed,
                lanes,
                highway,
                crime,
                period
            )

    print(
        "Risk enrichment complete."
    )

    return G


# ==========================================================
# SHORTEST ROUTE
# ==========================================================

def get_shortest_route(G, start_node, end_node):
    """
    Calculates the shortest route using road length.
    """

    route = nx.shortest_path(
        G,
        start_node,
        end_node,
        weight="length"
    )

    distance = nx.path_weight(
        G,
        route,
        weight="length"
    )

    distance = float(distance)

    # ------------------------------------------------------
    # SANITY CHECK
    # ------------------------------------------------------

    if distance <= 0:
        raise ValueError(
            "Start location and destination appear to be "
            "the same place (or too close together to "
            "calculate a route). Please enter two distinct "
            "locations."
        )

    return route, distance


# ==========================================================
# SAFEST ROUTE
# ==========================================================

def get_safest_route(
    G,
    start_node,
    end_node,
    time_period="Morning"
):

    risk_field = RISK_FIELDS[
        time_period
    ]

    for (
        u,
        v,
        key,
        data
    ) in G.edges(
        keys=True,
        data=True
    ):

        length = float(
            data.get(
                "length",
                100
            )
        )

        risk = data.get(
            risk_field,
            0
        )

        try:

            risk = float(risk)

        except (
            TypeError,
            ValueError
        ):

            risk = 0

        # Stronger penalty for risky roads

        safety_multiplier = (
            1
            +
            (risk ** 2) * 10
        )

        data[
            "safe_weight"
        ] = (
            length
            *
            safety_multiplier
        )

    route = nx.shortest_path(
        G,
        start_node,
        end_node,
        weight="safe_weight"
    )

    distance = nx.path_weight(
        G,
        route,
        weight="length"
    )

    return route, distance


# ==========================================================
# AI ROUTE EXPLANATION
# ==========================================================

def generate_route_explanation(
    shortest_risk,
    safest_risk,
    shortest_distance,
    safest_distance,
    shortest_breakdown,
    safest_breakdown
):
    """
    Builds a natural-language explanation of NaiRoute's
    routing decision, using only values already produced
    by the existing risk model and route calculations.

    No additional AI model is used here — this is a rule-based
    reasoning layer over the existing results.
    """

    # ------------------------------------------------------
    # RISK REDUCTION
    # ------------------------------------------------------

    if shortest_risk > 0:

        risk_reduction = (
            (shortest_risk - safest_risk)
            / shortest_risk
        ) * 100

    else:

        risk_reduction = 0.0

    # ------------------------------------------------------
    # DISTANCE INCREASE
    # ------------------------------------------------------

    if shortest_distance > 0:

        distance_increase = (
            (safest_distance - shortest_distance)
            / shortest_distance
        ) * 100

    else:

        distance_increase = 0.0

    extra_distance_km = (
        safest_distance - shortest_distance
    ) / 1000

    # ------------------------------------------------------
    # DECISION
    # ------------------------------------------------------

    if (
        risk_reduction >= 15
        and distance_increase <= 30
    ):

        recommendation = "Safest Route"

        decision_text = (
            "Safety improvement justifies the "
            "additional distance."
        )

    elif distance_increase > 30:

        recommendation = "Shortest Route"

        decision_text = (
            "The safer route requires a distance increase "
            "that isn't proportionate to the safety gain."
        )

    elif risk_reduction > 0:

        recommendation = "Shortest Route"

        decision_text = (
            "The safety improvement is too small to justify "
            "the longer route."
        )

    else:

        recommendation = "Shortest Route"

        decision_text = (
            "No meaningful safety advantage was found, so "
            "the shortest route is recommended."
        )

    # ------------------------------------------------------
    # IDENTIFY MAIN DRIVER OF THE SAFETY DIFFERENCE
    # ------------------------------------------------------
    # Compares risk-band composition between the two routes
    # to explain *why* the safer route is safer (or isn't).

    high_delta = (
        float(shortest_breakdown.get("high", 0))
        - float(safest_breakdown.get("high", 0))
    )

    medium_delta = (
        float(shortest_breakdown.get("medium", 0))
        - float(safest_breakdown.get("medium", 0))
    )

    low_delta = (
        float(safest_breakdown.get("low", 0))
        - float(shortest_breakdown.get("low", 0))
    )

    if high_delta >= 5 and high_delta >= medium_delta:

        reasoning = (
            "The main advantage comes from avoiding "
            "high-risk road segments — high-risk exposure "
            f"drops from {shortest_breakdown.get('high', 0):.0f}% "
            f"to {safest_breakdown.get('high', 0):.0f}% during "
            "the selected travel period."
        )

    elif medium_delta >= 5:

        reasoning = (
            "The main advantage comes from reducing time spent "
            "on medium-risk road segments — medium-risk exposure "
            f"drops from {shortest_breakdown.get('medium', 0):.0f}% "
            f"to {safest_breakdown.get('medium', 0):.0f}%."
        )

    elif low_delta >= 5:

        reasoning = (
            "The main advantage comes from routing through more "
            "low-risk segments — low-risk coverage increases from "
            f"{shortest_breakdown.get('low', 0):.0f}% to "
            f"{safest_breakdown.get('low', 0):.0f}%."
        )

    else:

        reasoning = (
            "There is minimal difference in road-segment risk "
            "composition between the two routes for this "
            "travel period."
        )

    # ------------------------------------------------------
    # SUMMARY SENTENCE
    # ------------------------------------------------------

    if recommendation == "Safest Route":

        headline = "🛡️ Safest Route recommended"

        summary = (
            f"This route is {abs(distance_increase):.0f}% "
            f"longer but has {abs(risk_reduction):.0f}% "
            "lower predicted road risk."
        )

    else:

        headline = "🚗 Shortest Route recommended"

        summary = (
            f"The alternative route would only reduce "
            f"predicted risk by {max(risk_reduction, 0):.0f}% "
            f"while adding {abs(distance_increase):.0f}% "
            "more distance."
        )

    return {
        "recommendation": recommendation,
        "risk_reduction": risk_reduction,
        "distance_increase": distance_increase,
        "extra_distance_km": extra_distance_km,
        "headline": headline,
        "summary": summary,
        "reasoning": reasoning,
        "decision_text": decision_text,
    }


# ==========================================================
# ROUTE HAZARD HOTSPOTS
# ==========================================================

def get_nearest_landmark(lat, lon):
    """
    Finds the nearest known crime hotspot name to a
    given coordinate, using the existing CRIME_HOTSPOTS
    reference list. Used to give hotspots a human-readable
    'near X' description without any extra API calls.
    """

    nearest_name = "an unnamed area"
    nearest_dist = float("inf")

    for hotspot in CRIME_HOTSPOTS:

        distance = (
            (lat - hotspot["lat"]) ** 2
            + (lon - hotspot["lon"]) ** 2
        ) ** 0.5

        if distance < nearest_dist:

            nearest_dist = distance
            nearest_name = hotspot["name"]

    return nearest_name


def identify_route_hotspots(
    G,
    route,
    time_period="Morning",
    max_hotspots=5
):
    """
    Scans a route's edges and identifies clusters of
    medium/high risk road segments — 'hazard hotspots'.

    Uses only the risk scores already produced by the
    existing model. No additional AI model is used.
    """

    risk_field = RISK_FIELDS.get(
        time_period,
        "risk_morning"
    )

    raw_hotspots = []

    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        edge_data = G.get_edge_data(u, v)

        if not edge_data:
            continue

        data = next(iter(edge_data.values()))

        risk = data.get(risk_field)

        if risk is None:
            continue

        try:

            risk = float(risk)

        except (TypeError, ValueError):

            continue

        # Only medium/high risk segments count as hotspots.
        if risk < 0.35:
            continue

        risk_label = (
            "High Risk"
            if risk >= 0.70
            else "Medium Risk"
        )

        lat = float(G.nodes[u]["y"])
        lon = float(G.nodes[u]["x"])

        near = get_nearest_landmark(lat, lon)

        length_m = float(
            data.get("length", 0) or 0
        )

        raw_hotspots.append({
            "risk": risk,
            "risk_label": risk_label,
            "near": near,
            "length_m": length_m,
            "lat": lat,
            "lon": lon,
        })

    if not raw_hotspots:
        return []

    # ------------------------------------------------------
    # MERGE CONSECUTIVE SEGMENTS NEAR THE SAME LANDMARK
    # ------------------------------------------------------

    merged = []

    for h in raw_hotspots:

        if merged and merged[-1]["near"] == h["near"]:

            merged[-1]["risk"] = max(
                merged[-1]["risk"],
                h["risk"]
            )

            merged[-1]["risk_label"] = (
                "High Risk"
                if merged[-1]["risk"] >= 0.70
                else "Medium Risk"
            )

            merged[-1]["length_m"] += h["length_m"]

        else:

            merged.append(dict(h))

    # ------------------------------------------------------
    # RANK BY RISK, KEEP TOP N
    # ------------------------------------------------------

    merged.sort(
        key=lambda h: h["risk"],
        reverse=True
    )

    return merged[:max_hotspots]


def add_hotspot_markers_to_map(
    m,
    hotspots,
    layer_name="🚨 Hazard Hotspots"
):
    """
    Adds a map layer with a colored marker at each
    identified hazard hotspot location.
    """

    if not hotspots:
        return m

    hotspot_layer = folium.FeatureGroup(
        name=layer_name
    )

    for i, hotspot in enumerate(hotspots, start=1):

        color = (
            "red"
            if hotspot["risk_label"] == "High Risk"
            else "orange"
        )

        popup_html = folium.Popup(
            (
                "<div style='font-family: sans-serif; "
                "font-size: 13px; line-height: 1.5;'>"
                f"<b>🚨 Hotspot {i} — "
                f"{hotspot['risk_label']}</b><br><br>"
                f"🤖 <b>Risk:</b> {hotspot['risk']:.2f}<br>"
                f"📍 <b>Near:</b> {hotspot['near']}"
                "</div>"
            ),
            max_width=220,
        )

        folium.CircleMarker(
            location=[
                hotspot["lat"],
                hotspot["lon"]
            ],
            radius=10,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup_html,
            tooltip=(
                f"Hotspot {i}: {hotspot['risk_label']} "
                f"({hotspot['risk']:.2f})"
            ),
        ).add_to(hotspot_layer)

    hotspot_layer.add_to(m)

    return m


# ==========================================================
# CREATE ROUTE MAP
# ==========================================================

def create_route_map(
    G,
    shortest_route,
    safest_route,
    time_period="Morning",
    hotspots=None
):

    risk_field = RISK_FIELDS.get(
        time_period,
        "risk_morning"
    )

    # ------------------------------------------------------
    # ROUTE COORDINATES
    # ------------------------------------------------------

    shortest_points = [
        (
            G.nodes[node]["y"],
            G.nodes[node]["x"]
        )
        for node in shortest_route
    ]

    safest_points = [
        (
            G.nodes[node]["y"],
            G.nodes[node]["x"]
        )
        for node in safest_route
    ]

    all_points = (
        shortest_points
        +
        safest_points
    )

    if not all_points:
        return None

    # ------------------------------------------------------
    # MAP CENTER
    # ------------------------------------------------------

    center_lat = (
        sum(
            point[0]
            for point in all_points
        )
        /
        len(all_points)
    )

    center_lon = (
        sum(
            point[1]
            for point in all_points
        )
        /
        len(all_points)
    )

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # ======================================================
    # AI ROAD RISK LAYER
    # ======================================================

    risk_layer = folium.FeatureGroup(
        name=(
            f"🤖 AI Road Risk — "
            f"{time_period}"
        )
    )

    route_nodes = set(
        shortest_route
        +
        safest_route
    )

    # Only draw edges that actually touch a node on one of
    # the two routes. The previous version also expanded to
    # every neighboring node, which for long routes (many
    # kilometers) could balloon into tens of thousands of
    # map elements and cause the map to fail to render.
    nearby_nodes = route_nodes

    candidate_edges = []

    for (
        u,
        v,
        key,
        data
    ) in G.edges(
        keys=True,
        data=True
    ):

        if (
            u not in nearby_nodes
            and v not in nearby_nodes
        ):

            continue

        candidate_edges.append(
            (u, v, key, data)
        )

    # ------------------------------------------------------
    # SAFETY CAP
    # ------------------------------------------------------
    # For very long routes this can still be a large number
    # of edges. Cap the total drawn so the map always renders
    # quickly, prioritizing the highest-risk segments so
    # nothing important gets dropped.

    MAX_RISK_SEGMENTS = 300

    if len(candidate_edges) > MAX_RISK_SEGMENTS:

        def _edge_risk(edge):

            edge_data = edge[3]

            edge_risk = edge_data.get(risk_field, 0)

            try:

                return float(edge_risk)

            except (TypeError, ValueError):

                return 0

        candidate_edges.sort(
            key=_edge_risk,
            reverse=True
        )

        candidate_edges = candidate_edges[:MAX_RISK_SEGMENTS]

    for (
        u,
        v,
        key,
        data
    ) in candidate_edges:

        risk = data.get(
            risk_field
        )

        geometry = data.get(
            "geometry"
        )

        if (
            risk is None
            or geometry is None
        ):

            continue

        try:

            risk = float(risk)

        except (
            TypeError,
            ValueError
        ):

            continue

        if risk < 0.35:

            risk_color = "green"
            risk_label = "Low Risk"

        elif risk < 0.70:

            risk_color = "orange"
            risk_label = "Medium Risk"

        else:

            risk_color = "red"
            risk_label = "High Risk"

        # Simplify the geometry (fewer points along curves) to
        # keep the map's HTML payload small. ~5m tolerance in
        # degrees — visually indistinguishable at road scale.
        try:

            display_geometry = geometry.simplify(
                0.00005,
                preserve_topology=False
            )

        except Exception:

            display_geometry = geometry

        coordinates = [
            (
                lat,
                lon
            )
            for lon, lat
            in display_geometry.coords
        ]

        # ------------------------------------------------------
        # ROAD SEGMENT ATTRIBUTES (for the popup)
        # ------------------------------------------------------

        segment_length_m = float(
            data.get("length", 0) or 0
        )

        road_type = clean_highway_label(
            data.get("highway")
        )

        speed_limit = clean_maxspeed_label(
            data.get("maxspeed")
        )

        lane_count = clean_lanes_label(
            data.get("lanes")
        )

        popup_html = folium.Popup(
            (
                "<div style='font-family: sans-serif; "
                "font-size: 13px; line-height: 1.5;'>"
                f"<b style='font-size: 14px;'>"
                f"🤖 AI Road Risk</b><br><br>"
                f"🚨 <b>Risk Level:</b> {risk_label}<br>"
                f"🤖 <b>AI Risk Score:</b> {risk:.2f}<br>"
                f"🕒 <b>Travel Period:</b> {time_period}<br>"
                f"📏 <b>Segment Length:</b> "
                f"{segment_length_m:.0f} m<br>"
                f"🛣️ <b>Road Type:</b> {road_type}<br>"
                f"🚗 <b>Speed Limit:</b> {speed_limit}<br>"
                f"🛣️ <b>Lanes:</b> {lane_count}"
                "</div>"
            ),
            max_width=250,
        )

        folium.PolyLine(
            locations=coordinates,
            color=risk_color,
            weight=5,
            opacity=0.75,
            popup=popup_html,
            tooltip=(
                f"{risk_label} "
                f"({risk:.2f}) — click for details"
            )
        ).add_to(
            risk_layer
        )

    risk_layer.add_to(m)

    # ======================================================
    # SHORTEST ROUTE
    # ======================================================

    shortest_layer = folium.FeatureGroup(
        name="🚗 Shortest Route"
    )

    folium.PolyLine(
        locations=shortest_points,
        color="blue",
        weight=7,
        opacity=0.9,
        tooltip="🚗 Shortest Route",
        popup=(
            "<b>Shortest Route</b><br>"
            f"Travel period: "
            f"{time_period}"
        )
    ).add_to(
        shortest_layer
    )

    shortest_layer.add_to(m)

    # ======================================================
    # SAFEST ROUTE
    # ======================================================

    safest_layer = folium.FeatureGroup(
        name="🛡️ Safest Route"
    )

    folium.PolyLine(
        locations=safest_points,
        color="#8e44ad",
        weight=7,
        opacity=0.9,
        tooltip="🛡️ Safest Route",
        popup=(
            "<b>Safest Route</b><br>"
            f"Travel period: "
            f"{time_period}"
        )
    ).add_to(
        safest_layer
    )

    safest_layer.add_to(m)

    # ======================================================
    # START MARKER
    # ======================================================

    folium.Marker(
        location=shortest_points[0],
        popup="🚗 Start Location",
        tooltip="Start",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(m)

    # ======================================================
    # DESTINATION MARKER
    # ======================================================

    folium.Marker(
        location=shortest_points[-1],
        popup="🏁 Destination",
        tooltip="Destination",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(m)

    # ======================================================
    # LEGEND
    # ======================================================

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 12px;
        border: 2px solid grey;
        border-radius: 8px;
        font-size: 13px;
        box-shadow: 2px 2px 6px
        rgba(0,0,0,0.3);
    ">

        <b>🤖 AI Road Risk</b><br>
        <small>{time_period}</small>
        <br><br>

        <span style="color:green;">━</span>
        Low Risk<br>

        <span style="color:orange;">━</span>
        Medium Risk<br>

        <span style="color:red;">━</span>
        High Risk<br><br>

        <b>Routes</b><br>

        <span style="color:blue;">━</span>
        Shortest<br>

        <span style="color:#8e44ad;">━</span>
        Safest
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )

    # ======================================================
    # HAZARD HOTSPOTS
    # ======================================================
    # Added before LayerControl, since Leaflet's layer
    # control must be initialized after all overlay layers
    # already exist on the map.

    if hotspots:

        m = add_hotspot_markers_to_map(
            m,
            hotspots
        )

    # ======================================================
    # LAYER CONTROL
    # ======================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # ======================================================
    # FIT MAP TO BOTH ROUTES
    # ======================================================

    lats = [
        point[0]
        for point in all_points
    ]

    lons = [
        point[1]
        for point in all_points
    ]

    m.fit_bounds(
        [
            [
                min(lats),
                min(lons)
            ],
            [
                max(lats),
                max(lons)
            ]
        ],
        padding=(30, 30)
    )

    return m