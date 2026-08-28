# ==========================================================
# SAFEROUTE PRE-DEPLOYMENT VALIDATION
# validate_saferoute.py
# ==========================================================
#
# Runs the routing / risk-model / recommendation-logic tests
# from the pre-deployment plan. Doesn't touch Streamlit at
# all — calls utils.py functions directly so you get fast,
# repeatable, scriptable results.
#
# Run with:
#   python validate_saferoute.py
#
# Distances are printed so you can manually cross-check them
# against Google Maps / another trusted mapping service —
# that comparison can't be automated here since it needs an
# external source of truth.
# ==========================================================

import traceback

from utils import (
    load_graph,
    get_nearest_nodes,
    get_shortest_route,
    get_safest_route,
    calculate_route_risk,
    calculate_route_risk_breakdown,
    generate_route_explanation,
)


results_log = []


def record(section, name, passed, detail=""):

    results_log.append({
        "section": section,
        "name": name,
        "passed": passed,
        "detail": detail,
    })

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {section} — {name}")

    if detail:
        print(f"       {detail}")


# ==========================================================
# SECTION 1 — ROUTING VALIDATION
# ==========================================================

def run_routing_tests(G):

    print("\n" + "=" * 60)
    print("SECTION 1 — ROUTING VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------
    # VALID ROUTE PAIRS
    # ------------------------------------------------------
    # (label, start, destination)

    valid_pairs = [
        ("Nairobi -> JKIA", "Nairobi", "JKIA"),
        ("Westlands -> CBD", "Westlands", "CBD"),
        ("CBD -> Karen", "CBD", "Karen"),
        ("Short route (Westlands -> Parklands)", "Westlands", "Parklands"),
        ("Long route (Karen -> JKIA)", "Karen", "JKIA"),
        ("Ambiguous name (Karen)", "Karen", "town centre"),
    ]

    for label, start, dest in valid_pairs:

        try:

            start_node, end_node = get_nearest_nodes(
                G, start, dest
            )

            shortest_route, shortest_distance = get_shortest_route(
                G, start_node, end_node
            )

            safest_route, safest_distance = get_safest_route(
                G, start_node, end_node, "Morning"
            )

            record(
                "Routing",
                label,
                True,
                (
                    f"shortest={shortest_distance / 1000:.2f} km, "
                    f"safest={safest_distance / 1000:.2f} km "
                    "-- cross-check these against Google Maps"
                ),
            )

        except Exception as e:

            record(
                "Routing",
                label,
                False,
                f"Raised {type(e).__name__}: {e}",
            )

    # ------------------------------------------------------
    # INVALID LOCATION — SHOULD FAIL GRACEFULLY
    # ------------------------------------------------------

    try:

        get_nearest_nodes(
            G,
            "asdkjhaslkdjhaslkdjh nonexistent place 12345",
            "JKIA",
        )

        record(
            "Routing",
            "Invalid location raises a clear error",
            False,
            "Expected a ValueError but none was raised.",
        )

    except ValueError as e:

        record(
            "Routing",
            "Invalid location raises a clear error",
            True,
            f"Raised ValueError as expected: {e}",
        )

    except Exception as e:

        record(
            "Routing",
            "Invalid location raises a clear error",
            False,
            (
                f"Raised {type(e).__name__} instead of "
                f"ValueError: {e}"
            ),
        )

    # ------------------------------------------------------
    # SAME START / DESTINATION
    # ------------------------------------------------------
    # Expect a clear, graceful error here (like most mapping
    # apps do), not a meaningless 0-length route.

    try:

        start_node, end_node = get_nearest_nodes(
            G, "Westlands", "Westlands"
        )

        get_shortest_route(
            G, start_node, end_node
        )

        record(
            "Routing",
            "Same start/destination raises a clear error",
            False,
            "Expected a ValueError but a route was returned.",
        )

    except ValueError as e:

        record(
            "Routing",
            "Same start/destination raises a clear error",
            True,
            f"Raised ValueError as expected: {e}",
        )

    except Exception as e:

        record(
            "Routing",
            "Same start/destination raises a clear error",
            False,
            (
                f"Raised {type(e).__name__} instead of "
                f"ValueError: {e}"
            ),
        )

    # ------------------------------------------------------
    # LOCATION OUTSIDE NAIROBI
    # ------------------------------------------------------
    # Tests the coordinate-based distance check directly with
    # known real-world coordinates (Mombasa city), rather than
    # through geocoding -- geocoding "Mombasa" through the app
    # is unreliable to test with because SafeRoute's geocoder
    # appends ", Nairobi, Kenya" to any query that doesn't
    # mention Nairobi, and "Mombasa Road" is a real, major road
    # inside Nairobi, so that specific query can legitimately
    # resolve to an in-network location.

    from utils import validate_location_within_range

    nairobi_cbd_lat, nairobi_cbd_lon = -1.2864, 36.8172
    mombasa_city_lat, mombasa_city_lon = -4.0435, 39.6682

    try:

        validate_location_within_range(
            "Test location",
            mombasa_city_lat,
            mombasa_city_lon,
            nairobi_cbd_lat,
            nairobi_cbd_lon,
        )

        record(
            "Routing",
            "Location outside Nairobi handled",
            False,
            (
                "No error was raised for a location "
                "~480 km outside the road network."
            ),
        )

    except ValueError as e:

        record(
            "Routing",
            "Location outside Nairobi handled",
            True,
            f"Raised ValueError as expected: {e}",
        )

    except Exception as e:

        record(
            "Routing",
            "Location outside Nairobi handled",
            False,
            f"Raised {type(e).__name__}: {e}",
        )

    # Sanity check the other direction too: a location
    # genuinely within range should NOT raise.

    try:

        westlands_lat, westlands_lon = -1.2675, 36.8108

        validate_location_within_range(
            "Test location",
            westlands_lat,
            westlands_lon,
            nairobi_cbd_lat,
            nairobi_cbd_lon,
        )

        record(
            "Routing",
            "In-range location does not falsely trigger",
            True,
            "No error raised for a nearby location, as expected.",
        )

    except Exception as e:

        record(
            "Routing",
            "In-range location does not falsely trigger",
            False,
            f"Raised {type(e).__name__}: {e}",
        )


# ==========================================================
# SECTION 2 — RISK MODEL VALIDATION
# ==========================================================

def run_risk_model_tests(G):

    print("\n" + "=" * 60)
    print("SECTION 2 — RISK MODEL VALIDATION")
    print("=" * 60)

    label = "Westlands -> CBD"

    try:

        start_node, end_node = get_nearest_nodes(
            G, "Westlands", "CBD"
        )

    except Exception as e:

        record(
            "Risk Model",
            "Setup route for time-period tests",
            False,
            f"Could not geocode test route: {e}",
        )

        return

    time_periods = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
    ]

    risks_by_period = {}

    for period in time_periods:

        try:

            safest_route, _ = get_safest_route(
                G, start_node, end_node, period
            )

            risk = calculate_route_risk(
                G, safest_route, period
            )

            risks_by_period[period] = risk

            record(
                "Risk Model",
                f"Compute risk for {period}",
                True,
                f"risk={risk:.3f}",
            )

        except Exception as e:

            record(
                "Risk Model",
                f"Compute risk for {period}",
                False,
                f"Raised {type(e).__name__}: {e}",
            )

    # ------------------------------------------------------
    # SANITY CHECK: night should generally not be the
    # lowest-risk period, given the TIME_PERIODS multiplier
    # used when the dataset was generated (Night = 1.50x,
    # the highest of the four).
    # ------------------------------------------------------

    if len(risks_by_period) == 4:

        night_is_lowest = (
            risks_by_period["Night"]
            == min(risks_by_period.values())
        )

        record(
            "Risk Model",
            "Night is not the lowest-risk period",
            not night_is_lowest,
            f"risks by period: {risks_by_period}",
        )

    # ------------------------------------------------------
    # SAFEST ROUTE SHOULD HAVE RISK <= SHORTEST ROUTE
    # ------------------------------------------------------

    try:

        shortest_route, _ = get_shortest_route(
            G, start_node, end_node
        )

        safest_route, _ = get_safest_route(
            G, start_node, end_node, "Morning"
        )

        shortest_risk = calculate_route_risk(
            G, shortest_route, "Morning"
        )

        safest_risk = calculate_route_risk(
            G, safest_route, "Morning"
        )

        record(
            "Risk Model",
            "Safest route risk <= shortest route risk",
            safest_risk <= shortest_risk,
            (
                f"shortest_risk={shortest_risk:.3f}, "
                f"safest_risk={safest_risk:.3f}"
            ),
        )

        breakdown = calculate_route_risk_breakdown(
            G, safest_route, "Morning"
        )

        breakdown_sums_to_100 = abs(
            sum(breakdown.values()) - 100
        ) < 0.5 or sum(breakdown.values()) == 0

        record(
            "Risk Model",
            "Risk breakdown percentages sum to ~100%",
            breakdown_sums_to_100,
            f"breakdown={breakdown}",
        )

    except Exception as e:

        record(
            "Risk Model",
            "Safest vs shortest risk comparison",
            False,
            f"Raised {type(e).__name__}: {e}",
        )


# ==========================================================
# SECTION 3 — RECOMMENDATION LOGIC VALIDATION
# ==========================================================

def run_recommendation_tests():

    print("\n" + "=" * 60)
    print("SECTION 3 — RECOMMENDATION LOGIC VALIDATION")
    print("=" * 60)

    neutral_breakdown_shortest = {
        "low": 40,
        "medium": 30,
        "high": 30,
    }

    neutral_breakdown_safest = {
        "low": 70,
        "medium": 20,
        "high": 10,
    }

    # ------------------------------------------------------
    # CASE A: 5% longer, 30% lower risk -> expect Safest Route
    # ------------------------------------------------------

    shortest_distance = 10000  # 10 km
    safest_distance = 10500    # 5% longer

    shortest_risk = 1.0
    safest_risk = 0.70          # 30% lower

    explanation = generate_route_explanation(
        shortest_risk,
        safest_risk,
        shortest_distance,
        safest_distance,
        neutral_breakdown_shortest,
        neutral_breakdown_safest,
    )

    record(
        "Recommendation Logic",
        "5% longer / 30% lower risk -> Safest Route",
        explanation["recommendation"] == "Safest Route",
        f"got '{explanation['recommendation']}'",
    )

    # ------------------------------------------------------
    # CASE B: 60% longer, 3% lower risk -> expect Shortest Route
    # ------------------------------------------------------

    shortest_distance = 10000
    safest_distance = 16000     # 60% longer

    shortest_risk = 1.0
    safest_risk = 0.97          # 3% lower

    explanation = generate_route_explanation(
        shortest_risk,
        safest_risk,
        shortest_distance,
        safest_distance,
        neutral_breakdown_shortest,
        neutral_breakdown_safest,
    )

    record(
        "Recommendation Logic",
        "60% longer / 3% lower risk -> Shortest Route",
        explanation["recommendation"] == "Shortest Route",
        f"got '{explanation['recommendation']}'",
    )

    # ------------------------------------------------------
    # CASE C: identical routes -> expect Shortest Route,
    # no crash on zero-division
    # ------------------------------------------------------

    explanation = generate_route_explanation(
        0.5, 0.5, 5000, 5000,
        neutral_breakdown_shortest,
        neutral_breakdown_safest,
    )

    record(
        "Recommendation Logic",
        "Identical routes handled without error",
        explanation["recommendation"] == "Shortest Route",
        f"got '{explanation['recommendation']}'",
    )

    # ------------------------------------------------------
    # CASE D: zero shortest_distance -> no crash
    # ------------------------------------------------------

    try:

        explanation = generate_route_explanation(
            0.0, 0.0, 0, 0,
            neutral_breakdown_shortest,
            neutral_breakdown_safest,
        )

        record(
            "Recommendation Logic",
            "Zero-distance edge case does not crash",
            True,
            f"got '{explanation['recommendation']}'",
        )

    except Exception as e:

        record(
            "Recommendation Logic",
            "Zero-distance edge case does not crash",
            False,
            f"Raised {type(e).__name__}: {e}",
        )


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary():

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results_log)
    passed = sum(1 for r in results_log if r["passed"])
    failed = total - passed

    print(f"Total checks: {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")

    if failed:

        print("\nFailed checks:")

        for r in results_log:

            if not r["passed"]:

                print(f"  - [{r['section']}] {r['name']}")
                if r["detail"]:
                    print(f"    {r['detail']}")

    print(
        "\nNote: routing distances above are NOT auto-verified "
        "against an external source. Cross-check the printed "
        "km values against Google Maps for each route pair."
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Loading graph (this can take a moment)...")

    try:

        G = load_graph()

    except Exception as e:

        print(f"FATAL: could not load graph: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    run_routing_tests(G)
    run_risk_model_tests(G)
    run_recommendation_tests()

    print_summary()