# ==========================================================
# SAFEROUTE V4.0
# app.py
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium

from utils import (
    load_graph,
    get_nearest_nodes,
    get_shortest_route,
    get_safest_route,
    calculate_route_risk,
    calculate_route_risk_breakdown,
    generate_route_explanation,
    identify_route_hotspots,
    create_route_map,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SafeRoute",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🛡️ SafeRoute")

st.markdown(
    """
    ### AI-Powered Safe Route Recommendation System

    Find the **shortest** and **safest** driving routes in Nairobi using:

    - 🚗 OpenStreetMap Road Network
    - 🤖 Machine Learning Road Risk Prediction
    - 🚨 Crime Hotspot Analysis
    - 🕒 Time-Aware Risk Prediction
    """
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "results" not in st.session_state:
    st.session_state.results = None

# ==========================================================
# ROUTE HISTORY
# ==========================================================

if "route_history" not in st.session_state:
    st.session_state.route_history = []


# ==========================================================
# SIDEBAR — ROUTE PLANNER
# ==========================================================

with st.sidebar:

    st.header("📍 Route Planner")

    start_location = st.text_input(
        "Start Location",
        placeholder="e.g. Westlands, Nairobi",
        key="start_location",
    )

    destination = st.text_input(
        "Destination",
        placeholder="e.g. JKIA, Nairobi",
        key="destination",
    )

    time_period = st.selectbox(
        "Travel Time",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
        ],
        key="time_period",
    )

    calculate_route = st.button(
        "🛡️ Calculate Safe Route",
        use_container_width=True,
        key="calculate_safe_route",
    )


# ==========================================================
# ROUTE CALCULATION
# ==========================================================

if calculate_route:

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if not start_location.strip():

        st.error(
            "Please enter a start location."
        )

    elif not destination.strip():

        st.error(
            "Please enter a destination."
        )

    elif (
        start_location.strip().lower()
        == destination.strip().lower()
    ):

        st.error(
            "Start location and destination are the same. "
            "Please enter two different locations."
        )

    else:

        try:

            with st.spinner(
                "🧠 Calculating routes..."
            ):

                # --------------------------------------------------
                # LOAD GRAPH
                # --------------------------------------------------

                G = load_graph()

                # --------------------------------------------------
                # FIND START / DESTINATION NODES
                # --------------------------------------------------

                start_node, end_node = get_nearest_nodes(
                    G,
                    start_location,
                    destination
                )

                st.info(
                    f"""
                    📍 **Route nodes selected**

                    Start node: `{start_node}`

                    Destination node: `{end_node}`

                    Graph CRS: `{G.graph.get("crs")}`
                    """
                )

                # --------------------------------------------------
                # SHORTEST ROUTE
                # --------------------------------------------------

                shortest_route, shortest_distance = (
                    get_shortest_route(
                        G,
                        start_node,
                        end_node,
                    )
                )

                shortest_risk = (
                    calculate_route_risk(
                        G,
                        shortest_route,
                        time_period,
                    )
                )

                # --------------------------------------------------
                # SAFEST ROUTE
                # --------------------------------------------------

                safest_route, safest_distance = (
                    get_safest_route(
                        G,
                        start_node,
                        end_node,
                        time_period,
                    )
                )

                safest_risk = (
                    calculate_route_risk(
                        G,
                        safest_route,
                        time_period,
                    )
                )

                # --------------------------------------------------
                # RISK BREAKDOWN
                # --------------------------------------------------

                shortest_breakdown = (
                    calculate_route_risk_breakdown(
                        G,
                        shortest_route,
                        time_period,
                    )
                )

                safest_breakdown = (
                    calculate_route_risk_breakdown(
                        G,
                        safest_route,
                        time_period,
                    )
                )

                # --------------------------------------------------
                # HAZARD HOTSPOTS
                # --------------------------------------------------

                safest_hotspots = identify_route_hotspots(
                    G,
                    safest_route,
                    time_period,
                )

                # --------------------------------------------------
                # CREATE MAP
                # --------------------------------------------------

                route_map = create_route_map(
                    G,
                    shortest_route,
                    safest_route,
                    time_period,
                    hotspots=safest_hotspots,
                )

                # --------------------------------------------------
                # SAVE RESULTS
                # --------------------------------------------------

                st.session_state.results = {

                    "graph": G,

                    "shortest_route": shortest_route,
                    "shortest_distance": shortest_distance,
                    "shortest_risk": shortest_risk,
                    "shortest_breakdown": shortest_breakdown,

                    "safest_route": safest_route,
                    "safest_distance": safest_distance,
                    "safest_risk": safest_risk,
                    "safest_breakdown": safest_breakdown,
                    "safest_hotspots": safest_hotspots,

                    "route_map": route_map,

                    "time_period": time_period,
                }

                # ==================================================
                # SAVE ROUTE TO HISTORY
                # ==================================================

                st.session_state.route_history.append({

                    "start": start_location,

                    "destination": destination,

                    "time_period": time_period,

                    "shortest_distance_km": (
                        shortest_distance / 1000
                    ),

                    "safest_distance_km": (
                        safest_distance / 1000
                    ),

                    "shortest_risk": shortest_risk,

                    "safest_risk": safest_risk,

                    "risk_reduction": risk_reduction
                    if "risk_reduction" in locals()
                    else 0,

                    "recommendation": (
                        "Safest Route"
                        if (
                            shortest_risk > 0
                            and (
                                (
                                    shortest_risk
                                    - safest_risk
                                )
                                / shortest_risk
                            ) * 100 >= 15
                            and (
                                (
                                    safest_distance
                                    - shortest_distance
                                )
                                / shortest_distance
                            ) * 100 <= 30
                        )
                        else "Shortest Route"
                    ),
                })

            st.success(
                "✅ Routes calculated successfully!"
            )

        except ValueError as e:

            # ValueErrors raised by SafeRoute's own validation
            # (invalid location, same start/destination, too
            # far from the road network, etc.) already carry a
            # clear, user-facing message — show it directly
            # instead of a raw traceback.

            st.error(
                f"⚠️ {e}"
            )

        except Exception as e:

            st.error(
                "❌ Route execution failed unexpectedly. "
                "If this keeps happening, please try a "
                "different location or travel time."
            )

            with st.expander("Technical details"):
                st.exception(e)


# ==========================================================
# RESULTS
# ==========================================================

if st.session_state.get("results") is not None:

    results = st.session_state.results

    # ======================================================
    # ROUTE RESULTS
    # ======================================================

    st.markdown("---")

    st.header("🗺️ Route Results")

    time_period = results.get(
        "time_period",
        "Morning",
    )

    st.write(
        f"🕒 Travel period: **{time_period}**"
    )


    # ======================================================
    # ROUTE VALUES
    # ======================================================

    shortest_distance = float(
        results["shortest_distance"]
    )

    safest_distance = float(
        results["safest_distance"]
    )

    shortest_risk = float(
        results["shortest_risk"]
    )

    safest_risk = float(
        results["safest_risk"]
    )


    # ======================================================
    # ROUTE COMPARISON
    # ======================================================

    st.subheader("📊 Route Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🚗 Shortest Route"
        )

        st.metric(
            "Distance",
            f"{shortest_distance / 1000:.2f} km",
        )

        st.metric(
            "Risk Score",
            f"{shortest_risk:.2f}",
        )

    with col2:

        st.markdown(
            "### 🛡️ Safest Route"
        )

        st.metric(
            "Distance",
            f"{safest_distance / 1000:.2f} km",
        )

        st.metric(
            "Risk Score",
            f"{safest_risk:.2f}",
        )


    # ======================================================
    # SAFETY BREAKDOWN
    # ======================================================

    st.subheader(
        "📊 Route Safety Breakdown"
    )

    shortest_breakdown = results.get(
        "shortest_breakdown",
        {
            "low": 0,
            "medium": 0,
            "high": 0,
        },
    )

    safest_breakdown = results.get(
        "safest_breakdown",
        {
            "low": 0,
            "medium": 0,
            "high": 0,
        },
    )

    breakdown_data = {

        "Route": [
            "🚗 Shortest Route",
            "🛡️ Safest Route",
        ],

        "🟢 Low Risk": [
            shortest_breakdown.get(
                "low",
                0,
            ),
            safest_breakdown.get(
                "low",
                0,
            ),
        ],

        "🟠 Medium Risk": [
            shortest_breakdown.get(
                "medium",
                0,
            ),
            safest_breakdown.get(
                "medium",
                0,
            ),
        ],

        "🔴 High Risk": [
            shortest_breakdown.get(
                "high",
                0,
            ),
            safest_breakdown.get(
                "high",
                0,
            ),
        ],
    }

    st.dataframe(
        breakdown_data,
        use_container_width=True,
        hide_index=True,
    )

    # ======================================================
    # ROUTE HAZARD VISUALIZATION
    # ======================================================

    st.subheader("📊 Route Hazard Visualization")

    st.caption(
        f"Risk exposure across road segments during the "
        f"{time_period} travel period."
    )

    hazard_chart_data = pd.DataFrame([
        {
            "Route": "🚗 Shortest Route",
            "Risk Level": "🟢 Low Risk",
            "Percentage": float(
                shortest_breakdown.get("low", 0)
            ),
        },
        {
            "Route": "🚗 Shortest Route",
            "Risk Level": "🟠 Medium Risk",
            "Percentage": float(
                shortest_breakdown.get("medium", 0)
            ),
        },
        {
            "Route": "🚗 Shortest Route",
            "Risk Level": "🔴 High Risk",
            "Percentage": float(
                shortest_breakdown.get("high", 0)
            ),
        },
        {
            "Route": "🛡️ Safest Route",
            "Risk Level": "🟢 Low Risk",
            "Percentage": float(
                safest_breakdown.get("low", 0)
            ),
        },
        {
            "Route": "🛡️ Safest Route",
            "Risk Level": "🟠 Medium Risk",
            "Percentage": float(
                safest_breakdown.get("medium", 0)
            ),
        },
        {
            "Route": "🛡️ Safest Route",
            "Risk Level": "🔴 High Risk",
            "Percentage": float(
                safest_breakdown.get("high", 0)
            ),
        },
    ])

    hazard_fig = px.bar(
        hazard_chart_data,
        x="Route",
        y="Percentage",
        color="Risk Level",
        barmode="stack",
        title="Route Risk Composition",
        color_discrete_map={
            "🟢 Low Risk": "#2ecc71",
            "🟠 Medium Risk": "#f39c12",
            "🔴 High Risk": "#e74c3c",
        },
        range_y=[0, 100],
    )

    hazard_fig.update_layout(
        yaxis_title="% of Route Segments",
        xaxis_title=None,
        legend_title=None,
    )

    st.plotly_chart(
        hazard_fig,
        use_container_width=True,
    )

    st.caption(
        "The chart shows how the road segments of each candidate "
        "route are distributed across the three AI risk levels."
    )

    # ======================================================
    # ROUTE HAZARD SUMMARY
    # ======================================================

    st.subheader("🚨 Route Hazard Summary")

    safest_breakdown = results.get(
        "safest_breakdown",
        {
            "low": 0,
            "medium": 0,
            "high": 0
        }
    )

    # ------------------------------------------------------
    # SAFEST ROUTE
    # ------------------------------------------------------

    safe_low = float(
        safest_breakdown.get("low", 0)
    )

    safe_medium = float(
        safest_breakdown.get("medium", 0)
    )

    safe_high = float(
        safest_breakdown.get("high", 0)
    )

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    st.markdown("### 🛡️ Recommended Safest Route")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🟢 Low Risk",
            f"{safe_low:.1f}%"
        )

    with col2:
        st.metric(
            "🟠 Medium Risk",
            f"{safe_medium:.1f}%"
        )

    with col3:
        st.metric(
            "🔴 High Risk",
            f"{safe_high:.1f}%"
        )

    # ------------------------------------------------------
    # SIMPLE EXPLANATION
    # ------------------------------------------------------

    if safe_high <= 10:

        st.success(
            "🛡️ Most of the recommended route consists "
            "of low-risk road segments."
        )

    elif safe_high <= 25:

        st.warning(
            "⚠️ The recommended route contains some "
            "higher-risk road segments."
        )

    else:

        st.error(
            "🚨 The recommended route contains a significant "
            "proportion of high-risk road segments."
        )

    # ======================================================
    # ROUTE HAZARD ANALYSIS
    # ======================================================

    st.subheader("🚨 Route Hazard Analysis")

    st.markdown(
        """
        SafeRoute analyses the road segments used by each route
        and identifies the overall hazard level for the selected
        travel period.
        """
    )

    # ------------------------------------------------------
    # HAZARD SUMMARY FUNCTION
    # ------------------------------------------------------

    def get_hazard_summary(breakdown):

        low = breakdown.get("low", 0)
        medium = breakdown.get("medium", 0)
        high = breakdown.get("high", 0)

        if high >= 40:

            return (
                "🔴 High hazard exposure",
                "A significant portion of this route contains "
                "high-risk road segments."
            )

        elif high >= 20 or medium >= 50:

            return (
                "🟠 Moderate hazard exposure",
                "This route contains a noticeable amount of "
                "medium or high-risk road segments."
            )

        else:

            return (
                "🟢 Low hazard exposure",
                "Most of this route consists of lower-risk "
                "road segments."
            )


    # ------------------------------------------------------
    # SHORTEST ROUTE HAZARDS
    # ------------------------------------------------------

    shortest_hazard_title, shortest_hazard_text = (
        get_hazard_summary(
            shortest_breakdown
        )
    )

    # ------------------------------------------------------
    # SAFEST ROUTE HAZARDS
    # ------------------------------------------------------

    safest_hazard_title, safest_hazard_text = (
        get_hazard_summary(
            safest_breakdown
        )
    )

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    hazard_col1, hazard_col2 = st.columns(2)

    with hazard_col1:

        st.markdown("### 🚗 Shortest Route")

        st.write(
            f"**{shortest_hazard_title}**"
        )

        st.caption(
            shortest_hazard_text
        )

        st.write(
            f"🟢 Low-risk segments: "
            f"**{shortest_breakdown.get('low', 0):.1f}%**"
        )

        st.write(
            f"🟠 Medium-risk segments: "
            f"**{shortest_breakdown.get('medium', 0):.1f}%**"
        )

        st.write(
            f"🔴 High-risk segments: "
            f"**{shortest_breakdown.get('high', 0):.1f}%**"
        )


    with hazard_col2:

        st.markdown("### 🛡️ Safest Route")

        st.write(
            f"**{safest_hazard_title}**"
        )

        st.caption(
            safest_hazard_text
        )

        st.write(
            f"🟢 Low-risk segments: "
            f"**{safest_breakdown.get('low', 0):.1f}%**"
        )

        st.write(
            f"🟠 Medium-risk segments: "
            f"**{safest_breakdown.get('medium', 0):.1f}%**"
        )

        st.write(
            f"🔴 High-risk segments: "
            f"**{safest_breakdown.get('high', 0):.1f}%**"
        )

    # ======================================================
    # ROUTE HAZARD HOTSPOTS
    # ======================================================

    st.subheader("📍 Route Hazard Hotspots")

    safest_hotspots = results.get(
        "safest_hotspots",
        []
    )

    if safest_hotspots:

        st.warning(
            f"🚨 {len(safest_hotspots)} hazard hotspot"
            f"{'s' if len(safest_hotspots) != 1 else ''} "
            "detected along the recommended route."
        )

        for i, hotspot in enumerate(safest_hotspots, start=1):

            hotspot_icon = (
                "🔴"
                if hotspot["risk_label"] == "High Risk"
                else "🟠"
            )

            with st.container():

                st.markdown(
                    f"**{hotspot_icon} Hotspot {i} — "
                    f"{hotspot['risk_label']}**"
                )

                st.write(
                    f"- 🤖 Risk: **{hotspot['risk']:.2f}**"
                )

                st.write(
                    f"- 📍 Near: **{hotspot['near']}**"
                )

                st.write(
                    f"- 🕒 Travel period: **{time_period}**"
                )

                st.write(
                    f"- 📏 Segment length: "
                    f"**{hotspot['length_m']:.0f} m**"
                )

    else:

        st.success(
            "🟢 No significant hazard hotspots detected "
            "along the recommended route."
        )

    # ======================================================
    # AI ROUTE EXPLANATION
    # ======================================================

    st.subheader(
        "🧠 AI Route Explanation"
    )

    explanation = generate_route_explanation(
        shortest_risk,
        safest_risk,
        shortest_distance,
        safest_distance,
        shortest_breakdown,
        safest_breakdown,
    )

    recommendation = explanation["recommendation"]
    risk_reduction = explanation["risk_reduction"]
    distance_increase = explanation["distance_increase"]
    extra_distance_km = explanation["extra_distance_km"]

    explanation_box = (
        st.success
        if recommendation == "Safest Route"
        else st.info
    )

    explanation_box(
        f"""
**{explanation['headline']}**

{explanation['summary']}

{explanation['reasoning']}

**Trade-off:** {'+' if extra_distance_km >= 0 else ''}{extra_distance_km:.1f} km
**Risk reduction:** {risk_reduction:.0f}%
**Decision:** {explanation['decision_text']}
"""
    )


    # ======================================================
    # AI METRICS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📉 Risk Reduction",
            f"{risk_reduction:.1f}%",
        )

    with col2:

        st.metric(
            "📏 Extra Distance",
            f"{extra_distance_km:.2f} km",
        )

    with col3:

        st.metric(
            "📊 Distance Increase",
            f"{distance_increase:.1f}%",
        )


    # ======================================================
    # EXPLANATION
    # ======================================================

    with st.expander(
        "🔍 Why did SafeRoute choose this route?"
    ):

        st.write(
            f"Shortest route: "
            f"**{shortest_distance / 1000:.2f} km**"
        )

        st.write(
            f"Shortest route risk: "
            f"**{shortest_risk:.2f}**"
        )

        st.write(
            f"Safest route: "
            f"**{safest_distance / 1000:.2f} km**"
        )

        st.write(
            f"Safest route risk: "
            f"**{safest_risk:.2f}**"
        )

        st.write(
            f"Risk reduction: "
            f"**{risk_reduction:.1f}%**"
        )

        st.write(
            f"Additional distance: "
            f"**{extra_distance_km:.2f} km**"
        )

        st.write(
            f"Final recommendation: "
            f"**{recommendation}**"
        )

    # ======================================================
    # SAFEROUTE ANALYTICS DASHBOARD
    # ======================================================

    st.subheader("📊 SafeRoute Analytics")

    # ------------------------------------------------------
    # DASHBOARD METRICS
    # ------------------------------------------------------

    dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = (
        st.columns(4)
    )

    with dashboard_col1:

        st.metric(
            "🚗 Shortest Route",
            f"{shortest_distance / 1000:.2f} km"
        )

    with dashboard_col2:

        st.metric(
            "🛡️ Safest Route",
            f"{safest_distance / 1000:.2f} km"
        )

    with dashboard_col3:

        st.metric(
            "📉 Risk Reduction",
            f"{risk_reduction:.1f}%"
        )

    with dashboard_col4:

        st.metric(
            "📏 Extra Distance",
            f"{extra_distance_km:.2f} km"
        )

    # ------------------------------------------------------
    # ROUTE ANALYTICS TABLE
    # ------------------------------------------------------

    st.markdown("#### Route Performance")

    analytics_data = {
        "Metric": [
            "Distance",
            "Risk Score",
            "Low Risk Roads",
            "Medium Risk Roads",
            "High Risk Roads",
        ],

        "🚗 Shortest Route": [
            f"{shortest_distance / 1000:.2f} km",
            f"{shortest_risk:.2f}",
            f"{shortest_breakdown.get('low', 0):.1f}%",
            f"{shortest_breakdown.get('medium', 0):.1f}%",
            f"{shortest_breakdown.get('high', 0):.1f}%",
        ],

        "🛡️ Safest Route": [
            f"{safest_distance / 1000:.2f} km",
            f"{safest_risk:.2f}",
            f"{safest_breakdown.get('low', 0):.1f}%",
            f"{safest_breakdown.get('medium', 0):.1f}%",
            f"{safest_breakdown.get('high', 0):.1f}%",
        ],
    }

    st.dataframe(
        analytics_data,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------
    # DECISION SUMMARY
    # ------------------------------------------------------

    st.markdown("#### 🧠 Decision Summary")

    if recommendation == "Safest Route":

        st.success(
            """
            🛡️ **SafeRoute recommends the Safest Route.**

            The alternative route provides a meaningful reduction
            in predicted road risk without an excessive increase
            in travel distance.
            """
        )

    else:

        st.info(
            """
            🚗 **SafeRoute recommends the Shortest Route.**

            The additional distance required by the safer route
            is not sufficiently justified by the predicted safety
            improvement.
            """
        )

    # ======================================================
    # ROUTE HISTORY
    # ======================================================

    st.subheader("📜 Route History")

    if st.session_state.route_history:

        history_data = []

        for i, trip in enumerate(
            reversed(st.session_state.route_history),
            start=1
        ):

            history_data.append({

                "#": i,

                "📍 Start": trip["start"],

                "🏁 Destination": trip["destination"],

                "🕒 Time": trip["time_period"],

                 "🚗 Shortest": (
                    f'{trip["shortest_distance_km"]:.2f} km'
                ),

                "🛡️ Safest": (
                    f'{trip["safest_distance_km"]:.2f} km'
                ),

                "🚨 Shortest Risk": (
                    f'{trip["shortest_risk"]:.2f}'
                ),

                "🚨 Safest Risk": (
                    f'{trip["safest_risk"]:.2f}'
                ),
            })

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "🗑️ Clear Route History",
            key="clear_route_history",
        ):

            st.session_state.route_history = []

            st.rerun()

    else:

        st.info(
            "📭 No previous routes yet. "
            "Calculate a route to create route history."
        )


    # ======================================================
    # INTERACTIVE MAP
    # ======================================================

    st.subheader(
        "🗺️ Interactive Route Map"
    )

    route_map = results.get(
        "route_map"
    )

    if route_map is not None:

        st_folium(
            route_map,
            width=1200,
            height=700,
            returned_objects=[],
            key="safe_route_map",
        )

    else:

        st.warning(
            "⚠️ Route map is unavailable."
        )


# ==========================================================
# NO RESULTS
# ==========================================================

else:

    st.markdown("---")

    welcome_col1, welcome_col2 = st.columns([2, 1])

    with welcome_col1:

        st.subheader("👋 Get started")

        st.write(
            "Enter a **start location** and **destination** "
            "in the sidebar, choose a travel time, then click "
            "**🛡️ Calculate Safe Route**."
        )

        st.write(
            "SafeRoute will show you the shortest route, the "
            "predicted-safest route, and explain the trade-off "
            "between them."
        )

        st.markdown("**Try an example:**")

        st.markdown(
            "- Start: `Westlands` → Destination: `CBD`\n"
            "- Start: `Karen` → Destination: `JKIA`\n"
            "- Start: `CBD` → Destination: `Eastleigh`"
        )

    with welcome_col2:

        st.info(
            "**What SafeRoute shows you:**\n\n"
            "🚗 Shortest route\n\n"
            "🛡️ Predicted-safest route\n\n"
            "🚨 Hazard hotspots along the way\n\n"
            "🧠 AI reasoning for the recommendation\n\n"
            "🗺️ An interactive map with a risk overlay"
        )

    st.caption(
        "SafeRoute currently covers Nairobi, Kenya only. "
        "Predicted risk is based on a machine learning model "
        "trained on available road and crime data — it is a "
        "prediction, not a guarantee of safety."
    )