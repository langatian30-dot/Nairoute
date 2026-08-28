import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="SafeRoute Map Test",
    layout="wide"
)

st.title("🗺️ SafeRoute Map Test")

m = folium.Map(
    location=[-1.286389, 36.817223],
    zoom_start=12,
    tiles="OpenStreetMap"
)

st_folium(
    m,
    width=1200,
    height=700,
    returned_objects=[]
)