import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

st.set_page_config(page_title="Tunisia Climate Risk Assessment", layout="wide", page_icon="🌍")

# Title and intro
st.title("🌍 Tunisia Climate Risk Assessment System")
st.markdown("**AI-powered climate risk analysis and insurance premium calculator**")

# Create synthetic Tunisia neighborhood data
@st.cache_data
def create_tunisia_data():
    neighborhoods_data = [
        {"name": "Tunis Centre Ville", "governorate": "Tunis", "lat": 36.8065, "lon": 10.1815, "population": 50000, "area_km2": 5.2, "coastal": True},
        {"name": "La Marsa", "governorate": "Tunis", "lat": 36.8781, "lon": 10.3250, "population": 92000, "area_km2": 18.5, "coastal": True},
        {"name": "Carthage", "governorate": "Tunis", "lat": 36.8526, "lon": 10.3233, "population": 21000, "area_km2": 7.8, "coastal": True},
        {"name": "Sfax Centre", "governorate": "Sfax", "lat": 34.7405, "lon": 10.7603, "population": 272801, "area_km2": 56.2, "coastal": True},
        {"name": "Sousse Medina", "governorate": "Sousse", "lat": 35.8256, "lon": 10.6369, "population": 221530, "area_km2": 45.0, "coastal": True},
        {"name": "Bizerte Centre", "governorate": "Bizerte", "lat": 37.2744, "lon": 9.8739, "population": 142966, "area_km2": 32.1, "coastal": True},
        {"name": "Hammamet", "governorate": "Nabeul", "lat": 36.4000, "lon": 10.6167, "population": 73236, "area_km2": 36.8, "coastal": True},
        {"name": "Kairouan Medina", "governorate": "Kairouan", "lat": 35.6781, "lon": 10.0963, "population": 186653, "area_km2": 42.0, "coastal": False},
        {"name": "Gabès Centre", "governorate": "Gabès", "lat": 33.8815, "lon": 10.0982, "population": 152921, "area_km2": 38.6, "coastal": True},
        {"name": "Djerba Houmt Souk", "governorate": "Medenine", "lat": 33.8076, "lon": 10.8451, "population": 75904, "area_km2": 45.2, "coastal": True},
    ]

    geometries = []
    for n in neighborhoods_data:
        polygon = Polygon([
            (n['lon'] - 0.02, n['lat'] - 0.02),
            (n['lon'] + 0.02, n['lat'] - 0.02),
            (n['lon'] + 0.02, n['lat'] + 0.02),
            (n['lon'] - 0.02, n['lat'] + 0.02),
        ])
        geometries.append(polygon)

    df = pd.DataFrame(neighborhoods_data)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf

# Calculate risk scores
def calculate_risk_scores(elevation, distance_to_coast, building_age, coastal, latitude, flood_zone, wildfire_zone):
    coastal_factor = 40 if coastal else 0
    elevation_factor = max(0, (100 - elevation) * 0.6)
    flood_zone_factor = {'Low': 0, 'Medium': 20, 'High': 40}.get(flood_zone, 20)
    flood_risk = min(100, coastal_factor + elevation_factor + flood_zone_factor)

    inland_factor = 30 if not coastal else 0
    vegetation_factor = 25 if latitude > 35.5 else 10
    wildfire_zone_factor = {'Low': 0, 'Medium': 25, 'High': 45}.get(wildfire_zone, 25)
    wildfire_risk = min(100, inland_factor + vegetation_factor + wildfire_zone_factor)

    hurricane_risk = (70 if coastal else 10) * (1 + (37 - latitude) * 0.05)
    hurricane_risk = min(100, max(0, hurricane_risk))

    latitude_factor = max(0, (36 - latitude) * 8)
    drought_risk = min(100, 30 + latitude_factor)

    age_factor = min(30, building_age * 0.3)
    flood_risk = min(100, flood_risk + age_factor * 0.3)
    wildfire_risk = min(100, wildfire_risk + age_factor * 0.25)
    hurricane_risk = min(100, hurricane_risk + age_factor * 0.3)

    composite_risk = flood_risk * 0.35 + wildfire_risk * 0.25 + hurricane_risk * 0.25 + drought_risk * 0.15

    return {
        'flood': float(flood_risk),
        'wildfire': float(wildfire_risk),
        'hurricane': float(hurricane_risk),
        'drought': float(drought_risk),
        'composite': float(composite_risk)
    }

# Calculate premium
def calculate_premium(property_value, risk_scores, building_age):
    base_rate = 0.003
    composite_risk = risk_scores.get('composite', 50)
    risk_multiplier = 1 + (composite_risk / 100) * 2
    age_multiplier = 1 + (building_age / 100) * 0.5
    base_premium = property_value * base_rate
    adjusted_premium = base_premium * risk_multiplier * age_multiplier
    loss_probability = (composite_risk / 100) * 0.05
    expected_loss = property_value * loss_probability * 0.3

    return {
        'base_annual': float(base_premium),
        'adjusted_annual': float(adjusted_premium),
        'monthly': float(adjusted_premium / 12),
        'expected_loss': float(expected_loss),
        'loss_probability': float(loss_probability),
        'risk_multiplier': float(risk_multiplier),
        'age_multiplier': float(age_multiplier)
    }

# Load data
gdf = create_tunisia_data()

# Sidebar inputs
st.sidebar.header("🏠 Property Information")
neighborhood = st.sidebar.selectbox("Select Neighborhood", gdf['name'].tolist())
property_value = st.sidebar.number_input("Property Value (TND)", min_value=50000, max_value=5000000, value=300000, step=10000)
building_age = st.sidebar.slider("Building Age (years)", 0, 100, 15)

# Get selected neighborhood data
selected_row = gdf[gdf['name'] == neighborhood].iloc[0]
lat = selected_row['lat']
lon = selected_row['lon']
coastal = selected_row['coastal']

# Additional inputs
st.sidebar.subheader("📍 Location Details")
elevation = st.sidebar.slider("Elevation (m)", 0, 500, 50 if coastal else 200)
distance_to_coast = st.sidebar.slider("Distance to Coast (km)", 0.0, 150.0, 2.0 if coastal else 50.0)
flood_zone = st.sidebar.selectbox("Flood Zone", ["Low", "Medium", "High"], index=1)
wildfire_zone = st.sidebar.selectbox("Wildfire Zone", ["Low", "Medium", "High"], index=1)

# Calculate risks
risk_scores = calculate_risk_scores(elevation, distance_to_coast, building_age, coastal, lat, flood_zone, wildfire_zone)
premium_info = calculate_premium(property_value, risk_scores, building_age)

# Main display - tabs
tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "🗺️ Map View", "💰 Premium Details"])

with tab1:
    st.subheader(f"Risk Assessment: {neighborhood}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Flood Risk", f"{risk_scores['flood']:.0f}%", delta=None)
    with col2:
        st.metric("Wildfire Risk", f"{risk_scores['wildfire']:.0f}%", delta=None)
    with col3:
        st.metric("Hurricane Risk", f"{risk_scores['hurricane']:.0f}%", delta=None)
    with col4:
        st.metric("Drought Risk", f"{risk_scores['drought']:.0f}%", delta=None)

    # Risk chart
    fig = go.Figure(data=[
        go.Bar(name='Risk Scores', x=['Flood', 'Wildfire', 'Hurricane', 'Drought'],
               y=[risk_scores['flood'], risk_scores['wildfire'], risk_scores['hurricane'], risk_scores['drought']],
               marker_color=['#3b82f6', '#ef4444', '#8b5cf6', '#f59e0b'])
    ])
    fig.update_layout(title="Climate Risk Breakdown", yaxis_title="Risk Score (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Composite Risk Score: {risk_scores['composite']:.1f}%**")

with tab2:
    st.subheader("Tunisia Climate Risk Map")

    m = folium.Map(location=[lat, lon], zoom_start=10)

    # Add marker
    folium.Marker(
        [lat, lon],
        popup=f"{neighborhood}<br>Risk: {risk_scores['composite']:.1f}%",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    st_folium(m, width=700, height=500)

with tab3:
    st.subheader("Insurance Premium Calculation")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monthly Premium", f"TND {premium_info['monthly']:,.2f}")
        st.metric("Annual Premium", f"TND {premium_info['adjusted_annual']:,.2f}")
    with col2:
        st.metric("Expected Loss", f"TND {premium_info['expected_loss']:,.2f}")
        st.metric("Loss Probability", f"{premium_info['loss_probability']*100:.2f}%")

    st.write("**Premium Breakdown:**")
    st.write(f"- Base Premium: TND {premium_info['base_annual']:,.2f}")
    st.write(f"- Risk Multiplier: {premium_info['risk_multiplier']:.2f}x")
    st.write(f"- Age Multiplier: {premium_info['age_multiplier']:.2f}x")

st.markdown("---")
st.caption("Tunisia Climate Risk Assessment System | Built with Streamlit & Python")
