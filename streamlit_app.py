"""Tunisia Climate Risk Assessment"""
import streamlit as st
from data_loader import get_data_loader
from risk_model import get_risk_model, get_premium_calculator
from visualization import MapVisualizer, ChartVisualizer
from streamlit_folium import st_folium

st.set_page_config(page_title="Tunisia Climate Risk", page_icon="🌍", layout="wide")

st.markdown("""<style>.main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}.sub-header {font-size: 1.5rem; color: #2563EB; font-weight: 600; margin-top: 1.5rem;}.metric-box {background-color: #F0F9FF; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2563EB;}</style>""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    with st.spinner('Loading...'):
        st.session_state.data_loader = get_data_loader()
        st.session_state.risk_model, st.session_state.metrics = get_risk_model()
        st.session_state.premium_calc = get_premium_calculator()
        st.session_state.map_viz = MapVisualizer()
        st.session_state.chart_viz = ChartVisualizer()
        st.session_state.initialized = True

gdf = st.session_state.data_loader.neighborhoods
hazard_zones = st.session_state.data_loader.hazard_zones

st.markdown('<p class="main-header">🌍 Tunisia Climate Risk Assessment</p>', unsafe_allow_html=True)
st.markdown("**Insurance Underwriting & Catastrophe Modeling**")
st.markdown("---")

with st.sidebar:
    st.header("📍 Select Neighborhood")
    selected_name = st.selectbox("Neighborhood:", sorted(gdf['name'].tolist()))
    st.markdown("---")
    st.header("🏠 Property Details")
    property_value = st.number_input("Property Value (TND)", 50000, 5000000, 350000, 10000)
    building_age = st.slider("Building Age (years)", 0, 100, 10)
    st.markdown("---")
    st.metric("Model R²", f"{st.session_state.metrics['r2']:.3f}")

selected = gdf[gdf['name'] == selected_name].iloc[0]
lat, lon = selected['lat'], selected['lon']
coastal, population, area_km2 = selected['coastal'], selected['population'], selected['area_km2']

elevation = st.session_state.data_loader.get_elevation_data(lat, lon)
distance_to_coast = st.session_state.data_loader.calculate_distance_to_coast(lat, lon, coastal)
climate_data = st.session_state.data_loader.generate_climate_data(selected_name)
hazard = hazard_zones[hazard_zones['neighborhood'] == selected_name].iloc[0]

features = {'property_value': property_value, 'building_age': building_age, 'elevation': elevation, 'distance_to_coast': distance_to_coast, 'latitude': lat, 'longitude': lon, 'coastal': coastal, 'population': population, 'area_km2': area_km2, **climate_data, 'flood_zone': hazard['flood_zone'], 'wildfire_zone': hazard['wildfire_zone'], 'hurricane_exposure': hazard['hurricane_exposure']}

risk_scores = st.session_state.risk_model.calculate_risk_scores(features)
predicted_loss = st.session_state.risk_model.predict_loss(features)
premium = st.session_state.premium_calc.calculate_premium(property_value, risk_scores, building_age)

st.markdown('<p class="sub-header">🗺️ Interactive Map</p>', unsafe_allow_html=True)

col_map, col_info = st.columns([3, 1])
with col_map:
    m = st.session_state.map_viz.create_tunisia_map([lat, lon], 11)
    m = st.session_state.map_viz.add_neighborhoods(m, gdf, selected_name)
    m = st.session_state.map_viz.add_hazard_layers(m, gdf, hazard_zones)
    m = st.session_state.map_viz.add_property_marker(m, lat, lon, {"value": property_value, "age": building_age})
    m = st.session_state.map_viz.finalize_map(m)
    st_folium(m, width=800, height=500)

with col_info:
    st.metric("Neighborhood", selected_name)
    st.metric("Population", f"{population:,}")
    st.metric("Elevation", f"{elevation} m")
    st.markdown(f"🌊 Flood: **{hazard['flood_zone']}**")
    st.markdown(f"🔥 Wildfire: **{hazard['wildfire_zone']}**")

st.markdown("---")
st.markdown('<p class="sub-header">📊 Risk Analysis</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌊 Flood", f"{risk_scores['flood']:.1f}")
col2.metric("🔥 Wildfire", f"{risk_scores['wildfire']:.1f}")
col3.metric("🌪️ Hurricane", f"{risk_scores['hurricane']:.1f}")
col4.metric("☀️ Drought", f"{risk_scores['drought']:.1f}")
composite = risk_scores['composite']
risk_cat = "High" if composite >= 70 else "Medium" if composite >= 40 else "Low"
col5.metric("Overall", f"{composite:.1f}", delta=risk_cat)

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(st.session_state.chart_viz.create_risk_radar(risk_scores), use_container_width=True)
with col_right:
    st.plotly_chart(st.session_state.chart_viz.create_risk_breakdown(risk_scores), use_container_width=True)

st.markdown("---")
st.markdown('<p class="sub-header">💰 Premium Calculation</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Base Premium**")
    st.markdown(f'<p style="font-size: 2rem; color: #1E3A8A; font-weight: bold;">{premium["base_annual"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Risk-Adjusted**")
    st.markdown(f'<p style="font-size: 2rem; color: #DC2626; font-weight: bold;">{premium["adjusted_annual"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown(f"Monthly: **{premium['monthly']:,.0f} TND**")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Expected Loss**")
    st.markdown(f'<p style="font-size: 2rem; color: #DC2626; font-weight: bold;">{premium["expected_loss"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.plotly_chart(st.session_state.chart_viz.create_premium_comparison(premium['base_annual'], premium['adjusted_annual']), use_container_width=True)
with col_c2:
    st.plotly_chart(st.session_state.chart_viz.create_loss_distribution(premium['expected_loss'], property_value), use_container_width=True)

st.markdown("---")
st.markdown('<p class="sub-header">📋 Recommendations</p>', unsafe_allow_html=True)

if risk_scores['flood'] > 60:
    st.warning("**⚠️ High Flood Risk**\n\nRequire flood barriers and drainage systems.")
if risk_scores['wildfire'] > 60:
    st.warning("**⚠️ High Wildfire Risk**\n\nMandate fire-resistant roofing.")
if risk_scores['composite'] < 40:
    st.success("**✅ Low Risk Property**\n\nEligible for preferred rates.")

st.markdown("---")
st.markdown('<p class="sub-header">🛡️ Risk Mitigation</p>', unsafe_allow_html=True)

mitigation_measures = {"Flood barriers": {"cost": 7000, "savings": 0.11, "reduction": 18}, "Fire-resistant roof": {"cost": 9000, "savings": 0.10, "reduction": 15}, "Hurricane shutters": {"cost": 3500, "savings": 0.09, "reduction": 12}}
mitigation_df = st.session_state.premium_calc.calculate_mitigation_impact(premium['adjusted_annual'], mitigation_measures)
st.dataframe(mitigation_df.style.format({'cost': '{:,.0f} TND', 'annual_savings': '{:,.0f} TND', 'payback_years': '{:.1f} years', 'roi_5year': '{:+.1f}%'}), use_container_width=True)
st.plotly_chart(st.session_state.chart_viz.create_mitigation_analysis(mitigation_df), use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6B7280; padding: 2rem;'><p><strong>Tunisia Climate Risk Assessment</strong></p><p>Master's Project | December 2025</p></div>", unsafe_allow_html=True)
