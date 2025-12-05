"""Tunisia Climate Risk Assessment System - Enhanced"""
import streamlit as st
from data_loader import get_data_loader
from risk_model import get_risk_model, get_premium_calculator
from visualization import MapVisualizer, ChartVisualizer
from streamlit_folium import st_folium

st.set_page_config(page_title="Tunisia Climate Risk", page_icon="🌍", layout="wide")

st.markdown("""<style>.main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}.sub-header {font-size: 1.5rem; color: #2563EB; font-weight: 600; margin-top: 1.5rem;}.metric-box {background-color: #F0F9FF; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2563EB;}</style>""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    with st.spinner('🔄 Loading data and training ML models...'):
        st.session_state.data_loader = get_data_loader()
        st.session_state.risk_model, st.session_state.metrics = get_risk_model()
        st.session_state.premium_calc = get_premium_calculator()
        st.session_state.map_viz = MapVisualizer()
        st.session_state.chart_viz = ChartVisualizer()
        st.session_state.initialized = True

gdf = st.session_state.data_loader.neighborhoods
hazard_zones = st.session_state.data_loader.hazard_zones

st.markdown('<p class="main-header">🌍 Tunisia Climate Risk Assessment</p>', unsafe_allow_html=True)
st.markdown("**ML-Powered Insurance Underwriting & Catastrophe Modeling**")
st.markdown("---")

with st.sidebar:
    st.header("📍 Select Neighborhood")
    selected_name = st.selectbox("Neighborhood:", sorted(gdf['name'].tolist()))
    st.markdown("---")
    st.header("🏠 Property Details")
    property_value = st.number_input("Property Value (TND)", 50000, 5000000, 350000, 10000, help="Total insurable value")
    building_age = st.slider("Building Age (years)", 0, 100, 10, help="Age affects vulnerability")
    st.markdown("---")
    st.header("📊 Model Performance")
    st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
    st.metric("MAE", f"{st.session_state.metrics['mae']:,.0f} TND")
    st.metric("RMSE", f"{st.session_state.metrics['rmse']:,.0f} TND")
    
    # Performance interpretation
    if st.session_state.metrics['r2'] > 0.8:
        st.success("✅ Excellent accuracy")
    elif st.session_state.metrics['r2'] > 0.6:
        st.info("ℹ️ Good accuracy")

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

# MAP SECTION
st.markdown('<p class="sub-header">🗺️ Interactive Risk Map</p>', unsafe_allow_html=True)

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
    st.metric("Governorate", selected['governorate'])
    st.metric("Population", f"{population:,}")
    st.metric("Elevation", f"{elevation} m")
    st.metric("Coast Distance", f"{distance_to_coast:.1f} km")
    st.markdown("---")
    st.markdown("**Hazard Zones**")
    flood_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.markdown(f"🌊 Flood: {flood_emoji.get(hazard['flood_zone'], '⚪')} **{hazard['flood_zone']}**")
    st.markdown(f"🔥 Wildfire: {flood_emoji.get(hazard['wildfire_zone'], '⚪')} **{hazard['wildfire_zone']}**")
    st.markdown(f"🌪️ Hurricane: {flood_emoji.get(hazard['hurricane_exposure'], '⚪')} **{hazard['hurricane_exposure']}**")

st.markdown("---")

# RISK ANALYSIS
st.markdown('<p class="sub-header">📊 Climate Risk Analysis</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌊 Flood", f"{risk_scores['flood']:.1f}/100")
col2.metric("🔥 Wildfire", f"{risk_scores['wildfire']:.1f}/100")
col3.metric("🌪️ Hurricane", f"{risk_scores['hurricane']:.1f}/100")
col4.metric("☀️ Drought", f"{risk_scores['drought']:.1f}/100")
composite = risk_scores['composite']
risk_cat = "High" if composite >= 70 else "Medium" if composite >= 40 else "Low"
risk_emoji = "🔴" if composite >= 70 else "🟡" if composite >= 40 else "🟢"
col5.metric(f"{risk_emoji} Overall", f"{composite:.1f}/100", delta=f"{risk_cat} Risk")

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(st.session_state.chart_viz.create_risk_radar(risk_scores), use_container_width=True)
with col_right:
    st.plotly_chart(st.session_state.chart_viz.create_risk_breakdown(risk_scores), use_container_width=True)

# CLIMATE DATA - NEW!
st.markdown("---")
st.markdown("### 🌡️ Climate Characteristics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Annual Temp", f"{climate_data['avg_annual_temp']:.1f}°C")
with col2:
    st.metric("Annual Precipitation", f"{climate_data['avg_annual_precip']:.0f} mm")
with col3:
    st.metric("Extreme Heat Days", f"{climate_data['extreme_heat_days']} days/yr")
with col4:
    st.metric("Heavy Rainfall Days", f"{climate_data['heavy_rainfall_days']} days/yr")

st.markdown("---")

# PREMIUM CALCULATION
st.markdown('<p class="sub-header">💰 Insurance Premium Calculation</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Base Annual Premium**")
    st.markdown(f'<p style="font-size: 2rem; color: #1E3A8A; font-weight: bold;">{premium["base_annual"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown("(Traditional Actuarial Model)")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Risk-Adjusted Premium**")
    color = "#DC2626" if composite >= 70 else "#F59E0B" if composite >= 40 else "#10B981"
    st.markdown(f'<p style="font-size: 2rem; color: {color}; font-weight: bold;">{premium["adjusted_annual"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown("(ML-Enhanced Climate Risk Model)")
    st.markdown(f"Monthly: **{premium['monthly']:,.0f} TND**")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**Expected Annual Loss**")
    st.markdown(f'<p style="font-size: 2rem; color: #DC2626; font-weight: bold;">{premium["expected_loss"]:,.0f} TND</p>', unsafe_allow_html=True)
    st.markdown(f"Loss Probability: **{premium['loss_probability']*100:.2f}%**")
    st.markdown(f"ML Predicted Loss: **{predicted_loss:,.0f} TND**")
    st.markdown('</div>', unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.plotly_chart(st.session_state.chart_viz.create_premium_comparison(premium['base_annual'], premium['adjusted_annual']), use_container_width=True)
with col_c2:
    st.plotly_chart(st.session_state.chart_viz.create_loss_distribution(premium['expected_loss'], property_value), use_container_width=True)

# PREMIUM FACTORS - NEW!
st.markdown("---")
st.markdown("### 📈 Premium Calculation Breakdown")
factors_df = pd.DataFrame({
    'Factor': ['Base Rate', 'Risk Multiplier', 'Age Multiplier', 'Final Multiplier'],
    'Value': ['0.30%', f"{premium['risk_multiplier']:.2f}x", f"{premium['age_multiplier']:.2f}x", f"{premium['risk_multiplier'] * premium['age_multiplier']:.2f}x"],
    'Impact': [f"{property_value * 0.003:,.0f} TND", f"+{(premium['risk_multiplier'] - 1) * 100:.0f}%", f"+{(premium['age_multiplier'] - 1) * 100:.0f}%", f"{premium['adjusted_annual']:,.0f} TND"]
})
st.dataframe(factors_df, use_container_width=True, hide_index=True)

st.markdown("---")

# RECOMMENDATIONS - ENHANCED!
st.markdown('<p class="sub-header">📋 Underwriting Recommendations</p>', unsafe_allow_html=True)

if risk_scores['flood'] > 60:
    st.warning("**⚠️ High Flood Risk**\n\n**Required Actions:**\n- Install flood barriers and improved drainage systems\n- Elevate critical utilities above projected flood levels\n- Exclude basement coverage or require waterproofing\n- Mandate annual flood preparedness inspections")

if risk_scores['wildfire'] > 60:
    st.warning("**⚠️ High Wildfire Risk**\n\n**Required Actions:**\n- Mandate fire-resistant roofing (Class A rated)\n- Require defensible space (30m vegetation clearance)\n- Install ember-resistant vents and screens\n- Conduct annual fire safety inspections")

if risk_scores['hurricane'] > 60:
    st.warning("**⚠️ High Hurricane Risk**\n\n**Required Actions:**\n- Install hurricane-resistant windows and doors\n- Reinforce roof-to-wall connections\n- Require impact-resistant shutters\n- Install backup power systems")

if building_age > 30:
    st.info("**ℹ️ Aging Building Assessment**\n\n**Recommended Actions:**\n- Conduct structural integrity inspection within 30 days\n- Update electrical and plumbing to current codes\n- Consider seismic retrofitting assessment")

if risk_scores['composite'] < 40:
    st.success("**✅ Low Risk Property - Preferred Rating**\n\n**Available Benefits:**\n- Eligible for preferred insurance rates\n- Extended coverage options available\n- Multi-policy discount opportunities\n- Lower deductible options")

if elevation < 20 and distance_to_coast < 5:
    st.warning("**⚠️ Sea Level Rise Exposure**\n\n**Long-term Considerations:**\n- Property exposed to sea level rise (+30cm by 2050)\n- Incorporate 50-year climate projections\n- Consider elevation requirements or flood-proofing")

st.markdown("---")

# RISK MITIGATION
st.markdown('<p class="sub-header">🛡️ Risk Mitigation Opportunities</p>', unsafe_allow_html=True)

mitigation_measures = {
    "Install flood barriers": {"cost": 7000, "savings": 0.11, "reduction": 18},
    "Fire-resistant roofing": {"cost": 9000, "savings": 0.10, "reduction": 15},
    "Hurricane shutters": {"cost": 3500, "savings": 0.09, "reduction": 12},
    "Drainage system upgrade": {"cost": 5500, "savings": 0.08, "reduction": 10},
    "Vegetation clearance": {"cost": 1500, "savings": 0.05, "reduction": 7},
    "Seismic retrofitting": {"cost": 12000, "savings": 0.07, "reduction": 9}
}

mitigation_df = st.session_state.premium_calc.calculate_mitigation_impact(premium['adjusted_annual'], mitigation_measures)
st.dataframe(mitigation_df.style.format({'cost': '{:,.0f} TND', 'annual_savings': '{:,.0f} TND', 'payback_years': '{:.1f} years', 'roi_5year': '{:+.1f}%'}), use_container_width=True)
st.plotly_chart(st.session_state.chart_viz.create_mitigation_analysis(mitigation_df), use_container_width=True)

# TOP MITIGATION RECOMMENDATIONS - NEW!
st.markdown("### 💡 Top Priority Actions")
top_3 = mitigation_df.nlargest(3, 'roi_5year')
for idx, row in top_3.iterrows():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**{row['measure']}**")
    with col2:
        st.metric("5-Year ROI", f"{row['roi_5year']:+.0f}%")
    with col3:
        st.metric("Payback", f"{row['payback_years']:.1f} yrs")

st.markdown("---")

# FOOTER - ENHANCED!
st.markdown(f"""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem; padding: 2rem 0;'>
        <p><strong>Tunisia Climate Risk Assessment System for Insurance Underwriting</strong></p>
        <p>Master's Project | Financial Markets Microstructure | December 2025</p>
        <p><strong>Technology Stack:</strong> Python, GeoPandas, Folium, Plotly, XGBoost, scikit-learn, Streamlit</p>
        <p><strong>Model Performance:</strong> R² = {st.session_state.metrics['r2']:.4f} | MAE = {st.session_state.metrics['mae']:,.0f} TND | RMSE = {st.session_state.metrics['rmse']:,.0f} TND</p>
    </div>
""", unsafe_allow_html=True)
