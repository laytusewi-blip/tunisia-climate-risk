# ============================================================================
# COMPLETE TUNISIA CLIMATE RISK ASSESSMENT SYSTEM - FIXED VERSION
# ============================================================================

# FILE 1: data_loader.py
with open('data_loader.py', 'w', encoding='utf-8') as f:
    f.write('''"""
Data Loader Module
Handles loading of geospatial data, climate data, and hazard maps
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

class TunisiaDataLoader:
    """Load and process Tunisia geospatial and climate data"""

    def __init__(self):
        self.neighborhoods = None
        self.climate_data = None
        self.hazard_zones = None

    def create_tunisia_neighborhoods(self):
        """Create synthetic neighborhood boundaries for Tunisia"""
        neighborhoods_data = [
            {"name": "Tunis Centre Ville", "governorate": "Tunis", "lat": 36.8065, "lon": 10.1815,
             "population": 50000, "area_km2": 5.2, "coastal": True},
            {"name": "La Marsa", "governorate": "Tunis", "lat": 36.8781, "lon": 10.3250,
             "population": 92000, "area_km2": 18.5, "coastal": True},
            {"name": "Carthage", "governorate": "Tunis", "lat": 36.8526, "lon": 10.3233,
             "population": 21000, "area_km2": 7.8, "coastal": True},
            {"name": "Bardo", "governorate": "Tunis", "lat": 36.8089, "lon": 10.1403,
             "population": 73300, "area_km2": 12.3, "coastal": False},
            {"name": "Ariana Ville", "governorate": "Ariana", "lat": 36.8625, "lon": 10.1956,
             "population": 97687, "area_km2": 16.2, "coastal": False},
            {"name": "Soukra", "governorate": "Ariana", "lat": 36.8489, "lon": 10.2156,
             "population": 129693, "area_km2": 22.4, "coastal": False},
            {"name": "Sfax Centre", "governorate": "Sfax", "lat": 34.7405, "lon": 10.7603,
             "population": 272801, "area_km2": 56.2, "coastal": True},
            {"name": "Sfax Medina", "governorate": "Sfax", "lat": 34.7431, "lon": 10.7614,
             "population": 45000, "area_km2": 8.1, "coastal": True},
            {"name": "Sousse Medina", "governorate": "Sousse", "lat": 35.8256, "lon": 10.6369,
             "population": 221530, "area_km2": 45.0, "coastal": True},
            {"name": "Port El Kantaoui", "governorate": "Sousse", "lat": 35.8942, "lon": 10.5954,
             "population": 8545, "area_km2": 15.3, "coastal": True},
            {"name": "Hammam Sousse", "governorate": "Sousse", "lat": 35.8603, "lon": 10.6031,
             "population": 38000, "area_km2": 18.7, "coastal": True},
            {"name": "Bizerte Centre", "governorate": "Bizerte", "lat": 37.2744, "lon": 9.8739,
             "population": 142966, "area_km2": 32.1, "coastal": True},
            {"name": "Menzel Bourguiba", "governorate": "Bizerte", "lat": 37.1542, "lon": 9.7853,
             "population": 39384, "area_km2": 11.5, "coastal": False},
            {"name": "Hammamet", "governorate": "Nabeul", "lat": 36.4000, "lon": 10.6167,
             "population": 73236, "area_km2": 36.8, "coastal": True},
            {"name": "Nabeul Centre", "governorate": "Nabeul", "lat": 36.4561, "lon": 10.7356,
             "population": 56387, "area_km2": 28.4, "coastal": True},
            {"name": "Monastir Centre", "governorate": "Monastir", "lat": 35.7774, "lon": 10.8264,
             "population": 93306, "area_km2": 24.5, "coastal": True},
            {"name": "Kairouan Medina", "governorate": "Kairouan", "lat": 35.6781, "lon": 10.0963,
             "population": 186653, "area_km2": 42.0, "coastal": False},
            {"name": "Gabès Centre", "governorate": "Gabès", "lat": 33.8815, "lon": 10.0982,
             "population": 152921, "area_km2": 38.6, "coastal": True},
            {"name": "Djerba Houmt Souk", "governorate": "Medenine", "lat": 33.8076, "lon": 10.8451,
             "population": 75904, "area_km2": 45.2, "coastal": True},
            {"name": "Medenine Centre", "governorate": "Medenine", "lat": 33.3549, "lon": 10.5055,
             "population": 61705, "area_km2": 28.9, "coastal": False},
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
        self.neighborhoods = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        return self.neighborhoods

    def generate_climate_data(self, neighborhood_name=None):
        """Generate synthetic climate data"""
        if neighborhood_name and self.neighborhoods is not None:
            row = self.neighborhoods[self.neighborhoods['name'] == neighborhood_name].iloc[0]
            lat = row['lat']
            coastal = row.get('coastal', False)
        else:
            lat = 36.8
            coastal = False

        base_temp = 25 - (lat - 33) * 1.5
        base_precip = 600 - (lat - 37) * 50

        climate_data = {
            'avg_annual_temp': base_temp,
            'avg_annual_precip': base_precip,
            'max_temp_recorded': base_temp + 15,
            'min_temp_recorded': base_temp - 10,
            'extreme_heat_days': int((36 - lat) * 20),
            'heavy_rainfall_days': int((lat - 33) * 8),
            'drought_frequency': 1.0 if lat < 35 else 0.5,
            'sea_level_rise_projection_2050': 0.3 if coastal else 0.0,
        }
        return climate_data

    def generate_hazard_zones(self):
        """Create hazard zone data"""
        hazard_zones = []

        if self.neighborhoods is not None:
            for idx, row in self.neighborhoods.iterrows():
                lat, lon = row['lat'], row['lon']
                flood_zone = "High" if row.get('coastal', False) and lat > 35 else "Medium" if row.get('coastal', False) else "Low"
                wildfire_zone = "High" if not row.get('coastal', False) and lat > 35.5 else "Medium"

                hazard_zones.append({
                    'neighborhood': row['name'],
                    'flood_zone': flood_zone,
                    'wildfire_zone': wildfire_zone,
                    'earthquake_zone': "Low",
                    'hurricane_exposure': "High" if row.get('coastal', False) else "Low"
                })

        self.hazard_zones = pd.DataFrame(hazard_zones)
        return self.hazard_zones

    def get_elevation_data(self, lat, lon):
        """Get elevation data for a location"""
        base_elevation = 50
        if lat < 35:
            base_elevation = 20
        elif lon < 9.5:
            base_elevation = 200
        return base_elevation + np.random.randint(-20, 50)

    def calculate_distance_to_coast(self, lat, lon, coastal):
        """Calculate approximate distance to coast"""
        if coastal:
            return np.random.uniform(0.5, 5.0)
        else:
            if lon < 9.5:
                return np.random.uniform(50, 100)
            else:
                return np.random.uniform(20, 50)

def get_data_loader():
    """Factory function to create and initialize data loader"""
    loader = TunisiaDataLoader()
    loader.create_tunisia_neighborhoods()
    loader.generate_hazard_zones()
    return loader
''')

print("✅ Created: data_loader.py")

# FILE 2: risk_model.py
with open('risk_model.py', 'w', encoding='utf-8') as f:
    f.write('''"""
Risk Model Module
Implements machine learning models for climate risk prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

class ClimateRiskModel:
    """ML model for predicting climate-related insurance losses"""

    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_features(self, data):
        """Prepare features for ML model"""
        features = pd.DataFrame()
        features['property_value'] = [data.get('property_value', 300000)]
        features['building_age'] = [data.get('building_age', 15)]
        features['elevation'] = [data.get('elevation', 50)]
        features['distance_to_coast'] = [data.get('distance_to_coast', 10)]
        features['latitude'] = [data.get('latitude', 36.8)]
        features['longitude'] = [data.get('longitude', 10.2)]
        features['coastal'] = [1 if data.get('coastal', False) else 0]
        features['population_density'] = [data.get('population', 50000) / data.get('area_km2', 10)]
        features['avg_annual_temp'] = [data.get('avg_annual_temp', 20)]
        features['avg_annual_precip'] = [data.get('avg_annual_precip', 500)]
        features['extreme_heat_days'] = [data.get('extreme_heat_days', 30)]
        features['heavy_rainfall_days'] = [data.get('heavy_rainfall_days', 15)]

        flood_zone_map = {'Low': 0, 'Medium': 1, 'High': 2}
        features['flood_zone'] = [flood_zone_map.get(data.get('flood_zone', 'Medium'), 1)]
        features['wildfire_zone'] = [flood_zone_map.get(data.get('wildfire_zone', 'Medium'), 1)]
        features['hurricane_exposure'] = [1 if data.get('hurricane_exposure', 'Low') == 'High' else 0]

        return features

    def generate_synthetic_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        np.random.seed(42)

        data = {
            'property_value': np.random.uniform(100000, 2000000, n_samples),
            'building_age': np.random.randint(0, 100, n_samples),
            'elevation': np.random.uniform(0, 500, n_samples),
            'distance_to_coast': np.random.uniform(0, 150, n_samples),
            'latitude': np.random.uniform(33, 37.5, n_samples),
            'longitude': np.random.uniform(8, 11.5, n_samples),
            'coastal': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'population_density': np.random.uniform(100, 10000, n_samples),
            'avg_annual_temp': np.random.uniform(15, 28, n_samples),
            'avg_annual_precip': np.random.uniform(200, 800, n_samples),
            'extreme_heat_days': np.random.randint(10, 80, n_samples),
            'heavy_rainfall_days': np.random.randint(5, 40, n_samples),
            'flood_zone': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),
            'wildfire_zone': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            'hurricane_exposure': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        }

        X = pd.DataFrame(data)

        y_flood = X['property_value'] * 0.0001 * X['flood_zone'] * (1 - X['elevation']/500) * (1 + X['coastal'] * 0.5) * np.random.uniform(0.8, 1.2, n_samples)
        y_wildfire = X['property_value'] * 0.00008 * X['wildfire_zone'] * (1 - X['coastal'] * 0.3) * np.random.uniform(0.8, 1.2, n_samples)
        y_hurricane = X['property_value'] * 0.00012 * X['hurricane_exposure'] * X['coastal'] * np.random.uniform(0.8, 1.2, n_samples)
        y_drought = X['property_value'] * 0.00005 * (37.5 - X['latitude']) * np.random.uniform(0.8, 1.2, n_samples)
        y_total = y_flood + y_wildfire + y_hurricane + y_drought

        return X, y_total, {'flood': y_flood, 'wildfire': y_wildfire, 'hurricane': y_hurricane, 'drought': y_drought}

    def train_model(self, X=None, y=None):
        """Train the risk prediction model"""
        if X is None or y is None:
            X, y, _ = self.generate_synthetic_training_data(n_samples=1000)

        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        self.is_trained = True
        return metrics

    def predict_loss(self, features_dict):
        """Predict expected loss for a property"""
        if not self.is_trained:
            self.train_model()

        features = self.prepare_features(features_dict)
        if self.feature_names:
            features = features[self.feature_names]

        predicted_loss = self.model.predict(features)[0]
        return max(0, predicted_loss)

    def calculate_risk_scores(self, features_dict):
        """Calculate individual risk scores for each hazard type"""
        elevation = features_dict.get('elevation', 50)
        distance_to_coast = features_dict.get('distance_to_coast', 10)
        building_age = features_dict.get('building_age', 15)
        coastal = features_dict.get('coastal', False)
        latitude = features_dict.get('latitude', 36.8)
        flood_zone = features_dict.get('flood_zone', 'Medium')
        wildfire_zone = features_dict.get('wildfire_zone', 'Medium')

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

class PremiumCalculator:
    """Calculate insurance premiums based on risk scores"""

    def __init__(self):
        self.base_rate = 0.003

    def calculate_premium(self, property_value, risk_scores, building_age):
        """Calculate insurance premium"""
        composite_risk = risk_scores.get('composite', 50)
        risk_multiplier = 1 + (composite_risk / 100) * 2
        age_multiplier = 1 + (building_age / 100) * 0.5
        base_premium = property_value * self.base_rate
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

    def calculate_mitigation_impact(self, current_premium, mitigation_measures):
        """Calculate impact of risk mitigation measures"""
        results = []
        for measure, details in mitigation_measures.items():
            annual_savings = current_premium * details['savings']
            payback_period = details['cost'] / annual_savings if annual_savings > 0 else float('inf')
            results.append({
                'measure': measure,
                'cost': details['cost'],
                'annual_savings': annual_savings,
                'risk_reduction': details['reduction'],
                'payback_years': payback_period,
                'roi_5year': (annual_savings * 5 - details['cost']) / details['cost'] * 100
            })
        return pd.DataFrame(results)

def get_risk_model(model_type='xgboost'):
    """Factory function to create and train risk model"""
    model = ClimateRiskModel(model_type=model_type)
    metrics = model.train_model()
    return model, metrics

def get_premium_calculator():
    """Factory function to create premium calculator"""
    return PremiumCalculator()
''')

print("✅ Created: risk_model.py")

# FILE 3: visualization.py
with open('visualization.py', 'w', encoding='utf-8') as f:
    f.write('''"""
Visualization Module
Handles all maps, charts, and dashboards
"""

import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class MapVisualizer:
    """Create interactive maps with Folium"""

    def __init__(self):
        self.map = None

    def create_tunisia_map(self, center=[36.8, 10.2], zoom=7):
        """Create base map of Tunisia"""
        return folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')

    def add_neighborhoods(self, m, gdf, selected_name=None):
        """Add neighborhood polygons to map"""
        for idx, row in gdf.iterrows():
            if selected_name and row['name'] == selected_name:
                style = {'fillColor': '#FFD700', 'color': '#FF0000', 'weight': 3, 'fillOpacity': 0.6}
            else:
                style = {'fillColor': '#3186cc', 'color': '#1E3A8A', 'weight': 1, 'fillOpacity': 0.15}

            popup_html = f"""<div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0; color: #1E3A8A;">{row['name']}</h4>
                <hr style="margin: 5px 0;">
                <b>Governorate:</b> {row['governorate']}<br>
                <b>Population:</b> {row.get('population', 'N/A'):,}<br>
                <b>Coastal:</b> {'Yes' if row.get('coastal', False) else 'No'}
            </div>"""

            folium.GeoJson(row['geometry'], name=row['name'], style_function=lambda x, style=style: style,
                          highlight_function=lambda x: {'fillColor': '#FFD700', 'color': '#FF0000', 'weight': 3, 'fillOpacity': 0.7},
                          tooltip=row['name'], popup=folium.Popup(popup_html, max_width=250)).add_to(m)
        return m

    def add_hazard_layers(self, m, gdf, hazard_zones):
        """Add hazard zone overlays"""
        flood_group = folium.FeatureGroup(name='Flood Zones', show=False)
        for idx, row in gdf.iterrows():
            hazard_row = hazard_zones[hazard_zones['neighborhood'] == row['name']]
            if not hazard_row.empty:
                flood_level = hazard_row.iloc[0]['flood_zone']
                color = {'Low': '#00FF00', 'Medium': '#FFA500', 'High': '#FF0000'}.get(flood_level, '#808080')
                folium.GeoJson(row['geometry'], style_function=lambda x, color=color: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.3},
                              tooltip=f"{row['name']}: {flood_level} Flood Risk").add_to(flood_group)
        flood_group.add_to(m)
        return m

    def add_property_marker(self, m, lat, lon, property_info):
        """Add marker for specific property"""
        folium.Marker(location=[lat, lon], icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(m)
        folium.Circle(location=[lat, lon], radius=500, color='#DC2626', fill=True, fillOpacity=0.2).add_to(m)
        return m

    def finalize_map(self, m):
        """Add layer control"""
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        return m

class ChartVisualizer:
    """Create charts with Plotly"""

    @staticmethod
    def create_risk_radar(risk_scores):
        """Create radar chart"""
        categories = ['Flood', 'Wildfire', 'Hurricane', 'Drought']
        values = [risk_scores.get('flood', 0), risk_scores.get('wildfire', 0), risk_scores.get('hurricane', 0), risk_scores.get('drought', 0)]
        composite = risk_scores.get('composite', 50)
        color = '#DC2626' if composite >= 70 else '#F59E0B' if composite >= 40 else '#10B981'
        fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=color, fillcolor=color, opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title='Multi-Hazard Risk Profile', height=400)
        return fig

    @staticmethod
    def create_premium_comparison(base_premium, adjusted_premium):
        """Create premium comparison chart"""
        data = pd.DataFrame({'Model': ['Traditional', 'ML-Enhanced'], 'Premium': [base_premium, adjusted_premium]})
        fig = px.bar(data, x='Model', y='Premium', color='Model', color_discrete_sequence=['#93C5FD', '#DC2626'])
        fig.update_layout(showlegend=False, height=400, title='Premium Comparison')
        return fig

    @staticmethod
    def create_loss_distribution(expected_loss, property_value):
        """Create loss gauge"""
        loss_ratio = (expected_loss / property_value) * 100 if property_value > 0 else 0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=loss_ratio, title={'text': "Loss Ratio %"},
                                     gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "darkred"}}))
        fig.update_layout(height=300)
        return fig

    @staticmethod
    def create_risk_breakdown(risk_scores):
        """Create risk breakdown"""
        data = pd.DataFrame({'Hazard': ['Flood', 'Wildfire', 'Hurricane', 'Drought'],
                           'Score': [risk_scores.get('flood', 0), risk_scores.get('wildfire', 0),
                                   risk_scores.get('hurricane', 0), risk_scores.get('drought', 0)]})
        fig = px.bar(data, x='Hazard', y='Score', color='Hazard',
                    color_discrete_map={'Flood': '#3B82F6', 'Wildfire': '#EF4444', 'Hurricane': '#8B5CF6', 'Drought': '#F59E0B'})
        fig.update_layout(showlegend=False, height=400, yaxis_range=[0, 100])
        return fig

    @staticmethod
    def create_mitigation_analysis(mitigation_df):
        """Create mitigation analysis"""
        fig = px.scatter(mitigation_df, x='payback_years', y='annual_savings', size='risk_reduction',
                        color='roi_5year', hover_data=['measure'], color_continuous_scale='RdYlGn')
        fig.update_layout(height=400, title='Mitigation ROI Analysis')
        return fig
''')

print("✅ Created: visualization.py")

# FILE 4: app.py (FIXED VERSION)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write('''"""Tunisia Climate Risk Assessment System"""

import streamlit as st
from data_loader import get_data_loader
from risk_model import get_risk_model, get_premium_calculator
from visualization import MapVisualizer, ChartVisualizer
from streamlit_folium import st_folium

st.set_page_config(page_title="Tunisia Climate Risk", page_icon="🌍", layout="wide")

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

st.title("🌍 Tunisia Climate Risk Assessment System")
st.markdown("**Insurance Underwriting & Catastrophe Modeling**")
st.markdown("---")

with st.sidebar:
    st.header("📍 Select Neighborhood")
    selected_name = st.selectbox("Neighborhood:", sorted(gdf['name'].tolist()))
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

features = {'property_value': property_value, 'building_age': building_age, 'elevation': elevation,
           'distance_to_coast': distance_to_coast, 'latitude': lat, 'longitude': lon, 'coastal': coastal,
           'population': population, 'area_km2': area_km2, **climate_data,
           'flood_zone': hazard['flood_zone'], 'wildfire_zone': hazard['wildfire_zone'],
           'hurricane_exposure': hazard['hurricane_exposure']}

risk_scores = st.session_state.risk_model.calculate_risk_scores(features)
predicted_loss = st.session_state.risk_model.predict_loss(features)
premium = st.session_state.premium_calc.calculate_premium(property_value, risk_scores, building_age)

st.header("🗺️ Interactive Map")
col_map, col_info = st.columns([3, 1])

with col_map:
    m = st.session_state.map_viz.create_tunisia_map([lat, lon], 11)
    m = st.session_state.map_viz.add_neighborhoods(m, gdf, selected_name)
    m = st.session_state.map_viz.add_hazard_layers(m, gdf, hazard_zones)
    m = st.session_state.map_viz.add_property_marker(m, lat, lon, {"value": property_value, "age": building_age, "elevation": elevation, "risk_score": risk_scores['composite']})
    m = st.session_state.map_viz.finalize_map(m)
    st_folium(m, width=800, height=500)

with col_info:
    st.metric("Neighborhood", selected_name)
    st.metric("Governorate", selected['governorate'])
    st.metric("Population", f"{population:,}")
    st.metric("Elevation", f"{elevation} m")

st.markdown("---")
st.header("📊 Risk Analysis")

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
st.header("💰 Premium Calculation")

col1, col2, col3 = st.columns(3)
col1.metric("Base Premium", f"{premium['base_annual']:,.0f} TND", help="Traditional Model")
col2.metric("Risk-Adjusted", f"{premium['adjusted_annual']:,.0f} TND", help="ML Model")
col3.metric("Expected Loss", f"{premium['expected_loss']:,.0f} TND", help=f"Probability: {premium['loss_probability']*100:.2f}%")

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    st.plotly_chart(st.session_state.chart_viz.create_premium_comparison(premium['base_annual'], premium['adjusted_annual']), use_container_width=True)
with col_chart2:
    st.plotly_chart(st.session_state.chart_viz.create_loss_distribution(premium['expected_loss'], property_value), use_container_width=True)

st.markdown("---")
st.header("📋 Recommendations")

if risk_scores['flood'] > 60:
    st.warning("High Flood Risk: Require flood barriers and drainage systems.")
if risk_scores['wildfire'] > 60:
    st.warning("High Wildfire Risk: Mandate fire-resistant materials.")
if risk_scores['composite'] < 40:
    st.success("Low Risk Property: Eligible for preferred rates.")

st.header("🛡️ Risk Mitigation")
mitigation_measures = {"Flood barriers": {"cost": 7000, "savings": 0.11, "reduction": 18},
                       "Fire-resistant roof": {"cost": 9000, "savings": 0.10, "reduction": 15},
                       "Hurricane shutters": {"cost": 3500, "savings": 0.09, "reduction": 12}}
mitigation_df = st.session_state.premium_calc.calculate_mitigation_impact(premium['adjusted_annual'], mitigation_measures)
st.dataframe(mitigation_df, use_container_width=True)
st.plotly_chart(st.session_state.chart_viz.create_mitigation_analysis(mitigation_df), use_container_width=True)

st.markdown("---")
st.markdown("**Powered by:** GeoPandas, Folium, Plotly, XGBoost, scikit-learn, Streamlit | December 2025")
''')
