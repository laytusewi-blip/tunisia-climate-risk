"""
Visualization Module
"""
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class MapVisualizer:
    def __init__(self):
        self.map = None

    def create_tunisia_map(self, center=[36.8, 10.2], zoom=7):
        return folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')

    def add_neighborhoods(self, m, gdf, selected_name=None):
        for idx, row in gdf.iterrows():
            style = {'fillColor': '#FFD700', 'color': '#FF0000', 'weight': 3, 'fillOpacity': 0.6} if selected_name and row['name'] == selected_name else {'fillColor': '#3186cc', 'color': '#1E3A8A', 'weight': 1, 'fillOpacity': 0.15}
            popup_html = f"""<div style="font-family: Arial;"><h4>{row['name']}</h4><b>Gov:</b> {row['governorate']}<br><b>Pop:</b> {row.get('population', 'N/A'):,}</div>"""
            folium.GeoJson(row['geometry'], name=row['name'], style_function=lambda x, style=style: style, tooltip=row['name'], popup=folium.Popup(popup_html, max_width=250)).add_to(m)
        return m

    def add_hazard_layers(self, m, gdf, hazard_zones):
        flood_group = folium.FeatureGroup(name='Flood Zones', show=False)
        for idx, row in gdf.iterrows():
            hazard_row = hazard_zones[hazard_zones['neighborhood'] == row['name']]
            if not hazard_row.empty:
                flood_level = hazard_row.iloc[0]['flood_zone']
                color = {'Low': '#00FF00', 'Medium': '#FFA500', 'High': '#FF0000'}.get(flood_level, '#808080')
                folium.GeoJson(row['geometry'], style_function=lambda x, color=color: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.3}, tooltip=f"{row['name']}: {flood_level}").add_to(flood_group)
        flood_group.add_to(m)
        return m

    def add_property_marker(self, m, lat, lon, property_info):
        folium.Marker(location=[lat, lon], icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(m)
        folium.Circle(location=[lat, lon], radius=500, color='#DC2626', fill=True, fillOpacity=0.2).add_to(m)
        return m

    def finalize_map(self, m):
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        return m

class ChartVisualizer:
    @staticmethod
    def create_risk_radar(risk_scores):
        categories = ['Flood', 'Wildfire', 'Hurricane', 'Drought']
        values = [risk_scores.get(k, 0) for k in ['flood', 'wildfire', 'hurricane', 'drought']]
        composite = risk_scores.get('composite', 50)
        color = '#DC2626' if composite >= 70 else '#F59E0B' if composite >= 40 else '#10B981'
        fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=color, fillcolor=color, opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False, title='Risk Profile', height=400)
        return fig

    @staticmethod
    def create_premium_comparison(base, adjusted):
        data = pd.DataFrame({'Model': ['Traditional', 'ML-Enhanced'], 'Premium': [base, adjusted]})
        fig = px.bar(data, x='Model', y='Premium', color='Model', color_discrete_sequence=['#93C5FD', '#DC2626'])
        fig.update_layout(showlegend=False, height=400, title='Premium Comparison')
        return fig

    @staticmethod
    def create_loss_distribution(expected_loss, property_value):
        loss_ratio = (expected_loss / property_value) * 100 if property_value > 0 else 0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=loss_ratio, title={'text': "Loss Ratio %"}, gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "darkred"}}))
        fig.update_layout(height=300)
        return fig

    @staticmethod
    def create_risk_breakdown(risk_scores):
        data = pd.DataFrame({'Hazard': ['Flood', 'Wildfire', 'Hurricane', 'Drought'], 'Score': [risk_scores.get(k, 0) for k in ['flood', 'wildfire', 'hurricane', 'drought']]})
        fig = px.bar(data, x='Hazard', y='Score', color='Hazard', color_discrete_map={'Flood': '#3B82F6', 'Wildfire': '#EF4444', 'Hurricane': '#8B5CF6', 'Drought': '#F59E0B'})
        fig.update_layout(showlegend=False, height=400, yaxis_range=[0, 100])
        return fig

    @staticmethod
    def create_mitigation_analysis(mitigation_df):
        fig = px.scatter(mitigation_df, x='payback_years', y='annual_savings', size='risk_reduction', color='roi_5year', hover_data=['measure'], color_continuous_scale='RdYlGn')
        fig.update_layout(height=400, title='Mitigation ROI')
        return fig
