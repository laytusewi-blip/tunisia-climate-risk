"""
Data Loader Module
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

class TunisiaDataLoader:
    def __init__(self):
        self.neighborhoods = None
        self.hazard_zones = None

    def create_tunisia_neighborhoods(self):
        neighborhoods_data = [
          {"name": "Tunis Centre Ville", "governorate": "Tunis", "lat": 36.8065, "lon": 10.1815, "population": 50000, "area_km2": 5.2, "coastal": True},
            {"name": "La Marsa", "governorate": "Tunis", "lat": 36.8781, "lon": 10.3250, "population": 92000, "area_km2": 18.5, "coastal": True},
            {"name": "Carthage", "governorate": "Tunis", "lat": 36.8526, "lon": 10.3233, "population": 21000, "area_km2": 7.8, "coastal": True},
            {"name": "Bardo", "governorate": "Tunis", "lat": 36.8089, "lon": 10.1403, "population": 73300, "area_km2": 12.3, "coastal": False},
            {"name": "Ariana Ville", "governorate": "Ariana", "lat": 36.8625, "lon": 10.1956, "population": 97687, "area_km2": 16.2, "coastal": False},
            {"name": "Soukra", "governorate": "Ariana", "lat": 36.8489, "lon": 10.2156, "population": 129693, "area_km2": 22.4, "coastal": False},
            {"name": "Sfax Centre", "governorate": "Sfax", "lat": 34.7405, "lon": 10.7603, "population": 272801, "area_km2": 56.2, "coastal": True},
            {"name": "Sfax Medina", "governorate": "Sfax", "lat": 34.7431, "lon": 10.7614, "population": 45000, "area_km2": 8.1, "coastal": True},
            {"name": "Sousse Medina", "governorate": "Sousse", "lat": 35.8256, "lon": 10.6369, "population": 221530, "area_km2": 45.0, "coastal": True},
            {"name": "Port El Kantaoui", "governorate": "Sousse", "lat": 35.8942, "lon": 10.5954, "population": 8545, "area_km2": 15.3, "coastal": True},
            {"name": "Hammam Sousse", "governorate": "Sousse", "lat": 35.8603, "lon": 10.6031, "population": 38000, "area_km2": 18.7, "coastal": True},
            {"name": "Bizerte Centre", "governorate": "Bizerte", "lat": 37.2744, "lon": 9.8739, "population": 142966, "area_km2": 32.1, "coastal": True},
            {"name": "Menzel Bourguiba", "governorate": "Bizerte", "lat": 37.1542, "lon": 9.7853, "population": 39384, "area_km2": 11.5, "coastal": False},
            {"name": "Hammamet", "governorate": "Nabeul", "lat": 36.4000, "lon": 10.6167, "population": 73236, "area_km2": 36.8, "coastal": True},
            {"name": "Nabeul Centre", "governorate": "Nabeul", "lat": 36.4561, "lon": 10.7356, "population": 56387, "area_km2": 28.4, "coastal": True},
            {"name": "Monastir Centre", "governorate": "Monastir", "lat": 35.7774, "lon": 10.8264, "population": 93306, "area_km2": 24.5, "coastal": True},
            {"name": "Kairouan Medina", "governorate": "Kairouan", "lat": 35.6781, "lon": 10.0963, "population": 186653, "area_km2": 42.0, "coastal": False},
            {"name": "Gabès Centre", "governorate": "Gabès", "lat": 33.8815, "lon": 10.0982, "population": 152921, "area_km2": 38.6, "coastal": True},
            {"name": "Djerba Houmt Souk", "governorate": "Medenine", "lat": 33.8076, "lon": 10.8451, "population": 75904, "area_km2": 45.2, "coastal": True},
            {"name": "Medenine Centre", "governorate": "Medenine", "lat": 33.3549, "lon": 10.5055, "population": 61705, "area_km2": 28.9, "coastal": False},
        ]
        geometries = [Polygon([(n['lon']-0.02, n['lat']-0.02), (n['lon']+0.02, n['lat']-0.02), (n['lon']+0.02, n['lat']+0.02), (n['lon']-0.02, n['lat']+0.02)]) for n in neighborhoods_data]
        self.neighborhoods = gpd.GeoDataFrame(pd.DataFrame(neighborhoods_data), geometry=geometries, crs="EPSG:4326")
        return self.neighborhoods

    def generate_climate_data(self, neighborhood_name=None):
        if neighborhood_name and self.neighborhoods is not None:
            row = self.neighborhoods[self.neighborhoods['name'] == neighborhood_name].iloc[0]
            lat, coastal = row['lat'], row.get('coastal', False)
        else:
            lat, coastal = 36.8, False
        base_temp = 25 - (lat - 33) * 1.5
        base_precip = 600 - (lat - 37) * 50
        return {'avg_annual_temp': base_temp, 'avg_annual_precip': base_precip, 'max_temp_recorded': base_temp + 15, 'min_temp_recorded': base_temp - 10, 'extreme_heat_days': int((36 - lat) * 20), 'heavy_rainfall_days': int((lat - 33) * 8), 'drought_frequency': 1.0 if lat < 35 else 0.5, 'sea_level_rise_projection_2050': 0.3 if coastal else 0.0}

    def generate_hazard_zones(self):
        hazard_zones = []
        if self.neighborhoods is not None:
            for idx, row in self.neighborhoods.iterrows():
                lat, coastal = row['lat'], row.get('coastal', False)
                flood_zone = "High" if coastal and lat > 35 else "Medium" if coastal else "Low"
                wildfire_zone = "High" if not coastal and lat > 35.5 else "Medium"
                hazard_zones.append({'neighborhood': row['name'], 'flood_zone': flood_zone, 'wildfire_zone': wildfire_zone, 'earthquake_zone': "Low", 'hurricane_exposure': "High" if coastal else "Low"})
        self.hazard_zones = pd.DataFrame(hazard_zones)
        return self.hazard_zones

    def get_elevation_data(self, lat, lon):
        return (50 if lat >= 35 else 20) + np.random.randint(-20, 50)

    def calculate_distance_to_coast(self, lat, lon, coastal):
        return np.random.uniform(0.5, 5.0) if coastal else np.random.uniform(20, 100)

def get_data_loader():
    loader = TunisiaDataLoader()
    loader.create_tunisia_neighborhoods()
    loader.generate_hazard_zones()
    return loader
