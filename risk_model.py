"""
Risk Model Module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

class ClimateRiskModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_features(self, data):
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
        np.random.seed(42)
        data = {'property_value': np.random.uniform(100000, 2000000, n_samples), 'building_age': np.random.randint(0, 100, n_samples), 'elevation': np.random.uniform(0, 500, n_samples), 'distance_to_coast': np.random.uniform(0, 150, n_samples), 'latitude': np.random.uniform(33, 37.5, n_samples), 'longitude': np.random.uniform(8, 11.5, n_samples), 'coastal': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]), 'population_density': np.random.uniform(100, 10000, n_samples), 'avg_annual_temp': np.random.uniform(15, 28, n_samples), 'avg_annual_precip': np.random.uniform(200, 800, n_samples), 'extreme_heat_days': np.random.randint(10, 80, n_samples), 'heavy_rainfall_days': np.random.randint(5, 40, n_samples), 'flood_zone': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]), 'wildfire_zone': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]), 'hurricane_exposure': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])}
        X = pd.DataFrame(data)
        y_total = X['property_value'] * 0.0003 * (X['flood_zone'] + X['wildfire_zone'] + X['hurricane_exposure']) * np.random.uniform(0.8, 1.2, n_samples)
        return X, y_total, {}

    def train_model(self, X=None, y=None):
        if X is None or y is None:
            X, y, _ = self.generate_synthetic_training_data(n_samples=1000)
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42) if self.model_type == 'xgboost' else RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = {'mae': mean_absolute_error(y_test, y_pred), 'rmse': np.sqrt(mean_squared_error(y_test, y_pred)), 'r2': r2_score(y_test, y_pred)}
        self.is_trained = True
        return metrics

    def predict_loss(self, features_dict):
        if not self.is_trained:
            self.train_model()
        features = self.prepare_features(features_dict)
        if self.feature_names:
            features = features[self.feature_names]
        return max(0, self.model.predict(features)[0])

    def calculate_risk_scores(self, features_dict):
        elevation = features_dict.get('elevation', 50)
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

        hurricane_risk = min(100, max(0, (70 if coastal else 10) * (1 + (37 - latitude) * 0.05)))
        drought_risk = min(100, 30 + max(0, (36 - latitude) * 8))

        age_factor = min(30, building_age * 0.3)
        flood_risk = min(100, flood_risk + age_factor * 0.3)
        wildfire_risk = min(100, wildfire_risk + age_factor * 0.25)
        hurricane_risk = min(100, hurricane_risk + age_factor * 0.3)

        composite_risk = flood_risk * 0.35 + wildfire_risk * 0.25 + hurricane_risk * 0.25 + drought_risk * 0.15

        return {'flood': float(flood_risk), 'wildfire': float(wildfire_risk), 'hurricane': float(hurricane_risk), 'drought': float(drought_risk), 'composite': float(composite_risk)}

class PremiumCalculator:
    def __init__(self):
        self.base_rate = 0.003

    def calculate_premium(self, property_value, risk_scores, building_age):
        composite_risk = risk_scores.get('composite', 50)
        risk_multiplier = 1 + (composite_risk / 100) * 2
        age_multiplier = 1 + (building_age / 100) * 0.5
        base_premium = property_value * self.base_rate
        adjusted_premium = base_premium * risk_multiplier * age_multiplier
        loss_probability = (composite_risk / 100) * 0.05
        expected_loss = property_value * loss_probability * 0.3
        return {'base_annual': float(base_premium), 'adjusted_annual': float(adjusted_premium), 'monthly': float(adjusted_premium / 12), 'expected_loss': float(expected_loss), 'loss_probability': float(loss_probability), 'risk_multiplier': float(risk_multiplier), 'age_multiplier': float(age_multiplier)}

    def calculate_mitigation_impact(self, current_premium, mitigation_measures):
        results = []
        for measure, details in mitigation_measures.items():
            annual_savings = current_premium * details['savings']
            payback_period = details['cost'] / annual_savings if annual_savings > 0 else float('inf')
            results.append({'measure': measure, 'cost': details['cost'], 'annual_savings': annual_savings, 'risk_reduction': details['reduction'], 'payback_years': payback_period, 'roi_5year': (annual_savings * 5 - details['cost']) / details['cost'] * 100})
        return pd.DataFrame(results)

def get_risk_model(model_type='xgboost'):
    model = ClimateRiskModel(model_type=model_type)
    metrics = model.train_model()
    return model, metrics

def get_premium_calculator():
    return PremiumCalculator()
