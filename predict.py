import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from src.architecture import GrowthRegressor

class ModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._estimator_type = "regressor"

    def fit(self, X, y=None): return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Returns [Batch, 2]
            return self.model(X_t).cpu().numpy()

def engineer_features(df):
    """
    Centralized feature engineering to ensure consistency.
    """
    df = df.copy()
    
    # Cyclical Month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    def butterworth_window(x, center, half_width, order=4):
        relative_dist = (x - center) / half_width
        return 1 / (1 + np.power(relative_dist, 2 * order))
        
    def sigmoid_window(x, low, high, steepness=0.8):
        def sig(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
        return sig(steepness * (x - low) * sig(steepness* (high - x)))    

        # Apply the same engineering you did in predict_growth
    df['porcini_temp_bw'] = butterworth_window(df['air_temp_day_avg'], 18, 4, order=2)
    df['chanterelle_temp_bw'] = butterworth_window(df['air_temp_day_avg'], 16.5, 4.0, order=2)

    #df['porcini_night_sig'] = sigmoid_window(df['air_temp_night_avg'], low=6.0, high=20.0, steepness=0.8)
    #df['chanterelle_night_sig'] = sigmoid_window(df['air_temp_night_avg'], low=9.5, high=22.0, steepness=0.7)

    df["temp_drop"] = df['air_temp_day_avg'] - df["air_temp_night_avg"]
    df['moisture_stability'] = df['rain_days_7d'] / (df['rainfall_7d_total'] + 1)

    df["moisture_trend"] = df["rainfall_7d_total"] / df["soil_moisture_avg"]

    df['wind_speed_avg'] = np.log1p(df['wind_speed_avg'])

    # Model the "Trigger Rain Event" (15-30mm)
    # High steepness makes it a clear "Switch"
    #df['porcini_rain_suitability'] = sigmoid_window(df['rainfall_7d_total'], low=15, high=30, steepness=0.9)

    # Model the "Humidity Floor" (75%)
    #df['humidity_suitability'] = sigmoid_window(df['air_humidity_avg'], low=75, high=100, steepness=0.5)

    expected_cols = [
        "air_temp_day_avg", 
        'air_temp_night_avg', 
        'soil_temp_avg', 
        'soil_moisture_avg', 'air_humidity_avg', 'wind_speed_avg',
        'rainfall_3d_total', 'rainfall_7d_total', 'rain_days_7d',
        'max_daily_rain_7d', 'month_sin', 'month_cos', "porcini_temp_bw", "chanterelle_temp_bw",
        # "temp_drop","moisture_stability"
    ]
    return df[expected_cols]

def predict_growth(weather_data, model, preprocessor, device):
    df_input = pd.DataFrame([weather_data])

    df_engineered = engineer_features(df_input)

    X_processed = preprocessor.transform(df_engineered)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor)
        # Scaled back to 0-100
        final_scores = prediction.cpu().numpy()[0] * 100

    return {
        "Porcini Score": round(float(final_scores[0]), 2),
        "Chanterelle Score": round(float(final_scores[1]), 2)
    }

def get_feature_importance(test_scenarios, model, preprocessor, device):
    # 1. Prepare Data
    df_raw = pd.DataFrame([case['data'] for case in test_scenarios])
    df_val = engineer_features(df_raw)
    
    X_val = preprocessor.transform(df_val)
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    # 2. Wrapped Model for Scikit-Learn
    wrapped_model = ModelWrapper(model, device)
    
    # Use the model's own predictions as 'y' to see what features it relies on most
    y_pseudo = wrapped_model.predict(X_val)

    results = permutation_importance(
        wrapped_model, X_val, y_pseudo, 
        n_repeats=10, random_state=42
    )

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': results.importances_mean  
    }).sort_values(by='importance', ascending=False)

    return importance_df

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # Load assets
    preprocessor = joblib.load('utils/preprocessor.joblib')
    model = GrowthRegressor(input_size=14, num_classes=2)
    checkpoint = torch.load("models/growth_model_best_v5.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    print(model)

    test_scenarios = [
    {
        "name": "💎 SCENARIO: Perfect Porcini Flush", 
        "msg": "Autumn cooling after rain. Classic 'thermal shock' trigger.", 
        "data": {
            "month": 9,
            "air_temp_day_avg": 17.0,
            "air_temp_night_avg": 9.0,
            "soil_temp_avg": 13.5,
            "soil_moisture_avg": 38.0,
            "air_humidity_avg": 85.0,
            "wind_speed_avg": 4.0,
            "rainfall_3d_total": 22.0,
            "rainfall_7d_total": 45.0,
            "rain_days_7d": 4,
            "max_daily_rain_7d": 18.0
        }
    },
    {
    "name": "💎 SCENARIO: Ultimate Peak Flush", 
    "msg": "Optimal saturation and thermal stability.", 
    "data": {
        "month": 10,
        "air_temp_day_avg": 18.0,
        "air_temp_night_avg": 11.0,
        "soil_temp_avg": 14.5,
        "soil_moisture_avg": 75.0, 
        "air_humidity_avg": 90.0,
        "wind_speed_avg": 1.0,
        "rainfall_3d_total": 20.0,
        "rainfall_7d_total": 40.0,
        "rain_days_7d": 5,
        "max_daily_rain_7d": 15.0
    }
},
{
    "name": "❌ SCENARIO: Porcini Heat-Stroke", 
    "msg": "High moisture and rain, but lethal heat and drying winds.", 
    "data": {
        "month": 7,
        "air_temp_day_avg": 31.5,
        "air_temp_night_avg": 20.0,
        "soil_temp_avg": 24.0,
        "soil_moisture_avg": 72.0,
        "air_humidity_avg": 45.0,
        "wind_speed_avg": 6.5,
        "rainfall_3d_total": 25.0,
        "rainfall_7d_total": 45.0,
        "rain_days_7d": 4,
        "max_daily_rain_7d": 15.0
    }
},
{
    "name": "💎 SCENARIO: Porcini Peak Flush", 
    "msg": "Perfect thermal shock, high moisture, and zero wind stress.", 
    "data": {
        "month": 10,
        "air_temp_day_avg": 16.5,
        "air_temp_night_avg": 9.5,
        "soil_temp_avg": 14.0,
        "soil_moisture_avg": 82.0,
        "air_humidity_avg": 92.0,
        "wind_speed_avg": 0.5,
        "rainfall_3d_total": 18.0,
        "rainfall_7d_total": 38.0,
        "rain_days_7d": 5,
        "max_daily_rain_7d": 12.0
    }
},
    {
        "name": "💎 SCENARIO: Midsummer Chanterelle Peak", 
        "msg": "High humidity, warm nights, and consistent light rain.", 
        "data": {
            "month": 7,
            "air_temp_day_avg": 23.5,
            "air_temp_night_avg": 16.0,
            "soil_temp_avg": 19.5,
            "soil_moisture_avg": 42.0,
            "air_humidity_avg": 88.0,
            "wind_speed_avg": 6.5,
            "rainfall_3d_total": 12.0,
            "rainfall_7d_total": 30.0,
            "rain_days_7d": 5,
            "max_daily_rain_7d": 12.0
        }
    },
    {
        "name": "💎 SCENARIO: The Drying Wind (False Hope)", 
        "msg": "Recent rain looks good, but high winds are drying the forest floor.", 
        "data": {
            "month": 9,
            "air_temp_day_avg": 20.0,
            "air_temp_night_avg": 12.0,
            "soil_temp_avg": 15.0,
            "soil_moisture_avg": 20.0,
            "air_humidity_avg": 45.0,
            "wind_speed_avg": 28.0,
            "rainfall_3d_total": 15.0,
            "rainfall_7d_total": 20.0,
            "rain_days_7d": 2,
            "max_daily_rain_7d": 15.0
        }
    },
    {
        "name": "💎 SCENARIO: Extreme Summer Drought", 
        "msg": "Total dormancy. Soil moisture is too low for mycelial activity.", 
        "data": {
            "month": 8,
            "air_temp_day_avg": 32.0,
            "air_temp_night_avg": 20.0,
            "soil_temp_avg": 25.0,
            "soil_moisture_avg": 5.0,
            "air_humidity_avg": 22.0,
            "wind_speed_avg": 10.0,
            "rainfall_3d_total": 0.0,
            "rainfall_7d_total": 0.0,
            "rain_days_7d": 0,
            "max_daily_rain_7d": 0.0
        }
    },
    {
        "name": "💎 SCENARIO: Early Spring (Too Cold)", 
        "msg": "Plenty of rain, but soil temperature is below the fruiting threshold.", 
        "data": {
            "month": 4,
            "air_temp_day_avg": 10.0,
            "air_temp_night_avg": 2.0,
            "soil_temp_avg": 6.0,
            "soil_moisture_avg": 60.0,
            "air_humidity_avg": 75.0,
            "wind_speed_avg": 15.0,
            "rainfall_3d_total": 30.0,
            "rainfall_7d_total": 55.0,
            "rain_days_7d": 5,
            "max_daily_rain_7d": 25.0
        }
    },
    {
        "name": "💎 SCENARIO: Late Season Frost", 
        "msg": "Season is ending; night frosts kill off existing fruit bodies.", 
        "data": {
            "month": 11,
            "air_temp_day_avg": 6.0,
            "air_temp_night_avg": -3.0,
            "soil_temp_avg": 4.5,
            "soil_moisture_avg": 50.0,
            "air_humidity_avg": 80.0,
            "wind_speed_avg": 12.0,
            "rainfall_3d_total": 5.0,
            "rainfall_7d_total": 15.0,
            "rain_days_7d": 2,
            "max_daily_rain_7d": 10.0
        }
    },
    {
        "name": "💎 SCENARIO: Saturated/Flooded Soil", 
        "msg": "Too much rain in a short window can drown mycelium/limit oxygen.", 
        "data": {
            "month": 10,
            "air_temp_day_avg": 14.0,
            "air_temp_night_avg": 8.0,
            "soil_temp_avg": 11.0,
            "soil_moisture_avg": 95.0,
            "air_humidity_avg": 98.0,
            "wind_speed_avg": 5.0,
            "rainfall_3d_total": 85.0,
            "rainfall_7d_total": 140.0,
            "rain_days_7d": 6,
            "max_daily_rain_7d": 60.0
        }
    },
    {
        "name": "💎 SCENARIO: Recovery Phase", 
        "msg": "The ground is starting to dry after a big rain. Prime for growth.", 
        "data": {
            "month": 9,
            "air_temp_day_avg": 19.0,
            "air_temp_night_avg": 11.0,
            "soil_temp_avg": 15.0,
            "soil_moisture_avg": 30.0,
            "air_humidity_avg": 65.0,
            "wind_speed_avg": 8.0,
            "rainfall_3d_total": 0.0,
            "rainfall_7d_total": 35.0,
            "rain_days_7d": 3,
            "max_daily_rain_7d": 20.0
        }
    },
    {
    "name": "💎 SCENARIO: The Ghost Flush",
    "msg": "Perfect conditions today, but the last 7 days were bone dry. Testing trigger lag.",
    "data": {
        "month": 9,
        "air_temp_day_avg": 18.0,
        "air_temp_night_avg": 10.0,
        "soil_temp_avg": 14.0,
        "soil_moisture_avg": 45.0,
        "air_humidity_avg": 85.0,
        "wind_speed_avg": 2.0,
        "rainfall_3d_total": 25.0, 
        "rainfall_7d_total": 25.0,
        "rain_days_7d": 1, 
        "max_daily_rain_7d": 25.0
    }
},
{
    "name": "💎 SCENARIO: The Golden Week",
    "msg": "Consistent drizzle and warmth. Ideal for slow-growing Chanterelles.",
    "data": {
        "month": 8,
        "air_temp_day_avg": 22.0,
        "air_temp_night_avg": 15.0,
        "soil_temp_avg": 18.5,
        "soil_moisture_avg": 55.0,
        "air_humidity_avg": 80.0,
        "wind_speed_avg": 5.0,
        "rainfall_3d_total": 12.0,
        "rainfall_7d_total": 35.0,
        "rain_days_7d": 6,
        "max_daily_rain_7d": 8.0
    }
},
{
    "name": "💎 SCENARIO: DEAD ZONE",
    "msg": "DEAD ZONE",
    "data": {
        "month": 1,
        "air_temp_day_avg": 0,
        "air_temp_night_avg": -10,
        "soil_temp_avg": 5,
        "soil_moisture_avg": 20,
        "air_humidity_avg": 50,
        "wind_speed_avg": 30,
        "rainfall_3d_total": 0,
        "rainfall_7d_total": 0,
        "rain_days_7d": 0,
        "max_daily_rain_7d": 0
    }
}
]
    test_scenarios.extend([
    {
        "name": "💎 SCENARIO: The Morel Spring Awakening", 
        "msg": "Early spring warmup. Soil crosses the 10°C threshold with high moisture.", 
        "data": {
            "month": 5,
            "air_temp_day_avg": 19.0,
            "air_temp_night_avg": 8.0,
            "soil_temp_avg": 11.5,
            "soil_moisture_avg": 65.0,
            "air_humidity_avg": 70.0,
            "wind_speed_avg": 12.0,
            "rainfall_3d_total": 15.0,
            "rainfall_7d_total": 40.0,
            "rain_days_7d": 4,
            "max_daily_rain_7d": 15.0
        }
    },
    {
        "name": "❌ SCENARIO: The False Spring Trap", 
        "msg": "Warm week followed by a sudden deep freeze. Mycelium is active but fruit bodies freeze.", 
        "data": {
            "month": 4,
            "air_temp_day_avg": 5.0,
            "air_temp_night_avg": -5.0,
            "soil_temp_avg": 4.0,
            "soil_moisture_avg": 55.0,
            "air_humidity_avg": 40.0,
            "wind_speed_avg": 25.0,
            "rainfall_3d_total": 0.0,
            "rainfall_7d_total": 20.0,
            "rain_days_7d": 2,
            "max_daily_rain_7d": 12.0
        }
    },
    {
        "name": "❌ SCENARIO: The Sahara Breath", 
        "msg": "Sufficient soil moisture, but extremely low air humidity and high wind (evapotranspiration kill).", 
        "data": {
            "month": 8,
            "air_temp_day_avg": 28.0,
            "air_temp_night_avg": 18.0,
            "soil_temp_avg": 21.0,
            "soil_moisture_avg": 45.0,
            "air_humidity_avg": 15.0,
            "wind_speed_avg": 35.0,
            "rainfall_3d_total": 0.0,
            "rainfall_7d_total": 10.0,
            "rain_days_7d": 1,
            "max_daily_rain_7d": 10.0
        }
    },
    {
        "name": "💎 SCENARIO: The Indian Summer Tail-End", 
        "msg": "Warm November day. Soil still holds summer heat, moisture is perfect.", 
        "data": {
            "month": 11,
            "air_temp_day_avg": 15.0,
            "air_temp_night_avg": 7.0,
            "soil_temp_avg": 10.0,
            "soil_moisture_avg": 50.0,
            "air_humidity_avg": 85.0,
            "wind_speed_avg": 5.0,
            "rainfall_3d_total": 10.0,
            "rainfall_7d_total": 25.0,
            "rain_days_7d": 3,
            "max_daily_rain_7d": 10.0
        }
    },
    {
        "name": "❌ SCENARIO: Post-Peak Exhaustion", 
        "msg": "Conditions are perfect, but the forest has just finished a massive flush (biological cooldown).", 
        "data": {
            "month": 10,
            "air_temp_day_avg": 15.0,
            "air_temp_night_avg": 9.0,
            "soil_temp_avg": 12.0,
            "soil_moisture_avg": 60.0,
            "air_humidity_avg": 90.0,
            "wind_speed_avg": 2.0,
            "rainfall_3d_total": 5.0,
            "rainfall_7d_total": 90.0, # Massive rain earlier indicates the flush already happened
            "rain_days_7d": 6,
            "max_daily_rain_7d": 45.0
        }
    }
])
    from datetime import datetime 
    print("--- Running Stress Tests ---")
    for case in test_scenarios:
        t1 = datetime.now()
        res = predict_growth(case['data'], model, preprocessor, device)
        print(datetime.now() - t1)
        print(f"{case['name']}: Porcini {res['Porcini Score']}% | Chanterelle {res['Chanterelle Score']}%")


    # Importance Analysis
    print("\n--- Global Feature Importance ---")
    importance = get_feature_importance(test_scenarios, model, preprocessor, device)
    print(importance.to_string(index=False))