import psycopg2
import psycopg2.extras
from typing import List, Optional
from pydantic import BaseModel, RootModel

import joblib
from datetime import datetime

import pandas as pd
import numpy as np

import torch
from sklearn.base import BaseEstimator, RegressorMixin

from src.architecture import GrowthRegressor

class WeatherForecast(BaseModel):
    forecastts: datetime
    latitude: float
    longitude: float
    temperature: float
    rain: float
    humidity: float
    windspeed: float
    soiltemperature: float
    soilmoisture: float

class WeatherForecastList(RootModel):
    root: List[WeatherForecast]

def getData() -> Optional[List[WeatherForecast]]:
    # query = """ 
    #     SELECT DISTINCT ON (forecastts)
    #         forecastts,
    #         latitude,
    #         longitude,
    #         temperature,
    #         rain,
    #         relativehumidity AS humidity,
    #         windspeed,
    #         soiltemperature,
    #         soilmoisture
    #     FROM public."WeatherForecastHourly"
    #     WHERE 
    #         forecastts >= CURRENT_TIMESTAMP 
    #         AND 
    #         forecastts < CURRENT_TIMESTAMP + INTERVAL '7 days'
    #     ORDER BY forecastts ASC, id DESC;
    # """

    query = """
        SELECT DISTINCT ON (forecastts)
            forecastts,
            latitude,
            longitude,
            temperature,
            rain,
            relativehumidity AS humidity,
            windspeed,
            soiltemperature,
            soilmoisture
        FROM public."WeatherForecastHourly"
        WHERE 
            forecastts >= CURRENT_TIMESTAMP - INTERVAL '4 days'
            AND forecastts < CURRENT_TIMESTAMP + INTERVAL '3 days'
        ORDER BY forecastts ASC;
    """
    
    try:
        with psycopg2.connect(
            database="agrodb", 
            host="10.10.17.17", 
            user="agro", 
            password="agr0c@cttu$", 
            port="5432"
        ) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                
                if not rows:
                    print("Query returned 0 rows.")
                    return None
                
                validated_data = WeatherForecastList.model_validate([dict(row) for row in rows])

                data_dicts = [item.model_dump() for item in validated_data.root]

                df = pd.DataFrame(data_dicts)

                return df

    except Exception as e:
        print(f"Database error: {e}")
        return None


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
            return self.model(X_t).cpu().numpy()
        

def engineer_features(df):
    df = df.copy()

    print(df)

    df['hour'] = df['forecastts'].dt.hour
    df['date'] = df['forecastts'].dt.date

    night_mask = (df["hour"] >= 21) | (df["hour"] <= 6)
    night_stats = df[night_mask].groupby('date')['temperature'].mean().rename('air_temp_night_avg')

    day_mask = (df["hour"] >= 6) & (df["hour"] <= 21) 
    day_stats = df[day_mask].groupby('date')['temperature'].mean().rename('air_temp_day_avg') 

    daily_stats = df.groupby('date').agg({
        'soiltemperature': 'mean',
        'soilmoisture': 'mean',
        'humidity': 'mean',
        'windspeed': 'mean',
        'rain': 'sum'
    })

    df_daily = daily_stats.join([night_stats, day_stats]).reset_index()

    df_daily = df_daily.rename(columns={
        'date': 'forecastts',
        'soiltemperature': 'soil_temp_avg',
        'soilmoisture': 'soil_moisture_avg',
        'humidity': 'air_humidity_avg',
        'windspeed': 'wind_speed_avg',
        'rain': 'daily_rain'
    })

    df_daily['soil_moisture_avg'] *= 100

    df_daily = df_daily.sort_values('forecastts')

    df_daily["rainfall_3d_total"] = df_daily["daily_rain"].rolling(window=3, min_periods=1).sum()
    df_daily["rainfall_7d_total"] = df_daily["daily_rain"].rolling(window=7, min_periods=1).sum()

    df_daily['rain_days_7d'] = df_daily['daily_rain'].rolling(window=7, min_periods=1).apply(lambda x: (x > 0.1).sum())

    df_daily['max_daily_rain_7d'] = df_daily['daily_rain'].rolling(window=7, min_periods=1).max()

    df_daily['month'] = pd.to_datetime(df_daily['forecastts']).dt.month
    df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month'] / 12)
    df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['month'] / 12)

    def butterworth_window(x, center, half_width, order=4):
        relative_dist = (x - center) / half_width
        return 1 / (1 + np.power(relative_dist, 2 * order))

    df_daily['porcini_temp_bw'] = butterworth_window(df_daily['air_temp_day_avg'], 18, 4, order=2)
    df_daily['chanterelle_temp_bw'] = butterworth_window(df_daily['air_temp_day_avg'], 16.5, 4.0, order=2)

    air_temp_day_avg = df_daily['air_temp_day_avg'].mean()
    air_temp_night_avg = df_daily['air_temp_night_avg'].mean()
    soil_temp_avg = df_daily['soil_temp_avg'].mean()
    air_humidity_avg = df_daily['air_humidity_avg'].mean()
    wind_speed_avg = df_daily['wind_speed_avg'].mean()
    
    soil_moisture_avg = df_daily['soil_moisture_avg'].mean()

    rainfall_3d_total = df_daily['rainfall_3d_total'].max()
    rainfall_7d_total = df_daily['rainfall_7d_total'].max()
    rain_days_7d = df_daily['rain_days_7d'].max()
    max_daily_rain_7d = df_daily['max_daily_rain_7d'].max()

    month_sin = df_daily['month_sin'].iloc[0]
    month_cos = df_daily['month_cos'].iloc[0]
    porcini_temp_bw = df_daily['porcini_temp_bw'].mean()
    chanterelle_temp_bw = df_daily['chanterelle_temp_bw'].mean()

    single_row = {
        "air_temp_day_avg": air_temp_day_avg,
        "air_temp_night_avg": air_temp_night_avg,
        "soil_temp_avg": soil_temp_avg,
        "soil_moisture_avg": soil_moisture_avg,
        "air_humidity_avg": air_humidity_avg,
        "wind_speed_avg": wind_speed_avg,
        "rainfall_3d_total": rainfall_3d_total,
        "rainfall_7d_total": rainfall_7d_total,
        "rain_days_7d": rain_days_7d,
        "max_daily_rain_7d": max_daily_rain_7d,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "porcini_temp_bw": porcini_temp_bw,
        "chanterelle_temp_bw": chanterelle_temp_bw
    }
    
    df = pd.DataFrame([single_row])

    return df


def predict_growth(weather_data, model, preprocessor, device):
    df_engineered = engineer_features(weather_data)

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

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = joblib.load('utils/preprocessor.joblib')
    model = GrowthRegressor(input_size=14, num_classes=2)
    checkpoint = torch.load("models/growth_model_best_v5.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    print(model)

    data = getData()
    res = predict_growth(data, model, preprocessor, DEVICE)
    print(res)
