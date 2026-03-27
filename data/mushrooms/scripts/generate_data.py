import pandas as pd
import numpy as np

def generate_mushroom_dataset(n_samples=5000):
    """
    Generates a synthetic dataset for Porcini and Chanterelle growth probability
    based on the specific conditions provided.
    """
    np.random.seed(42)
    
    # 1. Generate Input Features (X)
    data = {
        'air_temp_day': np.random.uniform(5, 38, n_samples),
        'air_temp_night': np.random.uniform(2, 25, n_samples),
        'soil_temp': np.random.uniform(5, 28, n_samples),
        'air_humidity': np.random.uniform(20, 100, n_samples),
        'soil_moisture': np.random.uniform(10, 95, n_samples),
        'wind_speed': np.random.uniform(0, 10, n_samples),
        'rain_cumulative_7d': np.random.uniform(0, 80, n_samples),
        'rain_days_in_window': np.random.randint(0, 8, n_samples),
        'hours_above_28c': np.random.uniform(0, 24, n_samples),
        'consecutive_moist_days': np.random.randint(0, 15, n_samples),
        'is_peak_season': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    }

    df = pd.DataFrame(data)

    def calculate_mushroom_logic(row, species='porcini'):
        score = 1.0
        
        # --- TEMPERATURE LOGIC ---
        # Optimal 15-22. Below 10-12 is min.
        if row['air_temp_day'] < 12:
            score *= np.interp(row['air_temp_day'], [5, 12], [0.1, 0.8])
        elif 15 <= row['air_temp_day'] <= 22:
            score *= 1.0
        elif row['air_temp_day'] > 22:
            # Decay towards suppression at 28
            score *= np.interp(row['air_temp_day'], [22, 28], [1.0, 0.3])
        
        # Night Temp (10-16 preferred)
        if not (10 <= row['air_temp_night'] <= 16):
            score *= 0.8

        # --- RAIN & MOISTURE LOGIC ---
        # Trigger: 15-30mm cumulative
        if 15 <= row['rain_cumulative_7d'] <= 30:
            score *= 1.2 
        elif row['rain_cumulative_7d'] > 60:
            score *= 0.4 # Soil saturation penalty
        else:
            score *= 0.5 # Sub-optimal rain

        # Light continuous rain vs storm (rain_days_in_window)
        # 3-5 days of rain out of 7 is better than 1 day of heavy rain
        if row['rain_days_in_window'] >= 3:
            score *= 1.1
        elif row['rain_days_in_window'] == 1:
            score *= 0.7

        # Soil Moisture (60-80% optimal)
        if 60 <= row['soil_moisture'] <= 80:
            score *= 1.0
        else:
            dist = min(abs(row['soil_moisture'] - 60), abs(row['soil_moisture'] - 80))
            score *= np.exp(-0.05 * dist)

        # Consecutive Moist Days (5-10 days requirement)
        if 5 <= row['consecutive_moist_days'] <= 10:
            score *= 1.0
        else:
            score *= 0.6

        # --- WIND & HUMIDITY ---
        if not (75 <= row['air_humidity'] <= 95):
            score *= 0.7
        if row['wind_speed'] > 3:
            score *= 0.5

        # --- SPECIES SPECIFIC KILL SWITCHES ---
        if species == 'porcini':
            if row['air_temp_day'] >= 32: 
                return 0.0 # Fruiting stops
        
        if species == 'chanterelle':
            if row['air_temp_day'] >= 28:
                score *= 0.2 # Heavily suppressed

        # --- SEASONALITY ---
        if row['is_peak_season'] == 0:
            score *= 0.05 # Growth very unlikely outside May-June/Sept-Oct
            
        return min(max(score, 0), 1.0)

    # Apply calculations to generate Targets (Y)
    df['porcini_score'] = df.apply(lambda r: calculate_mushroom_logic(r, 'porcini'), axis=1)
    df['chanterelle_score'] = df.apply(lambda r: calculate_mushroom_logic(r, 'chanterelle'), axis=1)

    return df

# Example Usage:
df = generate_mushroom_dataset(1000)
# print(df[['air_temp_day', 'rain_cumulative_7d', 'porcini_score', 'chanterelle_score']].head())
df.to_csv("mushroom_growth_dataset_test.csv")