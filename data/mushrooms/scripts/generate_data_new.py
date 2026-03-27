import pandas as pd
import numpy as np

def generate_balanced_mushroom_dataset(n_samples=5000):
    """
    Generates a synthetic dataset for Porcini and Chanterelle growth probability
    with scores balanced across the 0-1 range using rank-based normalization.
    """
    np.random.seed(42)
    
    # 1. Generate Input Features (X)
    # Using specific distributions to hit "optimal" zones more often
    data = {
        'air_temp_day': np.random.normal(18, 7, n_samples).clip(5, 38),
        'air_temp_night': np.random.normal(13, 5, n_samples).clip(2, 25),
        'soil_temp': np.random.normal(16, 5, n_samples).clip(5, 28),
        'air_humidity': np.random.uniform(40, 100, n_samples),
        'soil_moisture': np.random.normal(65, 15, n_samples).clip(10, 95),
        'wind_speed': np.random.exponential(2, n_samples).clip(0, 15),
        'rain_cumulative_7d': np.random.gamma(2, 10, n_samples).clip(0, 80),
        'rain_days_in_window': np.random.randint(0, 8, n_samples),
        'hours_above_28c': np.random.exponential(2, n_samples).clip(0, 24),
        'consecutive_moist_days': np.random.poisson(6, n_samples).clip(0, 15),
        'is_peak_season': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    }

    df = pd.DataFrame(data)

    def calculate_mushroom_logic(row, species='porcini'):
        # Start with a base potential
        score = 0.5 
        
        # --- TEMPERATURE (Additive adjustments) ---
        if 15 <= row['air_temp_day'] <= 22:
            score += 0.2
        elif 12 <= row['air_temp_day'] < 15 or 22 < row['air_temp_day'] <= 26:
            score += 0.1
        else:
            score -= 0.2
            
        # --- RAIN & MOISTURE ---
        if 15 <= row['rain_cumulative_7d'] <= 40:
            score += 0.2
        elif row['rain_cumulative_7d'] > 60 or row['rain_cumulative_7d'] < 5:
            score -= 0.2
            
        # Soil Moisture (60-80% optimal)
        if 60 <= row['soil_moisture'] <= 85:
            score += 0.1
        elif row['soil_moisture'] < 40:
            score -= 0.2

        # --- MULTIPLICATIVE PENALTIES (Critical inhibitors) ---
        # Severe wind dries out the mushrooms
        if row['wind_speed'] > 6:
            score *= 0.5
        
        # Seasonality (Reduced penalty to maintain variation)
        if row['is_peak_season'] == 0:
            score *= 0.2
        
        # Species Specific Kill-switches
        if species == 'porcini':
            if row['air_temp_day'] >= 32: 
                return 0.0
        if species == 'chanterelle':
            if row['air_temp_day'] >= 28:
                score *= 0.3
                
        return max(score, 0)

    # Calculate raw underlying scores
    df['porcini_raw'] = df.apply(lambda r: calculate_mushroom_logic(r, 'porcini'), axis=1)
    df['chanterelle_raw'] = df.apply(lambda r: calculate_mushroom_logic(r, 'chanterelle'), axis=1)

    # 2. Balancing: Rank-based Scaling
    # This transforms the distribution into a uniform [0, 1] range
    def balance_series(series):
        # Convert values to their percentile rank (0.0 to 1.0)
        return (series - series.min()) / (series.max() - series.min())

    df['porcini_score'] = balance_series(df['porcini_raw'])
    df['chanterelle_score'] = balance_series(df['chanterelle_raw'])

    # Cleanup raw helper columns
    return df.drop(columns=['porcini_raw', 'chanterelle_raw'])

# Generate and Save
df_final = generate_balanced_mushroom_dataset(50000)
df_final.to_csv("balanced_mushroom_growth_dataset.csv", index=False)

print("Score statistics (Balanced):")
print(df_final[['porcini_score', 'chanterelle_score']].describe())