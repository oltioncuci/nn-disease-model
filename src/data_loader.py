import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

class GrowthDataset(Dataset):
    def __init__(self, X, y):
        """
        Expects X and y to be pre-processed numpy arrays or tensors.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_features(df):
    """Handles feature engineering like cyclical encoding."""
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Cyclical Encoding for Month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df['wind_speed_avg'] = np.log1p(df['wind_speed_avg'])

    def butterworth_window(x, center, half_width, order=4):
        relative_dist = (x - center) / half_width
        return 1 / (1 + np.power(relative_dist, 2 * order))
        
    def sigmoid_window(x, low, high, steepness=0.8):
        def sig(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
        return sig(steepness * (x - low) * sig(steepness* (high - x)))    

        # Apply the same engineering you did in predict_growth
    df['porcini_temp_bw'] = butterworth_window(df['air_temp_day_avg'], 18.5, 3.5, order=4)
    df['chanterelle_temp_bw'] = butterworth_window(df['air_temp_day_avg'], 16.5, 4.0, order=4)
    # --- Porcini Night Logic (Needs a 'Cold Snap' Veto) ---
# Porcini usually stops growing if nights drop below 6-7°C consistently.
    #df['porcini_night_sig'] = sigmoid_window(df['air_temp_night_avg'], low=6.0, high=20.0, steepness=0.8)

    # --- Chanterelle Night Logic (Needs more warmth) ---
    # Chanterelles are tougher but prefer nights above 10°C for peak production.
    #df['chanterelle_night_sig'] = sigmoid_window(df['air_temp_night_avg'], low=9.5, high=22.0, steepness=0.7)

    df["temp_drop"] = df['air_temp_day_avg'] - df["air_temp_night_avg"]
    df['moisture_stability'] = df['rain_days_7d'] / (df['rainfall_7d_total'] + 1)


    #df["moisture_trend"] = df["rainfall_7d_total"] / df["soil_moisture_avg"]

    # Model the "Trigger Rain Event" (15-30mm)
    # High steepness makes it a clear "Switch"
    #df['porcini_rain_suitability'] = sigmoid_window(df['rainfall_7d_total'], low=15, high=30, steepness=0.9)

    # Model the "Humidity Floor" (75%)
    #df['humidity_suitability'] = sigmoid_window(df['air_humidity_avg'], low=75, high=100, steepness=0.5)

   #df.drop(["air_temp_day_avg", "air_temp_night_avg", "month"], axis=1, inplace=True)
    return df

def get_data_loaders(csv_path, batch_size=32, val_split=0.2, test_split=0.1):
    # 1. Load and Shuffle
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = preprocess_features(df)

    # 2. Define Columns # 'air_temp_day_avg','air_humidity_avg',
    # feature_cols = [
    #     'month',  'air_temp_night_avg', 'soil_temp_avg', 
    #     'soil_moisture_avg',  'wind_speed_avg',
    #     'rainfall_3d_total', 'rainfall_7d_total', 'rain_days_7d',
    #     'max_daily_rain_7d', 'month_sin', 'month_cos', "porcini_temp_bw", "chanterelle_temp_bw", "porcini_rain_suitability", "humidity_suitability",
    # ]
    feature_cols = [
        "air_temp_day_avg", 
        'air_temp_night_avg', 
        'soil_temp_avg', 
        'soil_moisture_avg', 'air_humidity_avg', 'wind_speed_avg',
        'rainfall_3d_total', 'rainfall_7d_total', 'rain_days_7d',
        'max_daily_rain_7d', "porcini_temp_bw", "chanterelle_temp_bw", #"temp_drop", #"moisture_stability"
    ]

    month_data = ["month_sin", "month_cos"]
    target_cols = ['porcini_growth_score', 'chanterelle_growth_score']

    # 3. Split Data (Clean Indices)
    n = len(df)
    test_idx = int(n * (1 - test_split))
    val_idx = int(test_idx * (1 - val_split))

    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]

    # 4. Fit Preprocessor ONLY on Train Data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("months", "passthrough", month_data)
        ],
        remainder='drop'
    )

    all_features = feature_cols + month_data
    
    X_train = preprocessor.fit_transform(train_df[all_features])
    X_val = preprocessor.transform(val_df[all_features])
    X_test = preprocessor.transform(test_df[all_features])

    # 5. Prepare Targets (Scaled 0-1)
    y_train = train_df[target_cols].values / 100.0
    y_val = val_df[target_cols].values / 100.0
    y_test = test_df[target_cols].values / 100.0

    # 6. Save Preprocessor
    os.makedirs('utils', exist_ok=True)
    joblib.dump(preprocessor, 'utils/preprocessor1.joblib')

    # 7. Create Datasets and Loaders
    train_ds = GrowthDataset(X_train, y_train)
    val_ds = GrowthDataset(X_val, y_val)
    test_ds = GrowthDataset(X_test, y_test)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )

if __name__ == "__main__":
    # Note: Only one CSV path is needed if you are splitting a single dataset
    train_l, val_l, test_l = get_data_loaders('../data/growth_data.csv')
    
    # Quick Check
    features, targets = next(iter(train_l))
    print(f"Batch Shape: {features.shape}")
    print(f"Sample Target: {targets[0]}")