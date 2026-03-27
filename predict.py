import torch
import joblib
import pandas as pd
import numpy as np
import os
from src.architecture import GrowthClassifer

def predict_growth(weather_data, model_path="models/growth_model_best.pth", hidden_size=64):
    """
    Takes weather conditions and returns growth probabilities for all species.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Scaler (Ensure this was saved during training)
    scaler = joblib.load("utils/scaler.joblib")
    
    # 2. Define Species Names (Must match training order)
    mushroom_species = ['Porcini', 'Chanterelle']
    
    # 3. Initialize & Load Model
    # input_dim should be 11 (the features we defined in the generator)
    input_dim = scaler.n_features_in_ 
    output_dim = len(mushroom_species)
    
    model = GrowthClassifer(input_size=input_dim, hidden_size=hidden_size, num_classes=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    # 4. Process Input Data
    if isinstance(weather_data, dict):
        df = pd.DataFrame(weather_data)
    else:
        df = weather_data.copy()

    # 5. Feature Selection (Ordering MUST match the 11 columns in the generator)
    feature_cols = [
        'air_temp_day', 'air_temp_night', 'soil_temp', 'air_humidity', 
        'soil_moisture', 'wind_speed', 'rain_cumulative_7d', 
        'rain_days_in_window', 'hours_above_28c', 'consecutive_moist_days', 
        'is_peak_season'
    ]
    
    X = df[feature_cols]
    
    # 6. Inference
    X_scaled = scaler.transform(X.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(X_tensor) # Shape: [N, 2]
    
    # Convert from 0-1 to 0-100%
    growth_scores = outputs.cpu().numpy() * 100
    
    results = []
    for i in range(len(df)):
        # Create a dictionary of results for each row
        prediction_map = {
            mushroom_species[j]: round(float(growth_scores[i][j]), 2) 
            for j in range(output_dim)
        }
        results.append(prediction_map)
    
    return results

if __name__ == "__main__":
    # Sample "Current Sensor Data"
    # Row 1: Perfect Porcini conditions | Row 2: Too hot/dry
    current_sensors = {
    # Location 1: "Muggy Summer" (Chanterelle Thrives, Porcini Suppressed)
    # Location 2: "Crisp Autumn" (Porcini Thrives, Chanterelle Suppressed)
    'air_temp_day': [29.5, 14.5],           
    'air_temp_night': [19.0, 9.5],         
    'soil_temp': [22.0, 13.0],              
    'air_humidity': [92.0, 82.0],           
    'soil_moisture': [80.0, 72.0],          
    'wind_speed': [0.5, 1.0],               
    'rain_cumulative_7d': [35.0, 22.0],     
    'rain_days_in_window': [5, 4],          
    'hours_above_28c': [6, 0],              
    'consecutive_moist_days': [8, 7],      
    'is_peak_season': [1, 1]                
}

    predictions = predict_growth(current_sensors)

    print("\n" + "="*40)
    print(f"{'SITE':<10} | {'PORCINI %':<12} | {'CHANTERELLE %':<12}")
    print("-" * 40)
    
    for i, res in enumerate(predictions):
        print(f"Location {i+1:<2} | {res['Porcini']:>10}% | {res['Chanterelle']:>12}%")
    print("="*40)