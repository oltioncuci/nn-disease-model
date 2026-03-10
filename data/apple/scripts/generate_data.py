import pandas as pd
import numpy as np

import random



def generate_apple_disease_data(samples_per_class=500):
    data = []
    random.seed(2)
    # Disease names and their target months
    # 0: Healthy, 1: Apple Scab, 2: Powdery Mildew, 3: Fire Blight, 4: Cedar Rust, 5: Alternaria
    
    for _ in range(samples_per_class):
        # --- 1. APPLE SCAB (Venturia inaequalis) ---
        month = np.random.choice([4, 5, 6])
        temp = np.random.uniform(10, 24)
        wetness = np.random.uniform(6, 15)
        humidity = np.random.uniform(85, 98)
        rain = np.random.uniform(0.2, 10.0)
        risk = "High" if (16 <= temp <= 22 and wetness >= 9) else "Medium"
        data.append([month, temp, humidity, wetness, rain, np.random.uniform(2, 5), np.random.uniform(0, 4), np.random.uniform(2000, 9000), "Apple Scab", risk])

        # --- 2. POWDERY MILDEW (Podosphaera leucotricha) ---
        month = np.random.choice([5, 6, 7])
        temp = np.random.uniform(15, 27)
        humidity = np.random.uniform(60, 80)
        wetness = 0  # Does not require free water
        rain = np.random.uniform(0, 1.9)
        risk = "High" if (18 <= temp <= 24) else "Medium"
        data.append([month, temp, humidity, wetness, rain, np.random.uniform(2, 4), np.random.uniform(2, 6), np.random.uniform(1000, 5000), "Powdery Mildew", risk])

        # --- 3. FIRE BLIGHT (Erwinia amylovora) ---
        month = np.random.choice([4, 5])
        temp = np.random.uniform(18, 30)
        humidity = np.random.uniform(60, 95)
        rain = np.random.uniform(0.1, 5.0)
        risk = "High" if (temp > 21 and humidity > 75) else "Medium"
        data.append([month, temp, humidity, np.random.uniform(1, 4), rain, np.random.uniform(3, 6), np.random.uniform(3, 8), np.random.uniform(5000, 15000), "Fire Blight", risk])

        # --- 4. CEDAR APPLE RUST ---
        month = np.random.choice([4, 5, 6])
        temp = np.random.uniform(10, 24)
        wetness = np.random.uniform(4, 10)
        humidity = np.random.uniform(85, 95)
        risk = "High" if (15 <= temp <= 21 and wetness >= 5) else "Medium"
        data.append([month, temp, humidity, wetness, np.random.uniform(1.0, 15.0), np.random.uniform(2, 5), np.random.uniform(0, 4), np.random.uniform(2000, 10000), "Cedar Apple Rust", risk])

        # --- 5. ALTERNARIA LEAF BLOTCH ---
        month = np.random.choice([6, 7, 8])
        temp = np.random.uniform(20, 30)
        wetness = np.random.uniform(8, 15)
        humidity = np.random.uniform(85, 95)
        risk = "High" if (temp >= 24 and wetness >= 10) else "Medium"
        data.append([month, temp, humidity, wetness, np.random.uniform(0.5, 5.0), np.random.uniform(2, 5), np.random.uniform(0, 4), np.random.uniform(1000, 8000), "Alternaria", risk])

        # --- 6. HEALTHY (No Disease) ---
        # Generate conditions that don't meet disease criteria (e.g., dry, high UV, or hot/cold)
        month = np.random.choice([1, 2, 3, 9, 10, 11, 12]) 
        temp = np.random.uniform(0, 35)
        humidity = np.random.uniform(20, 50) # Too dry for most
        wetness = 0
        rain = 0
        data.append([month, temp, humidity, wetness, rain, np.random.uniform(0, 10), np.random.uniform(7, 11), np.random.uniform(15000, 30000), "Healthy", "Low"])

    columns = [
        'Month', 'Temperature_C', 'Humidity_PRC', 'Leaf_Wetness_Hr', 
        'Rainfall_mm', 'Wind_Speed_ms', 'UV_Index', 'Light_Intensity_Lux', 
        'Disease_Type', 'Risk_Level'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.drop("Risk_Level", axis=1, inplace=True)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Generate 6000 rows (1000 for each of the 6 categories)
apple_df = generate_apple_disease_data(10000)

# Save to CSV
apple_df.to_csv('../apple_disease_training_data.csv', index=False)

print("Dataset generated successfully!")
print(apple_df['Disease_Type'].value_counts())
print(apple_df.head())