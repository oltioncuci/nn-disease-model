import torch
import joblib
import pandas as pd
import numpy as np
from src.architecture import DiseaseClassifer

def predict(input_data, model_path="models/nn_1_final_acc_97.8_2026-03-12_09-50-50.pth"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load("utils/scaler.joblib")
    encoder = joblib.load("utils/encoder.joblib")

    input_dim = scaler.n_features_in_
    output_dim = len(encoder.classes_)

    model = DiseaseClassifer(input_size=input_dim, num_classes=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if isinstance(input_data, (dict, list)):
        df = pd.DataFrame(input_data)
    else:
        df = input_data

    #actual_labels = df['Disease_Type'].values
    X = df.drop(columns=['Disease_Type'])

    X_scaled = scaler.transform(X.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)

    class_names = encoder.classes_
    batch_results = []
    for i in range(len(df)):
        sample_probs = probs[i].cpu().numpy()
        # Create a dictionary of {DiseaseName: Probability}
        prob_map = {class_names[j]: sample_probs[j] for j in range(len(class_names))}
        # Sort it so the highest confidence is at the top
        sorted_probs = dict(sorted(prob_map.items(), key=lambda item: item[1], reverse=True))
        
        batch_results.append({
            'actual': df['Disease_Type'].iloc[i],
            'predictions': sorted_probs
        })
    
    return batch_results

if __name__ == "__main__":

    # alternaria_data = {
    #     'Month': [7], 
    #     'Temperature_C': [28.5], 
    #     'Humidity_PRC': [88], 
    #     'Leaf_Wetness_Hr': [12], 
    #     'Rainfall_mm': [4.2], 
    #     'Wind_Speed_ms': [1.2], 
    #     'UV_Index': [8], 
    #     'Light_Intensity_Lux': [48000], 
    # }
    
    # disease, confidence = predict(alternaria_data)
    # print(f"Prediction: {disease} ({confidence.item()*100:.2f}%)")

    sample_data = {
        'Month': [7, 5, 5, 6, 7, 6], 
        'Temperature_C': [28, 18, 22, 25, 24, 22], 
        'Humidity_PRC': [85, 90, 80, 70, 65, 50], 
        'Leaf_Wetness_Hr': [10, 14, 8, 4, 2, 0], 
        'Rainfall_mm': [5, 12, 8, 2, 0, 0], 
        'Wind_Speed_ms': [1.5, 2.5, 3.0, 4.5, 2.0, 2.5], 
        'UV_Index': [8, 4, 5, 7, 9, 6], 
        'Light_Intensity_Lux': [45000, 20000, 30000, 40000, 50000, 35000], 
        'Disease_Type': [
            'Alternaria', 
            'Apple Scab', 
            'Cedar Apple Rust', 
            'Fire Blight', 
            'Prob_Powdery Mildew', 
            'Healthy'
        ]
    }

    results = predict(sample_data)

    output_file = "prediction_results.txt"

    with open(output_file, "w") as f:
        for res in results:
            # We use f.write instead of print
            f.write(f"\nTarget: {res['actual']}\n")
            f.write("-" * 30 + "\n")
            
            for disease, conf in res['predictions'].items():
                # Highlight the top prediction
                keys = list(res['predictions'].keys())
                prefix = "-> " if disease == keys[0] else "   "
                
                line = f"{prefix}{disease:<20}: {conf*100:>6.2f}%\n"
                f.write(line)
            
            f.write("\n" + "="*30 + "\n")

    print(f"Results successfully saved to {output_file}")