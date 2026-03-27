import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from src.data_loader import get_data_loaders
from src.architecture import GrowthClassifer

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained Multi-Target model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {DEVICE}")

    # 1. Load Data
    test_loader, _, _ = get_data_loaders(
        'data/mushrooms/scripts/balanced_mushroom_growth_dataset_test.csv',
        'data/mushrooms/scripts/balanced_mushroom_growth_dataset_test.csv',
        args.batch_size
    )

    # 2. Setup Architecture
    input_dim = test_loader.dataset.X.shape[1]
    output_dim = test_loader.dataset.y.shape[1]
    mushroom_names = ['Porcini', 'Chanterelle'] # Defined based on your target columns

    model = GrowthClassifer(
        input_size=input_dim, 
        hidden_size=args.hidden_size, 
        num_classes=output_dim
    ).to(DEVICE)

    # Load Weights
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_trues = []

    # 3. Inference Loop
    print(f"Processing {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Ensure Float32 to match model weights
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).float()
            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_trues.append(labels.cpu().numpy())

    y_pred_raw = np.vstack(all_preds)
    y_true_raw = np.vstack(all_trues)

    # 4. Evaluation Logic (Cleaned up - No Masking)
    report_data = []
    per_species_errors = {name: [] for name in mushroom_names}

    # Iterate through samples and species
    for i in range(len(y_true_raw)):
        for idx, species_name in enumerate(mushroom_names):
            actual = y_true_raw[i, idx] * 100
            pred = y_pred_raw[i, idx] * 100
            error = abs(actual - pred)
            
            per_species_errors[species_name].append(error)
            
            report_data.append({
                'Mushroom_Type': species_name,
                'Actual_Score': round(actual, 2),
                'Predicted_Score': round(pred, 2),
                'Error_Margin': round(error, 2)
            })

    # 5. Summary Report
    print("\n" + "="*45)
    print(f"{'Mushroom Type':<20} | {'MAE (%)':<10}")
    print("-" * 45)
    
    overall_errors = []
    for name in mushroom_names:
        avg_err = np.mean(per_species_errors[name])
        overall_errors.append(avg_err)
        print(f"{name:<20} | {avg_err:>8.2f}%")
    
    print("-" * 45)
    print(f"{'OVERALL SYSTEM MAE':<20} | {np.mean(overall_errors):>8.2f}%")
    print("="*45)

    # 6. Visualization
    report_df = pd.DataFrame(report_data)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=report_df, x='Actual_Score', y='Predicted_Score', hue='Mushroom_Type', alpha=0.4)
    
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Perfect Prediction')
    plt.title('Multi-Target Model: Actual vs Predicted Growth')
    plt.xlabel('Actual Growth Score (%)')
    plt.ylabel('Predicted Growth Score (%)')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/test_performance_scatter.png')
    plt.show()

    # 7. Save Detailed CSV
    os.makedirs('reports', exist_ok=True)
    report_df.to_csv("reports/test_detailed_results.csv", index=False)
    print(f"Results saved to reports/test_detailed_results.csv")

if __name__ == "__main__":
    main()