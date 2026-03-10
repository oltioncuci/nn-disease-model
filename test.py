import torch
import torch.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import argparse
import os
from datetime import datetime

from src.data_loader import get_data_loader
from src.architecture import DiseaseClassifer

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")

    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, encoder = get_data_loader(
        'data/apple/apple_disease_training_data.csv', 
        'data/apple/apple_disease_test_data.csv', 
        args.batch_size
    )

    sample_x, _ = test_loader.dataset[0]
    input_dim = sample_x.shape[0]
    output_dim = len(encoder.classes_)
    
    model = DiseaseClassifer(input_size=input_dim, hidden_size=32, num_classes=output_dim).to(DEVICE)

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    # Testing
    print("Starting Testing...")
    model.eval()

    y_pred = []
    y_true = []
    all_probs = []

    with torch.zero_grad():
        for inputs, labels in test_loader:
            inputs, outputs = inputs.to(DEVICE), model.to(DEVICE)

            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            _, predicted = torch.max(outputs, 1)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, 
                yticklabels=encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Apple Disease Classification Confusion Matrix')
    
    model_tag = os.path.basename(args.model_path).replace('.pth', '')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(f'figures/{model_tag}-{datetime.now().strftime(timestamp)}.png')
    plt.show()

    all_probs_matrix = np.vstack(all_probs)

    true_names = encoder.inverse_transform(y_true)
    pred_names = encoder.inverse_transform(y_pred)

    report_df = pd.DataFrame({
        'Actual_Disease': true_names,
        'Predicted_Disease': pred_names,
        'Correct': [t == p for t, p in zip(true_names, pred_names)]
    })

    for i, class_name in enumerate(encoder.classes_):
        report_df[f'Prob_{class_name}_%'] = probs[:, i] * 100

    csv_path = f'reports/{model_tag}_{timestamp}_detailed.csv'
    report_df.to_csv(csv_path, index=False)
    print(f"Detailed CSV Report saved.")

if __name__ == "__main__":
    main()