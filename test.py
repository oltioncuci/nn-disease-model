import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nbformat as nbf

from sklearn.metrics import confusion_matrix, classification_report

import argparse
import os
from datetime import datetime

from src.data_loader import get_data_loaders
from src.architecture import DiseaseClassifer

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")

    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, encoder = get_data_loaders(
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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

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
    figure_name = f"{model_tag}-figure.png"
    plt.savefig(f'figures/{figure_name}')
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
        report_df[f'Prob_{class_name}_%'] = all_probs_matrix[:, i] * 100

    csv_filename = f"{model_tag}_detailed.csv"
    csv_path = f"reports/{csv_filename}"
    report_df.to_csv(csv_path, index=False)
    print(report_df.head())
    print(f"Detailed CSV Report saved in {csv_path}.")

    os.makedirs('figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    nb = nbf.v4.new_notebook()

    header_text = f"# Model Evaluation EDA\n**Source Report:** `{csv_path}`\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H-%M')}"
    
    import_code = "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline\nsns.set_theme(style='whitegrid')"

    load_code = f"df = pd.read_csv('../{csv_path}')\ndf.head()"

    analysis_code = (
        "print('--- Accuracy Overview ---')\n"
        "print(df['Correct'].value_counts(normalize=True))\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.countplot(data=df, x='Actual_Disease', hue='Correct')\n"
        "plt.title('Correct vs Incorrect Predictions per Class')\n"
        "plt.xticks(rotation=45)\n"
        "plt.show()"
    )
    image_display_code = (
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.image as mpimg\n"
        "import os\n"
        "import glob\n\n"
        "# Model tag from your test.py run\n"
        "model_tag = 'best_disease_model_final_acc_92.6_2026-03-10_16-28-07'\n"
        "figures_path = '../figures'\n"
        "\n"
        "# Try to find the figure with this model tag\n"
        "expected_pattern = os.path.join(figures_path, f'{model_tag}-figure.png')\n"
        "matching_files = glob.glob(expected_pattern)\n"
        "\n"
        "if matching_files:\n"
        "    img_path = matching_files[0]\n"
        "    img = mpimg.imread(img_path)\n"
        "    plt.figure(figsize=(12, 10))\n"
        "    plt.imshow(img)\n"
        "    plt.axis('off')\n"
        "    plt.title(f'Confusion Matrix: {os.path.basename(img_path)}', fontsize=14, pad=20)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    print(f'✓ Displaying: {os.path.basename(img_path)}')\n"
        "else:\n"
        "    print(f'✗ No figure found matching: {expected_pattern}')\n"
        "    print('\\nAvailable figures:')\n"
        "    if os.path.exists(figures_path):\n"
        "        for f in os.listdir(figures_path):\n"
        "            if f.endswith('.png'):\n"
        "                print(f'  - {f}')\n"
        "    else:\n"
        "        print(f'Figures folder not found at: {figures_path}')\n"
    )

    nb['cells'] = [
        nbf.v4.new_markdown_cell(header_text),
        nbf.v4.new_code_cell(import_code),
        nbf.v4.new_code_cell(load_code),
        nbf.v4.new_code_cell(analysis_code),
        nbf.v4.new_code_cell(image_display_code)
    ]

    notebook_filename = f'notebooks/{model_tag}_analysis.ipynb'
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Success! Notebook generated: {notebook_filename}")


if __name__ == "__main__":
    main()