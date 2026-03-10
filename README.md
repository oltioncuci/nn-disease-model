# 🍎 Disease Classification System

A modular Deep Learning pipeline built with PyTorch to classify crop diseases based on symptom data. This system features an automated training workflow with Early Stopping and a comprehensive testing suite.

---

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   pip install -r requirements.txt
2. **Data Placement**:
   Ensure your CSV files are in the following directory:
   - e.g. data/apple/apple_disease_training_data.csv
   - e.g. data/apple/apple_disease_test_data.csv

---

## 🔄 Project Workflow

The project is executed in two distinct phases:

### Training Phase (train.py):
- Loads and splits data into Training and Validation sets
- Trains the Neural Network (src/architecture.py) using the Adam optimizer
- Monitors Validation Loss; if no improvement occurs within the "patience" window, Early Stopping triggers
- Only the best model weights (lowest validation loss) are saved to disk after the session ends

### Testing Phase (test.py):
- Loads the saved .pth model weights
- Evaluates performance on the hold-out test set
- Generates a Confusion Matrix (Heatmap) to visualize misclassifications
- Exports a Detailed Probability Report (CSV) showing the confidence score for every disease class

---

## 🚀 Execution Commands

### Phase 1: Training
Run the training script with your desired hyperparameters:
e.g. python train.py --epochs 500 --batch-size 64 --lr 0.001 --model-name apple_model.pth --patience 20

### Phase 2: Testing
Run the testing script by pointing it to your best-saved model:
e.g. python test.py --model-path models/apple_model_final_acc_93.8.pth

---

## 📈 Analysis & Outputs

- **Classification Report**: View Precision, Recall, and F1-Score in the terminal to see which diseases are most difficult for the model
- **Figures (/figures)**: Check the generated Heatmap to see exactly which labels are being confused (e.g., if "Scab" is often mistaken for "Healthy")
- **Reports (/reports)**: Use the CSV to perform "Ambiguity Analysis." in folder /notebooks/notebook.ipynb. Look for rows where the model was wrong but had high confidence (potential data noise) vs. low confidence (model uncertainty)

---

## ⚙️ Configuration (Arguments)

| Argument | Description | Default |
|----------|-------------|---------|
| --epochs | Max training iterations | 50 |
| --batch_size | Number of samples per batch | 32 |
| --lr | Learning rate for Adam Optimizer | 0.001 |
| --patience | Epochs to wait before Early Stopping | 20 |
| --model-name | (Training) Name for saved model file | apple_model.pth |
| --model-path | (Testing) Path to saved .pth file | REQUIRED |