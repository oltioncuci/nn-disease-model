import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from src.data_loader import get_data_loaders
from src.architecture import DiseaseClassifer

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Training Arguments")

    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float,  default=0.001, help="Learning rate")
    parser.add_argument("--model-name", type=str, default='best_disease_model.pth', help="Saved Model Name")
    parser.add_argument("--patience", type=int, default=20, help="Define Early Stoppage")
    parser.add_argument("--early-stop", action='store_true', help="Early Stoppage")
    # TODO LATER add Dataset

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(DEVICE)

    train_loader, val_loader, test_loader, encoder = get_data_loaders(
        'data/apple/apple_disease_realistic_data.csv', 'data/apple/apple_disease_realistic_data_test.csv', args.batch_size
    )

    input_dim = train_loader.dataset.X.shape[1]
    hidden_dim = 32
    output_dim = len(encoder.classes_)

    model = DiseaseClassifer(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim).to(DEVICE)

    #weights = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.5, 1.0])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=150)

    epochs = args.epochs
    patience = 15
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        "val_acc": []
    }

    # Training
    print("Starting Training")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            v_loss, correct, total = 0, 0, 0
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(DEVICE), batch_y_val.to(DEVICE)

                val_outputs = model(batch_X_val)

                v_loss = criterion(val_outputs, batch_y_val)

                val_loss += v_loss.item()

                _, predicted = torch.max(val_outputs.data, 1)
                total += batch_y_val.size(0)
                correct += (predicted == batch_y_val).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_acc_at_loss = val_acc
                best_acc = val_acc
                best_model_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if args.early_stop and counter >= patience:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

            #if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

        # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss\n')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'go-', label='Validation Accuracy')
    plt.title(f'Validation Accuracy\n')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the figure
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/{args.model_name}_learning_curves.png')
    plt.show()

    if best_model_state:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"models/{args.model_name.replace('.pth', '')}_final_acc_{best_acc_at_loss:.1f}_{timestamp}.pth"
        torch.save(best_model_state, save_path)
        print(f"\nTraining Finished. Best model (Loss: {best_val_loss:.4f}) saved to: {save_path}")
    else:
        print("\nTraining ended without a valid best model state.")

if __name__ == "__main__":
    main()