import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_data_loader
from src.architecture import DiseaseClassifer

import argparse

def main():
    parser = argparse.ArgumentParser(description="Training Arguments")

    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float,  default=0.001, help="Learning rate")
    parser.add_argument("--model-name", type=str, default='best_disease_model.pth', help="Saved Model Name")
    # TODO LATER add Dataset

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, encoder = get_data_loader(
        'data/apple/apple_disease_training_data.csv', 'data/apple/apple_disease_test_data.csv', args.batch_size
    )

    input_dim = train_loader.dataset.X.shape[1]
    hidden_dim = 32
    output_dim = len(encoder.classes_)

    model = DiseaseClassifer(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    epochs = args.epoch
    patience = 15
    best_val_loss = float('inf')
    counter = 0

    # Training
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                save_path = f"models/{args.model_name.replace('.pth', '')}_acc_{val_acc:.1f}.pth"

                torch.save(model.state_dict(), save_path)
                print(f"--> Saved better model to: {save_path}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")