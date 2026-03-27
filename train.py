import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

from src.data_loader import get_data_loaders
from src.architecture import GrowthClassifer

def main():
    parser = argparse.ArgumentParser(description="Mushroom Growth Training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--early-stop", action='store_true')
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    # 1. Load Data
    train_loader, val_loader, _ = get_data_loaders(
        'data/mushrooms/scripts/balanced_mushroom_growth_dataset.csv',
        'data/mushrooms/scripts/balanced_mushroom_growth_dataset.csv',
        args.batch_size
    )

    # 2. Setup Model
    input_dim = train_loader.dataset.X.shape[1]
    output_dim = train_loader.dataset.y.shape[1]
    hidden_dim = 64
    
    model = GrowthClassifer(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')  
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    counter = 0

    print(f"Model initialized for {output_dim} mushroom types.")

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_running_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Use the MASKED loss
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        avg_t_loss = train_running_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        v_loss = 0
        total_abs_error = 0
        total_valid_samples = 0

        with torch.no_grad():
            for batch_X_v, batch_y_v in val_loader:
                batch_X_v, batch_y_v = batch_X_v.to(DEVICE), batch_y_v.to(DEVICE)
                val_outputs = model(batch_X_v)
                
                v_loss += criterion(val_outputs, batch_y_v).item()
                
                # Update MAE stats (ignoring -1.0)
                batch_err = torch.abs(val_outputs - batch_y_v).mean().item()
                total_abs_error += batch_err

        avg_v_loss = v_loss / len(val_loader)
        
        # Calculate MAE as a percentage of growth (0-100)
        avg_mae_pct = (total_abs_error / len(val_loader)) * 100

        scheduler.step(avg_v_loss)

        # 5. Early Stopping & Saving
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), "models/growth_model_best.pth")
            counter = 0
        else:
            counter += 1
            if args.early_stop and counter >= args.patience:
                print(f"Stopping early at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_t_loss:.5f} | Val Loss: {avg_v_loss:.5f} | Error: {avg_mae_pct:.2f}%")
        
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['val_mae'].append(avg_mae_pct)

    # 6. Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Masked MSE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], color='green')
    plt.title('Mean Absolute Error (%)')
    plt.ylabel('Percentage Points Off')
    plt.show()

if __name__ == "__main__":
    main()