import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

from src.data_loader import get_data_loaders
from src.architecture import GrowthRegressor

def main():
    parser = argparse.ArgumentParser(description="Mushroom Growth Training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--early-stop", action='store_true', default=True)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    train_loader, val_loader, _ = get_data_loaders(
        'data/growth_data.csv',
        batch_size=args.batch_size
    )

    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    output_dim = sample_y.shape[1]
    
    model = GrowthRegressor(input_size=input_dim, hidden_size=64, num_classes=output_dim).to(DEVICE)
    
    #criterion = nn.MSELoss()  
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    counter = 0

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_t_loss = train_running_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        v_loss = 0.0
        total_mae = 0.0

        with torch.no_grad():
            for batch_X_v, batch_y_v in val_loader:
                batch_X_v, batch_y_v = batch_X_v.to(DEVICE), batch_y_v.to(DEVICE)
                val_outputs = model(batch_X_v)
                
                v_loss += criterion(val_outputs, batch_y_v).item()
                
                total_mae += torch.abs(val_outputs - batch_y_v).mean().item() * 100

        avg_v_loss = v_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)

        scheduler.step(avg_v_loss)

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_v_loss,
            }, "models/growth_model_best_v6.pth")
            counter = 0
            status = "Saved"
        else:
            counter += 1
            status = f"Patience: {counter}/{args.patience}"

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_t_loss:.5f} | Val Loss: {avg_v_loss:.5f} | MAE: {avg_mae:.2f}% | {status} | LR: {current_lr}")
        
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['val_mae'].append(avg_mae)

        if args.early_stop and counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 7. Visualization
    _plot_results(history)

def _plot_results(history):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train MSE', color='royalblue')
    ax1.plot(history['val_loss'], label='Val MSE', color='orange')
    ax1.set_title('Learning Curves (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history['val_mae'], color='forestgreen')
    ax2.set_title('Validation MAE (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage Points Off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()