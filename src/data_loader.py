import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted

import joblib

class GrowthDataset(Dataset):
    def __init__(self, csv_file, is_train=True, scaler=None, encoder=None):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file.copy()  # Use the dataframe directly
        else:
            df = pd.read_csv(csv_file)

        #print(df)

        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2:].values

        if is_train:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)
            
        print(X[:50])
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def get_data_loaders(train_csv, test_csv, batch_size=32, val_split=0.2):
    df = pd.read_csv(train_csv).sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df) * val_split)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    # print("val_size: ", val_size)
    # print("val_df: ", val_df)
    # print("train_df: ", train_df)

    train_dataset = GrowthDataset(train_df, is_train=True)
    
    val_dataset = GrowthDataset(val_df, is_train=False, 
                                    scaler=train_dataset.scaler)
    
    test_dataset = GrowthDataset(test_csv, is_train=False, 
                                    scaler=train_dataset.scaler)
    
    joblib.dump(train_dataset.scaler, 'utils/scaler.joblib')
    #joblib.dump(train_dataset.encoder, 'utils/encoder.joblib')

    #print("train_dataset: ",train_dataset)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(
        '../data/mushrooms/scripts/mushroom_growth_dataset.csv', '../data/mushrooms/scripts/mushroom_growth_dataset_test.csv', 32
    )