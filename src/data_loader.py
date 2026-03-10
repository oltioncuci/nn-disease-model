import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted

class DiseaseDataset(Dataset):
    def __init__(self, csv_file, is_train=True, scaler=None, encoder=None):
        df = pd.read_csv(csv_file)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if encoder is None:
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(y)
        else:
            self.encoder = encoder
            
            if not hasattr(self.encoder, 'classes_'):
                y = self.encoder.fit_transform(y)
            else:
                y = self.encoder.transform(y)

        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler

            if not hasattr(self.scaler, 'mean_'):
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def get_data_loader(train_csv, test_csv, batch_size=32):
    train_dataset = DiseaseDataset(train_csv, is_train=True)

    test_dataset = DiseaseDataset(
        test_csv,
        is_train=False,
        scaler=train_dataset.scaler,
        encoder=train_dataset.encoder
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.encoder
