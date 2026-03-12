import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted

import joblib

class DiseaseDataset(Dataset):
    def __init__(self, csv_file, is_train=True, scaler=None, encoder=None):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file.copy()  # Use the dataframe directly
        else:
            df = pd.read_csv(csv_file)

        #print(df)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if is_train:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(y)
        else:
            # Use provided scaler/encoder or it will crash/fail
            self.scaler = scaler
            self.encoder = encoder
            X = self.scaler.transform(X)
            y = self.encoder.transform(y)


        # if encoder is None:
        #     self.encoder = LabelEncoder()
        #     y = self.encoder.fit_transform(y)
        # else:
        #     self.encoder = encoder
            
        #     if not hasattr(self.encoder, 'classes_'):
        #         y = self.encoder.fit_transform(y)
        #     else:
        #         y = self.encoder.transform(y)

        # if scaler is None:
        #     self.scaler = StandardScaler()
        #     X = self.scaler.fit_transform(X)
        # else:
        #     self.scaler = scaler

        #     if not hasattr(self.scaler, 'mean_'):
        #         X = self.scaler.fit_transform(X)
        #     else:
        #         X = self.scaler.transform(X)
            

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def get_data_loaders(train_csv, test_csv, batch_size=32, val_split=0.2):
    df_train = pd.read_csv(train_csv)
    #df_test = pd.read_csv(test_csv)
    
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df_train) * val_split)

    val_df = df_train.iloc[:val_size]
    train_df = df_train.iloc[val_size:]

    train_dataset = DiseaseDataset(train_df, is_train=True)
    
    val_dataset = DiseaseDataset(val_df, is_train=False, 
                                    scaler=train_dataset.scaler, 
                                    encoder=train_dataset.encoder)
    
    test_dataset = DiseaseDataset(test_csv, is_train=False, 
                                    scaler=train_dataset.scaler, 
                                    encoder=train_dataset.encoder)
    
    joblib.dump(train_dataset.scaler, 'utils/scaler.joblib')
    joblib.dump(train_dataset.encoder, 'utils/encoder.joblib')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.encoder


if __name__ == "__main__":
    train_loader, val_loader, test_loader, encoder = get_data_loaders(
        'data/apple/apple_disease_realistic_data.csv', 'data/apple/apple_disease_realistic_data_test.csv', 32
    )