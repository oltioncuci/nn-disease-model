import torch
import torch.nn as nn
import torch.functional as F

# TODO LATER nn.BatchNorm1d()
# TODO LATER nn.Dropout(dropout_rate)


class GrowthRegressor(nn.Module):
    def __init__(self, input_size,  num_classes, hidden_size=32, dropout_rate=0.2):
        super(GrowthRegressor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2), 
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.net(x)
        return x

    def get_model(input_dim, output_dim, device):
        model = GrowthRegressor(input_size=input_dim, num_classes=output_dim)

        return model.to(device)