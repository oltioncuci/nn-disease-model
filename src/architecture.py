import torch
import torch.nn as nn
import torch.functional as F

# TODO LATER nn.BatchNorm1d()
# TODO LATER nn.Dropout(dropout_rate)


class GrowthClassifer(nn.Module):
    def __init__(self, input_size,  num_classes, hidden_size=32, dropout_rate=0.2):
        super(GrowthClassifer, self).__init__()

        self.network = nn.Sequential(
            # LAYER 1
            nn.Linear(input_size, 32),
            nn.ReLU(),

            # LAYER 2
            nn.Linear(32, 16),
            nn.ReLU(),

            # OUTPUT LAYER
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def get_model(input_dim, output_dim, device):
        model = GrowthClassifer(input_size=input_dim, num_classes=output_dim)

        return model.to(device)