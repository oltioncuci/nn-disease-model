import torch
import torch.nn as nn
import torch.functional as F

# TODO LATER nn.BatchNorm1d()
# TODO LATER nn.Dropout(dropout_rate)


class DiseaseClassifer(nn.Module):
    def __init__(self, input_size,  num_classes, hidden_size=32, dropout_rate=0.2):
        super(DiseaseClassifer, self).__init__()

        self.network = nn.Sequential(
            # LAYER 1
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # LAYER 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),

            # Output Layer
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def get_model(input_dim, output_dim, device):
        model = DiseaseClassifer(input_size=input_dim, num_classes=output_dim)

        return model.to(device)