import torch
import torch.nn as nn
from visuallearn.modelanalyzer import classify_model

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        return self.encoder(x)

class WeirdCustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_bongulator = nn.Identity()
        self.statistical_magic = nn.Parameter(torch.randn(5))

    def forward(self, x):
        return self.token_bongulator(x)


models = {
    "MLP": SimpleMLP(),
    "CNN": SimpleCNN(),
    "Transformer": TinyTransformer(),
    "WeirdCustomModule": WeirdCustomModule()
}

for name, model in models.items():
    model_type = classify_model(model)
    print(f"{name}: classified as {model_type}")
