import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from visuallearn.framerecorder import FrameRecorder

from visuallearn.activationtracker import ActivationTracker
from visuallearn.gradienttracker import GradientTracker
from visuallearn.visualizer import plot_decision_boundary
from visuallearn.liveplotter import LivePlotter

# Create dataset
X, y = make_moons(noise=0.3, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Model A: Shallow
class MLP_Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.net(x)

# Model B: Deeper
class MLP_Deeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate
modelA = MLP_Shallow()
modelB = MLP_Deeper()
optA = optim.SGD(modelA.parameters(), lr=0.1)
optB = optim.SGD(modelB.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# Setup figure
fig, (axA, axB, axLoss) = plt.subplots(1, 3, figsize=(15, 5))
plotterA = LivePlotter(axLoss, ylabel="Loss (A vs B)")
plotterB = LivePlotter(axLoss, ylabel="Loss (A vs B)")
recorder = FrameRecorder(fig)
plt.ion()
plt.show()

lossesA = []
lossesB = []

for epoch in range(50):
    # Train A
    modelA.train()
    optA.zero_grad()
    outA = modelA(X_train_tensor)
    lossA = loss_fn(outA, y_train_tensor)
    lossA.backward()
    optA.step()

    # Train B
    modelB.train()
    optB.zero_grad()
    outB = modelB(X_train_tensor)
    lossB = loss_fn(outB, y_train_tensor)
    lossB.backward()
    optB.step()

    # Eval loss
    modelA.eval()
    modelB.eval()
    with torch.no_grad():
        y_predA = torch.softmax(modelA(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
        y_predB = torch.softmax(modelB(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
        lossA_val = log_loss(y, y_predA)
        lossB_val = log_loss(y, y_predB)
        lossesA.append(lossA_val)
        lossesB.append(lossB_val)

    # Plot decision boundaries
    axA.clear()
    plot_decision_boundary(modelA, X, y, axA)
    axA.set_title(f"Shallow MLP (Epoch {epoch})")

    axB.clear()
    plot_decision_boundary(modelB, X, y, axB)
    axB.set_title(f"Deeper MLP (Epoch {epoch})")

    # Plot loss curves
    axLoss.clear()
    axLoss.plot(lossesA, label="Shallow")
    axLoss.plot(lossesB, label="Deep")
    axLoss.set_title("Log Loss")
    axLoss.set_xlabel("Epoch")
    axLoss.set_ylabel("Loss")
    axLoss.legend()


    plt.tight_layout()
    plt.pause(0.1)

    recorder.capture()


recorder.export("compare_models.gif", format="gif")
plt.ioff()
plt.show()
