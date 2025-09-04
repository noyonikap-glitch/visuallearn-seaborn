import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from visuallearn.activationtracker import ActivationTracker
from visuallearn.gradienttracker import GradientTracker

import numpy as np
from visuallearn.combinedplot import CombinedPlotCoordinator

# Dataset
X, y = make_moons(noise=0.3, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()

tracker = ActivationTracker()
tracker.attach_hooks(model)

grad_tracker = GradientTracker()
grad_tracker.attach_hooks(model)


# ⚠️ Do one dummy forward pass to populate tracker
model.eval()
with torch.no_grad():
    _ = model(torch.tensor(X, dtype=torch.float32))  # assuming X is numpy array

# create the combined visualizer
plotter = CombinedPlotCoordinator(X, y, activation_tracker=tracker, gradient_tracker=grad_tracker)
plotter.enable_recording()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()



for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # For loss plot
    y_probs = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).detach().numpy()
    loss_value = log_loss(y, y_probs)

    # Update visualizer
    plotter.update(model, epoch, loss_value)
    #tracker.plot_activations(epoch)
    plotter.export("training.gif", format="gif")
    
