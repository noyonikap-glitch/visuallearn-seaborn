# VisualLearn

A lightweight, intuitive tool for **visualizing how machine learning models learn — in real time.**  
Perfect for **education**, **debugging**, **content creation**, or just understanding the magic under the hood.

---

## ✨ What It Does

- ✅ Visualizes **decision boundaries** as they evolve
- 📉 Plots **loss curves** in real time
- 🔬 Tracks **activation distributions** per layer
- 🔁 Shows **gradient histograms** during backpropagation
- 🎞️ Exports training as **GIF or MP4**

---

## 📦 Currently Supports

- 🧠 **PyTorch** (MLPs with `nn.Linear`, `nn.ReLU`)
- 🔢 **scikit-learn** classifiers with `.fit()` and `.predict()`
- 📐 2D input datasets (e.g., `make_moons`, `make_circles`)
- 🆚 Side-by-side comparison of **multiple models** learning the same dataset
- 🎨 Manual or modular plotting (works with any `matplotlib` figure or axes)

---

## 🚀 Future Plans

- 🖼️ Visualizing **CNN feature maps** and filters
- 🔗 Support for **Transformers, RNNs, and custom activations**
- 🧱 Visualizing **weight updates** and **dead neurons**
- 📁 Integration with **experiment tracking** tools (e.g., Weights & Biases, TensorBoard)
- 🎯 CLI and web export (`mlvis run demo --gif`)

---

## 🎓 Use Cases

- 🏫 Teaching machine learning concepts visually
- 📊 Debugging model behavior, layer by layer
- 🎞️ Creating demos for presentations, lectures, or social media
- 🧪 Comparing architectures (e.g., shallow vs deep MLPs)
- 🔬 Observing **gradient flow**, **activation saturation**, and more

---

## 🛠️ Installation

### Basic Installation
```bash
pip install visuallearn
```

### With Optional Dependencies
```bash
# For PyTorch support
pip install visuallearn[pytorch]

# For video export (MP4)
pip install visuallearn[video]

# For development
pip install visuallearn[dev]

# Everything included
pip install visuallearn[all]
```

### From Source
```bash
git clone https://github.com/noyonikap-glitch/visuallearn.git
cd visuallearn
pip install -e .
```

## 🚀 Quick Start

```python
import visuallearn as vl
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X = StandardScaler().fit_transform(X)

# Train model
model = SVC(kernel='rbf', probability=True)
model.fit(X, y)

# Visualize decision boundary
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
vl.plot_decision_boundary(model, X, y, ax)
plt.show()
```

### PyTorch Real-time Training Visualization

```python
import torch
import torch.nn as nn
import visuallearn as vl

# Your PyTorch model
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(), 
    nn.Linear(8, 2)
)

# Set up tracking
activation_tracker = vl.ActivationTracker()
activation_tracker.attach_hooks(model)

# Create visualizer
visualizer = vl.CombinedPlotCoordinator(X, y, activation_tracker=activation_tracker)
visualizer.enable_recording()

# Training loop with live updates
for epoch in range(100):
    # ... your training code ...
    visualizer.update(model, epoch, loss_value)

# Export as GIF
visualizer.export("training.gif", format="gif")
```

