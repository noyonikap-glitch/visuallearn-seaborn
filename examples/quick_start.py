"""
Quick Start Example for VisualLearn

This example demonstrates the basic usage of the visuallearn library
for visualizing machine learning model training in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import visuallearn
import visuallearn as vl

def sklearn_example():
    """Example using scikit-learn models."""
    print("🚀 Running scikit-learn example...")
    
    # Generate sample data
    X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train model
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Visualize decision boundary
    fig, ax = plt.subplots(figsize=(8, 6))
    vl.plot_decision_boundary(model, X, y, ax)
    plt.title("SVM Decision Boundary")
    plt.show()
    
    print("✅ scikit-learn example completed!")

def pytorch_example():
    """Example using PyTorch models."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.metrics import log_loss
        
        print("🚀 Running PyTorch example...")
        
        # Generate data
        X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
        X = StandardScaler().fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Define model
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = SimpleNN()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Set up visualization
        activation_tracker = vl.ActivationTracker()
        activation_tracker.attach_hooks(model)
        
        gradient_tracker = vl.GradientTracker()
        gradient_tracker.attach_hooks(model)
        
        # Initialize with dummy forward pass
        model.eval()
        with torch.no_grad():
            _ = model(X_tensor)
        
        # Create visualizer
        visualizer = vl.CombinedPlotCoordinator(
            X, y, 
            activation_tracker=activation_tracker,
            gradient_tracker=gradient_tracker
        )
        visualizer.enable_recording()
        
        # Training loop with visualization
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            # Calculate loss for visualization
            model.eval()
            with torch.no_grad():
                y_probs = torch.softmax(model(X_tensor), dim=1).numpy()
                loss_value = log_loss(y, y_probs)
            model.train()
            
            # Update visualization every 5 epochs
            if epoch % 5 == 0:
                visualizer.update(model, epoch, loss_value)
                print(f"Epoch {epoch:3d}: Loss = {loss_value:.4f}")
        
        # Export training animation
        visualizer.export("pytorch_training.gif", format="gif", fps=2)
        print("✅ PyTorch example completed! Check pytorch_training.gif")
        
    except ImportError:
        print("⚠️  PyTorch not installed. Install with: pip install torch")

def auto_visualize_example():
    """Example using the auto_visualize function."""
    print("🚀 Running auto-visualize example...")
    
    # Generate data
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Auto-visualize
    try:
        visualizer = vl.auto_visualize(model, X, y)
        print("✅ Auto-visualize example completed!")
    except Exception as e:
        print(f"⚠️  Auto-visualize not fully implemented: {e}")

if __name__ == "__main__":
    print("🧠 VisualLearn Quick Start Examples")
    print("=" * 40)
    
    # Run examples
    sklearn_example()
    print()
    
    pytorch_example()
    print()
    
    auto_visualize_example()
    
    print("\n🎉 All examples completed!")
    print("Check the generated GIF files to see your model learning!")