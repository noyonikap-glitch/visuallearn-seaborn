"""
Multi-class Classification Example for VisualLearn

Demonstrates visualization of models with 3+ classes using iris dataset
and other multi-class datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import visuallearn as vl

def iris_multiclass_example():
    """Example with Iris dataset (3 classes)."""
    print("🌸 Running Iris Multi-class Classification Example...")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # Use only first 2 features for better visualization
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Train different models
    models = {
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    }
    
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Visualize decision boundary
        vl.plot_decision_boundary(model, X_scaled, y, axes[i], task_type='classification')
        axes[i].set_title(f'{name}\nAccuracy: {model.score(X_test_scaled, y_test):.3f}')
        
        # Add feature names
        axes[i].set_xlabel(iris.feature_names[0])
        axes[i].set_ylabel(iris.feature_names[1])
    
    plt.suptitle('Multi-class Classification: Iris Dataset (3 classes)', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ Iris example completed!")

def synthetic_multiclass_example():
    """Example with synthetic multi-class dataset."""
    print("🎯 Running Synthetic Multi-class Classification Example...")
    
    # Generate synthetic multi-class data
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                              n_redundant=0, n_clusters_per_class=1, n_classes=4, 
                              random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    vl.plot_decision_boundary(model, X, y, ax, task_type='classification')
    ax.set_title('Synthetic Multi-class Classification (4 classes)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    plt.show()
    print("✅ Synthetic multi-class example completed!")

def high_dimensional_multiclass_example():
    """Example with high-dimensional data (demonstrates PCA reduction)."""
    print("📊 Running High-dimensional Multi-class Example...")
    
    # Generate high-dimensional data
    X, y = make_classification(n_samples=300, n_features=10, n_informative=8,
                              n_redundant=2, n_clusters_per_class=1, n_classes=3,
                              random_state=42)
    
    # Train model
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X, y)
    
    # Visualize (will automatically use PCA for dimensionality reduction)
    fig, ax = plt.subplots(figsize=(10, 8))
    vl.plot_decision_boundary(model, X, y, ax, task_type='classification')
    ax.set_title('High-dimensional Multi-class (10D → 2D via PCA)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    plt.show()
    print("✅ High-dimensional example completed!")

def blob_multiclass_example():
    """Example with blob-like clusters."""
    print("🫧 Running Blob Multi-class Classification Example...")
    
    # Generate blob data
    X, y = make_blobs(n_samples=400, centers=5, n_features=2, 
                     cluster_std=1.2, random_state=42)
    
    # Train multiple models for comparison
    models = {
        'K-Nearest Neighbors': MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42),
        'SVM Linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"  Training {name}...")
        model.fit(X, y)
        
        # Visualize decision boundary
        vl.plot_decision_boundary(model, X, y, axes[i], task_type='classification')
        axes[i].set_title(f'{name}\nAccuracy: {model.score(X, y):.3f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.suptitle('Multi-class Classification: Blob Dataset (5 classes)', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ Blob example completed!")

if __name__ == "__main__":
    print("🎨 VisualLearn Multi-class Classification Examples")
    print("=" * 50)
    
    # Run all examples
    iris_multiclass_example()
    print()
    
    synthetic_multiclass_example()
    print()
    
    high_dimensional_multiclass_example()
    print()
    
    blob_multiclass_example()
    
    print("\n🎉 All multi-class classification examples completed!")
    print("Notice how the library automatically:")
    print("- Detects the number of classes")
    print("- Uses appropriate colormaps")
    print("- Applies PCA for high-dimensional data")
    print("- Shows class boundaries clearly")