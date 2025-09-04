import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from typing import Optional, Union, Tuple
import warnings

def plot_decision_boundary(model, X, y, ax, resolution=0.01, task_type='auto', **kwargs):
    """
    Enhanced decision boundary plotting supporting multiple task types.
    
    Args:
        model: Trained model (sklearn or PyTorch)
        X: Input features (2D array)
        y: Target values (1D array) 
        ax: Matplotlib axis
        resolution: Grid resolution for boundary
        task_type: 'classification', 'regression', 'clustering', or 'auto'
        **kwargs: Additional plotting arguments
    """
    # Ensure 2D data for visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X)
        warnings.warn(f"Data has {X.shape[1]} features. Using PCA to reduce to 2D for visualization.")
    else:
        X_plot = X
    
    # Auto-detect task type
    if task_type == 'auto':
        task_type = _detect_task_type(model, y)
    
    # Create grid
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform grid if PCA was used
    if X.shape[1] > 2:
        grid_original = pca.inverse_transform(grid)
    else:
        grid_original = grid

    ax.clear()
    
    if task_type == 'classification':
        _plot_classification_boundary(model, grid_original, xx, yy, X_plot, y, ax, **kwargs)
    elif task_type == 'regression':
        _plot_regression_surface(model, grid_original, xx, yy, X_plot, y, ax, **kwargs)
    elif task_type == 'clustering':
        _plot_clustering_boundary(model, grid_original, xx, yy, X_plot, y, ax, **kwargs)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

def _detect_task_type(model, y) -> str:
    """Auto-detect the task type based on model and labels."""
    if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
        return 'classification'
    elif hasattr(model, 'cluster_centers_') or hasattr(model, 'labels_'):
        return 'clustering'
    elif y is not None and len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer):
        return 'classification'
    else:
        return 'regression'

def _plot_classification_boundary(model, grid, xx, yy, X, y, ax, **kwargs):
    """Plot decision boundary for classification tasks."""
    # Get predictions
    try:
        Z = model.predict(grid)
    except AttributeError:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(grid, dtype=torch.float32)
            logits = model(inputs)
            Z = torch.argmax(logits, dim=1).numpy()
    
    Z = Z.reshape(xx.shape)
    n_classes = len(np.unique(y)) if y is not None else len(np.unique(Z))
    
    # Enhanced colormap for multi-class
    if n_classes <= 2:
        colors = ['lightcoral', 'lightblue']
    elif n_classes <= 3:
        colors = ['lightcoral', 'lightblue', 'lightgreen']
    elif n_classes <= 10:
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    cmap = ListedColormap(colors[:n_classes])
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(n_classes+1)-0.5)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='black', s=50)
    ax.set_title(f"Decision Boundary ({n_classes} classes)")
    
    # Add colorbar for multi-class
    if n_classes > 2:
        plt.colorbar(scatter, ax=ax, label='Class')

def _plot_regression_surface(model, grid, xx, yy, X, y, ax, **kwargs):
    """Plot prediction surface for regression tasks."""
    try:
        Z = model.predict(grid)
    except AttributeError:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(grid, dtype=torch.float32)
            Z = model(inputs).squeeze().numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot prediction surface
    contour = ax.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Prediction')
    
    # Plot data points
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=50)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='red', edgecolors='black', s=50)
    
    ax.set_title("Regression Surface")

def _plot_clustering_boundary(model, grid, xx, yy, X, y, ax, **kwargs):
    """Plot cluster assignments for clustering tasks."""
    try:
        if hasattr(model, 'predict'):
            Z = model.predict(grid)
        elif hasattr(model, 'labels_'):
            # For fitted clustering models, use cluster centers to assign
            from sklearn.metrics.pairwise import euclidean_distances
            centers = model.cluster_centers_
            distances = euclidean_distances(grid, centers)
            Z = np.argmin(distances, axis=1)
        else:
            raise AttributeError("Model doesn't support clustering prediction")
    except AttributeError as e:
        warnings.warn(f"Could not plot clustering boundary: {e}")
        return
    
    Z = Z.reshape(xx.shape)
    n_clusters = len(np.unique(Z))
    
    # Use distinct colors for clusters
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    cmap = ListedColormap(colors[:n_clusters])
    
    # Plot cluster regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(n_clusters+1)-0.5)
    
    # Plot data points
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='black', s=50)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='red', edgecolors='black', s=50)
    
    # Plot cluster centers if available
    if hasattr(model, 'cluster_centers_'):
        centers = model.cluster_centers_
        if centers.shape[1] == 2:  # Only if 2D
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    ax.set_title(f"Cluster Assignments ({n_clusters} clusters)")

