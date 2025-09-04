"""
Utility functions for VisualLearn library.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

def validate_data(X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate and preprocess input data.
    
    Args:
        X: Input features array
        y: Target labels array (optional)
        
    Returns:
        Tuple of validated X and y arrays
        
    Raises:
        ValueError: If data validation fails
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    if X.shape[1] != 2:
        warnings.warn("Visualization works best with 2D data. Higher dimensions will be ignored.")
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    return X, y

def setup_matplotlib_backend():
    """Setup matplotlib for interactive plotting."""
    try:
        plt.ion()
        return True
    except Exception as e:
        warnings.warn(f"Could not enable interactive mode: {e}")
        return False

def create_grid(X: np.ndarray, resolution: float = 0.01, padding: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a meshgrid for decision boundary plotting.
    
    Args:
        X: Input data array
        resolution: Grid resolution
        padding: Padding around data bounds
        
    Returns:
        Tuple of (xx, yy, grid) arrays
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

def safe_tensor_to_numpy(tensor) -> np.ndarray:
    """
    Safely convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        Numpy array
    """
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        return tensor.numpy()
    else:
        return np.array(tensor)