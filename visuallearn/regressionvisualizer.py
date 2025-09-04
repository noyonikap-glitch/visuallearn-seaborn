"""
Regression visualization components for VisualLearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Optional, Union, Tuple, List

class RegressionVisualizer:
    """Visualizer specifically designed for regression tasks."""
    
    def __init__(self, X, y, model_name: str = "Model"):
        """
        Initialize regression visualizer.
        
        Args:
            X: Input features (2D array)
            y: Target values (1D array)
            model_name: Name for the model being visualized
        """
        self.X = X
        self.y = y
        self.model_name = model_name
        self.predictions_history = []
        self.metrics_history = {'mse': [], 'mae': [], 'r2': []}
        self.epochs = []
        
        # Initialize plotting
        self.fig = None
        self.axes = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the plotting layout."""
        if self.X.shape[1] == 1:
            # 1D regression - show curve fitting
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
            self.ax_fit = self.axes[0, 0]
            self.ax_residuals = self.axes[0, 1] 
            self.ax_metrics = self.axes[1, 0]
            self.ax_pred_vs_true = self.axes[1, 1]
        else:
            # Multi-dimensional - show surface + metrics
            self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
            self.ax_surface = self.axes[0, 0]
            self.ax_residuals = self.axes[0, 1]
            self.ax_feature_importance = self.axes[0, 2]
            self.ax_metrics = self.axes[1, 0] 
            self.ax_pred_vs_true = self.axes[1, 1]
            self.ax_learning_curve = self.axes[1, 2]
        
        plt.tight_layout()
        self.initialized = True
        
    def update(self, model, epoch: int, y_pred: Optional[np.ndarray] = None):
        """
        Update visualization with current model state.
        
        Args:
            model: Trained regression model
            epoch: Current epoch/iteration
            y_pred: Predictions (if None, will be computed)
        """
        if not self.initialized:
            self.initialize()
            
        # Get predictions
        if y_pred is None:
            try:
                y_pred = model.predict(self.X)
            except Exception as e:
                warnings.warn(f"Could not get predictions: {e}")
                return
        
        # Store history
        self.predictions_history.append(y_pred.copy())
        self.epochs.append(epoch)
        
        # Calculate metrics
        mse = mean_squared_error(self.y, y_pred)
        mae = mean_absolute_error(self.y, y_pred) 
        r2 = r2_score(self.y, y_pred)
        
        self.metrics_history['mse'].append(mse)
        self.metrics_history['mae'].append(mae)
        self.metrics_history['r2'].append(r2)
        
        # Update plots
        self._update_fit_plot(model, y_pred)
        self._update_residuals_plot(y_pred)
        self._update_metrics_plot()
        self._update_pred_vs_true_plot(y_pred)
        
        if hasattr(self, 'ax_feature_importance'):
            self._update_feature_importance(model)
        if hasattr(self, 'ax_learning_curve'):
            self._update_learning_curve()
            
        # Update title
        self.fig.suptitle(f'{self.model_name} - Epoch {epoch} | MSE: {mse:.4f} | R²: {r2:.4f}', 
                         fontsize=14)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_fit_plot(self, model, y_pred):
        """Update the main fit visualization."""
        if self.X.shape[1] == 1:
            # 1D regression line
            self.ax_fit.clear()
            
            # Sort for smooth line plotting
            sort_idx = np.argsort(self.X[:, 0])
            self.ax_fit.scatter(self.X[:, 0], self.y, alpha=0.6, label='True Data')
            self.ax_fit.plot(self.X[sort_idx, 0], y_pred[sort_idx], 'r-', linewidth=2, label='Prediction')
            
            self.ax_fit.set_xlabel('X')
            self.ax_fit.set_ylabel('y')
            self.ax_fit.set_title('Regression Fit')
            self.ax_fit.legend()
            self.ax_fit.grid(True, alpha=0.3)
            
        else:
            # Multi-dimensional - show surface or feature plot
            if self.X.shape[1] == 2:
                from .visualizer import _plot_regression_surface
                _plot_regression_surface(model, self.X, self.X, self.y, self.ax_surface)
                self.ax_surface.set_title('Prediction Surface')
            else:
                # Show prediction vs first feature
                self.ax_surface.clear()
                self.ax_surface.scatter(self.X[:, 0], self.y, alpha=0.6, label='True')
                self.ax_surface.scatter(self.X[:, 0], y_pred, alpha=0.6, label='Predicted')
                self.ax_surface.set_xlabel('Feature 1')
                self.ax_surface.set_ylabel('Target')
                self.ax_surface.set_title('Predictions vs Feature 1')
                self.ax_surface.legend()
    
    def _update_residuals_plot(self, y_pred):
        """Update residuals plot."""
        self.ax_residuals.clear()
        residuals = self.y - y_pred
        
        self.ax_residuals.scatter(y_pred, residuals, alpha=0.6)
        self.ax_residuals.axhline(y=0, color='r', linestyle='--')
        self.ax_residuals.set_xlabel('Predicted Values')
        self.ax_residuals.set_ylabel('Residuals')
        self.ax_residuals.set_title('Residuals Plot')
        self.ax_residuals.grid(True, alpha=0.3)
    
    def _update_metrics_plot(self):
        """Update metrics over time."""
        self.ax_metrics.clear()
        
        epochs = list(range(len(self.metrics_history['mse'])))
        self.ax_metrics.plot(epochs, self.metrics_history['mse'], 'b-', label='MSE')
        self.ax_metrics.plot(epochs, self.metrics_history['mae'], 'g-', label='MAE')
        
        self.ax_metrics.set_xlabel('Epoch')
        self.ax_metrics.set_ylabel('Error')
        self.ax_metrics.set_title('Training Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)
        
        # Add secondary y-axis for R²
        ax2 = self.ax_metrics.twinx()
        ax2.plot(epochs, self.metrics_history['r2'], 'r-', label='R²')
        ax2.set_ylabel('R² Score')
        ax2.legend(loc='upper right')
    
    def _update_pred_vs_true_plot(self, y_pred):
        """Update predicted vs true values scatter plot."""
        self.ax_pred_vs_true.clear()
        
        self.ax_pred_vs_true.scatter(self.y, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val, max_val = min(self.y.min(), y_pred.min()), max(self.y.max(), y_pred.max())
        self.ax_pred_vs_true.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        self.ax_pred_vs_true.set_xlabel('True Values')
        self.ax_pred_vs_true.set_ylabel('Predicted Values')
        self.ax_pred_vs_true.set_title('Predicted vs True')
        self.ax_pred_vs_true.grid(True, alpha=0.3)
    
    def _update_feature_importance(self, model):
        """Update feature importance plot if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return
                
            self.ax_feature_importance.clear()
            features = [f'Feature {i+1}' for i in range(len(importance))]
            self.ax_feature_importance.barh(features, importance)
            self.ax_feature_importance.set_title('Feature Importance')
            self.ax_feature_importance.set_xlabel('Importance')
            
        except Exception:
            pass
    
    def _update_learning_curve(self):
        """Update learning curve visualization."""
        if len(self.metrics_history['mse']) < 2:
            return
            
        self.ax_learning_curve.clear()
        epochs = list(range(len(self.metrics_history['mse'])))
        
        self.ax_learning_curve.plot(epochs, self.metrics_history['mse'], 'b-', linewidth=2, label='MSE')
        self.ax_learning_curve.fill_between(epochs, self.metrics_history['mse'], alpha=0.3)
        
        self.ax_learning_curve.set_xlabel('Epoch')
        self.ax_learning_curve.set_ylabel('MSE')
        self.ax_learning_curve.set_title('Learning Curve')
        self.ax_learning_curve.grid(True, alpha=0.3)
        self.ax_learning_curve.legend()

class MultiModelRegressionComparator:
    """Compare multiple regression models side by side."""
    
    def __init__(self, X, y, model_names: List[str]):
        """
        Initialize multi-model comparator.
        
        Args:
            X: Input features
            y: Target values  
            model_names: List of model names to compare
        """
        self.X = X
        self.y = y
        self.model_names = model_names
        self.visualizers = {}
        
        # Initialize individual visualizers
        for name in model_names:
            self.visualizers[name] = RegressionVisualizer(X, y, name)
    
    def update_model(self, model_name: str, model, epoch: int):
        """Update specific model visualization."""
        if model_name in self.visualizers:
            self.visualizers[model_name].update(model, epoch)
    
    def create_comparison_plot(self):
        """Create a comparison plot of all models."""
        n_models = len(self.model_names)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, name in enumerate(self.model_names):
            viz = self.visualizers[name]
            
            if len(viz.predictions_history) > 0:
                y_pred = viz.predictions_history[-1]
                
                # Fit plot
                if self.X.shape[1] == 1:
                    sort_idx = np.argsort(self.X[:, 0])
                    axes[0, i].scatter(self.X[:, 0], self.y, alpha=0.6, label='True')
                    axes[0, i].plot(self.X[sort_idx, 0], y_pred[sort_idx], 'r-', label='Pred')
                else:
                    axes[0, i].scatter(self.y, y_pred, alpha=0.6)
                    min_val, max_val = min(self.y.min(), y_pred.min()), max(self.y.max(), y_pred.max())
                    axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--')
                
                axes[0, i].set_title(f'{name} - Fit')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # Metrics plot
                epochs = list(range(len(viz.metrics_history['mse'])))
                axes[1, i].plot(epochs, viz.metrics_history['mse'], 'b-', label='MSE')
                axes[1, i].plot(epochs, viz.metrics_history['r2'], 'g-', label='R²')
                axes[1, i].set_title(f'{name} - Metrics')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig