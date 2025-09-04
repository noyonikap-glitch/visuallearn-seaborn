"""
VisualLearn: A Python library for visualizing ML model learning patterns.

This library provides tools to visualize how machine learning models learn,
including decision boundaries, activation distributions, gradient flows,
and training dynamics over time.
"""

__version__ = "0.2.0"
__author__ = "Noyonika Puram"

from .autovisualizer import auto_visualize
from .combinedplot import CombinedPlotCoordinator
from .visualizer import plot_decision_boundary
from .activationtracker import ActivationTracker
from .gradienttracker import GradientTracker
from .liveplotter import LivePlotter
from .cnnvisualizer import CNNFeatureMapVisualizer
from .modelanalyzer import classify_model
from .framerecorder import FrameRecorder
from .regressionvisualizer import RegressionVisualizer, MultiModelRegressionComparator
from .clusteringvisualizer import ClusteringVisualizer, plot_clustering_comparison
from .utils import validate_data, setup_matplotlib_backend, create_grid, safe_tensor_to_numpy

__all__ = [
    'auto_visualize',
    'CombinedPlotCoordinator', 
    'plot_decision_boundary',
    'ActivationTracker',
    'GradientTracker',
    'LivePlotter',
    'CNNFeatureMapVisualizer',
    'classify_model',
    'FrameRecorder',
    'RegressionVisualizer',
    'MultiModelRegressionComparator',
    'ClusteringVisualizer',
    'plot_clustering_comparison',
    'validate_data',
    'setup_matplotlib_backend',
    'create_grid',
    'safe_tensor_to_numpy',
]