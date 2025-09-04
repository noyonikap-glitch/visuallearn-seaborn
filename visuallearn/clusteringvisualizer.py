"""
Clustering visualization components for VisualLearn.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings
from typing import Optional, Union, List, Tuple

class ClusteringVisualizer:
    """Visualizer specifically designed for clustering algorithms."""
    
    def __init__(self, X, model_name: str = "Clustering", true_labels: Optional[np.ndarray] = None):
        """
        Initialize clustering visualizer.
        
        Args:
            X: Input features (2D array)
            model_name: Name for the clustering algorithm
            true_labels: Ground truth labels if available (for supervised evaluation)
        """
        self.X = X
        self.model_name = model_name
        self.true_labels = true_labels
        
        # Prepare data for visualization
        if X.shape[1] > 2:
            self.pca = PCA(n_components=2, random_state=42)
            self.X_viz = self.pca.fit_transform(X)
            warnings.warn(f"Data has {X.shape[1]} features. Using PCA for 2D visualization.")
        else:
            self.X_viz = X
            self.pca = None
            
        # History tracking
        self.cluster_history = []
        self.metrics_history = {'silhouette': [], 'calinski': [], 'davies_bouldin': []}
        self.iterations = []
        
        # Plotting setup
        self.fig = None
        self.axes = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the plotting layout."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        
        self.ax_clusters = self.axes[0, 0]
        self.ax_centers = self.axes[0, 1]
        self.ax_metrics = self.axes[0, 2]
        self.ax_silhouette = self.axes[1, 0]
        self.ax_inertia = self.axes[1, 1]
        self.ax_comparison = self.axes[1, 2]
        
        plt.tight_layout()
        self.initialized = True
        
    def update(self, model, iteration: int = 0, custom_labels: Optional[np.ndarray] = None):
        """
        Update visualization with current clustering state.
        
        Args:
            model: Fitted clustering model
            iteration: Current iteration (for iterative algorithms)
            custom_labels: Custom cluster labels if model doesn't have predict method
        """
        if not self.initialized:
            self.initialize()
            
        # Get cluster assignments
        if custom_labels is not None:
            labels = custom_labels
        elif hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(self.X)
        else:
            warnings.warn("Could not extract cluster labels from model")
            return
            
        # Store history
        self.cluster_history.append(labels.copy())
        self.iterations.append(iteration)
        
        # Calculate metrics
        try:
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                silh_score = silhouette_score(self.X, labels)
                cal_score = calinski_harabasz_score(self.X, labels)
                db_score = davies_bouldin_score(self.X, labels)
                
                self.metrics_history['silhouette'].append(silh_score)
                self.metrics_history['calinski'].append(cal_score) 
                self.metrics_history['davies_bouldin'].append(db_score)
            else:
                # Single cluster case
                self.metrics_history['silhouette'].append(0)
                self.metrics_history['calinski'].append(0)
                self.metrics_history['davies_bouldin'].append(0)
        except Exception as e:
            warnings.warn(f"Could not calculate clustering metrics: {e}")
        
        # Update all plots
        self._update_cluster_plot(labels, model)
        self._update_centers_plot(labels, model)
        self._update_metrics_plot()
        self._update_silhouette_plot(labels)
        self._update_inertia_plot(model)
        self._update_comparison_plot(labels)
        
        # Update title
        n_clusters = len(np.unique(labels))
        silh = self.metrics_history['silhouette'][-1] if self.metrics_history['silhouette'] else 0
        self.fig.suptitle(f'{self.model_name} - Iteration {iteration} | Clusters: {n_clusters} | Silhouette: {silh:.3f}',
                         fontsize=14)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_cluster_plot(self, labels, model):
        """Update main cluster visualization."""
        self.ax_clusters.clear()
        
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
        
        # Plot points colored by cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            self.ax_clusters.scatter(self.X_viz[mask, 0], self.X_viz[mask, 1], 
                                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                                   alpha=0.7, s=50)
        
        # Plot cluster centers if available
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
            if self.pca is not None and centers.shape[1] > 2:
                centers_viz = self.pca.transform(centers)
            else:
                centers_viz = centers
                
            self.ax_clusters.scatter(centers_viz[:, 0], centers_viz[:, 1], 
                                   c='red', marker='X', s=200, 
                                   edgecolors='black', linewidth=2, label='Centers')
        
        self.ax_clusters.set_title('Cluster Assignments')
        self.ax_clusters.set_xlabel('Feature 1' if self.pca is None else 'PC1')
        self.ax_clusters.set_ylabel('Feature 2' if self.pca is None else 'PC2')
        self.ax_clusters.legend()
        self.ax_clusters.grid(True, alpha=0.3)
    
    def _update_centers_plot(self, labels, model):
        """Update cluster centers evolution plot."""
        if not hasattr(model, 'cluster_centers_'):
            self.ax_centers.text(0.5, 0.5, 'No cluster centers\navailable', 
                               ha='center', va='center', transform=self.ax_centers.transAxes)
            return
            
        self.ax_centers.clear()
        centers = model.cluster_centers_
        
        if self.pca is not None and centers.shape[1] > 2:
            centers_viz = self.pca.transform(centers)
        else:
            centers_viz = centers
            
        # Plot current centers
        self.ax_centers.scatter(centers_viz[:, 0], centers_viz[:, 1], 
                              c='red', marker='X', s=200, 
                              edgecolors='black', linewidth=2)
        
        # Add center labels
        for i, (x, y) in enumerate(centers_viz):
            self.ax_centers.annotate(f'C{i}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot data points lightly in background
        self.ax_centers.scatter(self.X_viz[:, 0], self.X_viz[:, 1], 
                              c='lightgray', alpha=0.3, s=20)
        
        self.ax_centers.set_title('Cluster Centers')
        self.ax_centers.set_xlabel('Feature 1' if self.pca is None else 'PC1')
        self.ax_centers.set_ylabel('Feature 2' if self.pca is None else 'PC2')
        self.ax_centers.grid(True, alpha=0.3)
    
    def _update_metrics_plot(self):
        """Update clustering metrics over iterations."""
        self.ax_metrics.clear()
        
        if len(self.metrics_history['silhouette']) == 0:
            return
            
        iterations = list(range(len(self.metrics_history['silhouette'])))
        
        # Plot silhouette score
        self.ax_metrics.plot(iterations, self.metrics_history['silhouette'], 
                           'b-o', label='Silhouette', linewidth=2, markersize=4)
        
        # Plot Davies-Bouldin (lower is better, so invert for visualization)
        if self.metrics_history['davies_bouldin']:
            db_inverted = [1/max(score, 0.01) for score in self.metrics_history['davies_bouldin']]
            db_normalized = np.array(db_inverted) / max(db_inverted) if max(db_inverted) > 0 else db_inverted
            self.ax_metrics.plot(iterations, db_normalized, 'r-s', label='DB Index (inv)', 
                               linewidth=2, markersize=4)
        
        self.ax_metrics.set_xlabel('Iteration')
        self.ax_metrics.set_ylabel('Score')
        self.ax_metrics.set_title('Clustering Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)
    
    def _update_silhouette_plot(self, labels):
        """Update silhouette analysis plot."""
        self.ax_silhouette.clear()
        
        try:
            from sklearn.metrics import silhouette_samples
            
            if len(np.unique(labels)) < 2:
                self.ax_silhouette.text(0.5, 0.5, 'Need ≥2 clusters\nfor silhouette analysis', 
                                      ha='center', va='center', transform=self.ax_silhouette.transAxes)
                return
                
            sample_silhouette_values = silhouette_samples(self.X, labels)
            
            y_lower = 10
            for cluster_id in np.unique(labels):
                cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
                cluster_silhouette_values.sort()
                
                size_cluster = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster
                
                color = plt.cm.Set1(cluster_id / max(np.unique(labels)))
                self.ax_silhouette.fill_betweenx(np.arange(y_lower, y_upper),
                                               0, cluster_silhouette_values,
                                               facecolor=color, edgecolor=color, alpha=0.7)
                
                self.ax_silhouette.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
                y_lower = y_upper + 10
            
            # Average silhouette score line
            avg_silhouette = silhouette_score(self.X, labels)
            self.ax_silhouette.axvline(x=avg_silhouette, color="red", linestyle="--", 
                                     label=f'Average: {avg_silhouette:.3f}')
            
            self.ax_silhouette.set_xlabel('Silhouette Coefficient')
            self.ax_silhouette.set_ylabel('Cluster Label')
            self.ax_silhouette.set_title('Silhouette Analysis')
            self.ax_silhouette.legend()
            
        except ImportError:
            self.ax_silhouette.text(0.5, 0.5, 'Silhouette analysis\nnot available', 
                                  ha='center', va='center', transform=self.ax_silhouette.transAxes)
    
    def _update_inertia_plot(self, model):
        """Update inertia/distortion plot if available."""
        self.ax_inertia.clear()
        
        if hasattr(model, 'inertia_'):
            # For K-means type algorithms
            inertias = [model.inertia_]
            self.ax_inertia.bar(['Current'], inertias, color='skyblue')
            self.ax_inertia.set_title(f'Inertia: {model.inertia_:.2f}')
            self.ax_inertia.set_ylabel('Within-cluster Sum of Squares')
        else:
            # Calculate average within-cluster distance
            if hasattr(model, 'labels_'):
                labels = model.labels_
                total_distance = 0
                n_points = 0
                
                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    cluster_points = self.X[mask]
                    if len(cluster_points) > 1:
                        distances = euclidean_distances(cluster_points)
                        total_distance += np.sum(distances) / 2  # Divide by 2 to avoid double counting
                        n_points += len(cluster_points)
                
                avg_distance = total_distance / max(n_points, 1)
                self.ax_inertia.bar(['Avg Distance'], [avg_distance], color='lightcoral')
                self.ax_inertia.set_title(f'Avg Within-Cluster Distance: {avg_distance:.2f}')
                self.ax_inertia.set_ylabel('Average Distance')
    
    def _update_comparison_plot(self, labels):
        """Update comparison with true labels if available."""
        self.ax_comparison.clear()
        
        if self.true_labels is not None:
            # Create confusion-like matrix for cluster-truth assignment
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            ari = adjusted_rand_score(self.true_labels, labels)
            nmi = normalized_mutual_info_score(self.true_labels, labels)
            
            # Plot true labels vs predicted clusters
            self.ax_comparison.scatter(self.X_viz[:, 0], self.X_viz[:, 1], 
                                     c=self.true_labels, cmap='viridis', alpha=0.7, s=50)
            self.ax_comparison.set_title(f'True Labels\nARI: {ari:.3f} | NMI: {nmi:.3f}')
        else:
            # Show cluster quality metrics
            n_clusters = len(np.unique(labels))
            metrics = []
            values = []
            
            if self.metrics_history['silhouette']:
                metrics.append('Silhouette')
                values.append(self.metrics_history['silhouette'][-1])
            
            if self.metrics_history['davies_bouldin']:
                metrics.append('Davies-Bouldin')  
                values.append(1/max(self.metrics_history['davies_bouldin'][-1], 0.01))
                
            if metrics:
                self.ax_comparison.bar(metrics, values, color=['skyblue', 'lightcoral'][:len(metrics)])
                self.ax_comparison.set_title('Quality Metrics')
                self.ax_comparison.set_ylabel('Score')
            else:
                self.ax_comparison.text(0.5, 0.5, 'No metrics\navailable', 
                                      ha='center', va='center', transform=self.ax_comparison.transAxes)
        
        self.ax_comparison.set_xlabel('Feature 1' if self.pca is None else 'PC1')
        if self.true_labels is not None:
            self.ax_comparison.set_ylabel('Feature 2' if self.pca is None else 'PC2')

def plot_clustering_comparison(X, models_dict: dict, true_labels: Optional[np.ndarray] = None):
    """
    Compare multiple clustering algorithms side by side.
    
    Args:
        X: Input data
        models_dict: Dictionary of {name: fitted_model}
        true_labels: Ground truth labels if available
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Prepare data for visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_viz = pca.fit_transform(X)
    else:
        X_viz = X
        pca = None
    
    for i, (name, model) in enumerate(models_dict.items()):
        # Get cluster assignments
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(X)
        else:
            continue
            
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
        
        # Plot clusters
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            axes[0, i].scatter(X_viz[mask, 0], X_viz[mask, 1], 
                             c=[colors[cluster_id]], alpha=0.7, s=30)
        
        # Plot centers if available
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
            if pca is not None and centers.shape[1] > 2:
                centers_viz = pca.transform(centers)
            else:
                centers_viz = centers
            axes[0, i].scatter(centers_viz[:, 0], centers_viz[:, 1], 
                             c='red', marker='X', s=100, edgecolors='black')
        
        axes[0, i].set_title(f'{name} ({n_clusters} clusters)')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot metrics
        try:
            silh = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
            db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else 0
            
            metrics = ['Silhouette', 'DB (inv)']
            values = [silh, 1/max(db, 0.01)]
            
            axes[1, i].bar(metrics, values, color=['skyblue', 'lightcoral'])
            axes[1, i].set_title(f'{name} Metrics')
            axes[1, i].set_ylim(0, 1)
        except Exception:
            axes[1, i].text(0.5, 0.5, 'Metrics\nunavailable', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    return fig