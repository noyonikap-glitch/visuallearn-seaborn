"""
Clustering Visualization Example for VisualLearn

Demonstrates visualization of various clustering algorithms including
K-means, DBSCAN, hierarchical clustering, and comparison tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import visuallearn as vl

def kmeans_clustering_example():
    """K-means clustering with live updates."""
    print("🎯 Running K-means Clustering Example...")
    
    # Generate blob data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                           cluster_std=1.5, random_state=42)
    
    # Create visualizer
    visualizer = vl.ClusteringVisualizer(X, "K-means", true_labels=y_true)
    
    # Run K-means with different numbers of clusters
    plt.ion()
    
    for k in [2, 3, 4, 5]:
        print(f"  Testing K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=1)
        
        # Simulate iterative convergence
        for iteration in range(10):
            kmeans.max_iter = iteration + 1
            kmeans.fit(X)
            visualizer.update(kmeans, iteration, custom_labels=kmeans.labels_)
            plt.pause(0.3)
    
    plt.ioff()
    plt.show()
    print("✅ K-means example completed!")

def dbscan_clustering_example():
    """DBSCAN clustering on different datasets."""
    print("🔍 Running DBSCAN Clustering Example...")
    
    # Generate different types of data
    datasets = {
        'Blobs': make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=1.0, random_state=42),
        'Moons': make_moons(n_samples=200, noise=0.15, random_state=42),
        'Circles': make_circles(n_samples=200, noise=0.05, factor=0.3, random_state=42)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, (X, y_true)) in enumerate(datasets.items()):
        print(f"  Clustering {name} dataset...")
        
        # Scale the data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply DBSCAN
        if name == 'Blobs':
            dbscan = DBSCAN(eps=0.3, min_samples=5)
        elif name == 'Moons':
            dbscan = DBSCAN(eps=0.2, min_samples=5)
        else:  # Circles
            dbscan = DBSCAN(eps=0.15, min_samples=5)
        
        dbscan.fit(X_scaled)
        
        # Visualize
        visualizer = vl.ClusteringVisualizer(X_scaled, f"DBSCAN - {name}", true_labels=y_true)
        visualizer.fig, visualizer.axes = fig, axes
        visualizer.ax_clusters = axes[i]
        visualizer.initialized = True
        
        visualizer._update_cluster_plot(dbscan.labels_, dbscan)
        axes[i].set_title(f'DBSCAN - {name}\nClusters: {len(np.unique(dbscan.labels_[dbscan.labels_ != -1]))}')
    
    plt.suptitle('DBSCAN Clustering on Different Datasets', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ DBSCAN example completed!")

def hierarchical_clustering_example():
    """Hierarchical clustering with visualization."""
    print("🌳 Running Hierarchical Clustering Example...")
    
    # Generate data
    X, y_true = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=1.2, random_state=42)
    
    # Different linkage criteria
    linkages = ['ward', 'complete', 'average']
    fig, axes = plt.subplots(1, len(linkages), figsize=(15, 5))
    
    for i, linkage in enumerate(linkages):
        print(f"  Testing {linkage} linkage...")
        
        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        hierarchical.fit(X)
        
        # Visualize
        visualizer = vl.ClusteringVisualizer(X, f"Hierarchical - {linkage}", true_labels=y_true)
        visualizer.fig, visualizer.axes = fig, axes
        visualizer.ax_clusters = axes[i]
        visualizer.initialized = True
        
        visualizer._update_cluster_plot(hierarchical.labels_, hierarchical)
        axes[i].set_title(f'Hierarchical - {linkage.title()}\nClusters: {len(np.unique(hierarchical.labels_))}')
    
    plt.suptitle('Hierarchical Clustering with Different Linkages', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ Hierarchical clustering example completed!")

def gaussian_mixture_example():
    """Gaussian Mixture Model clustering."""
    print("📊 Running Gaussian Mixture Model Example...")
    
    # Generate data with overlapping clusters
    np.random.seed(42)
    X1 = np.random.normal([2, 2], 0.8, (100, 2))
    X2 = np.random.normal([6, 6], 1.2, (100, 2))
    X3 = np.random.normal([4, 2], 0.6, (80, 2))
    X = np.vstack([X1, X2, X3])
    
    # Test different numbers of components
    n_components_list = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, n_comp in enumerate(n_components_list):
        print(f"  Testing {n_comp} components...")
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_comp, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        # Visualize
        visualizer = vl.ClusteringVisualizer(X, f"GMM - {n_comp} components")
        visualizer.fig, visualizer.axes = fig, axes
        visualizer.ax_clusters = axes[i]
        visualizer.initialized = True
        
        visualizer._update_cluster_plot(labels, gmm)
        
        # Add ellipses for Gaussian components
        from matplotlib.patches import Ellipse
        for j in range(n_comp):
            mean = gmm.means_[j]
            cov = gmm.covariances_[j]
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * 2 * np.sqrt(eigenvals)  # 2 sigma ellipse
            
            ellipse = Ellipse(mean, width, height, angle=angle, alpha=0.3, 
                            facecolor='none', edgecolor='red', linewidth=2)
            axes[i].add_patch(ellipse)
        
        axes[i].set_title(f'GMM - {n_comp} Components\nBIC: {gmm.bic(X):.1f}')
    
    plt.suptitle('Gaussian Mixture Model Clustering', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ Gaussian Mixture example completed!")

def clustering_comparison_example():
    """Compare multiple clustering algorithms side by side."""
    print("⚖️ Running Clustering Algorithm Comparison...")
    
    # Generate challenging dataset (two moons)
    X, y_true = make_moons(n_samples=200, noise=0.15, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Define clustering algorithms
    clustering_algorithms = {
        'K-means': KMeans(n_clusters=2, random_state=42),
        'DBSCAN': DBSCAN(eps=0.25, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=2),
        'GMM': GaussianMixture(n_components=2, random_state=42)
    }
    
    # Fit all models
    fitted_models = {}
    for name, model in clustering_algorithms.items():
        print(f"  Fitting {name}...")
        model.fit(X_scaled)
        fitted_models[name] = model
    
    # Create comparison plot
    comparison_fig = vl.plot_clustering_comparison(X_scaled, fitted_models, y_true)
    comparison_fig.suptitle('Clustering Algorithm Comparison - Two Moons Dataset', fontsize=16)
    plt.show()
    
    print("✅ Clustering comparison completed!")

def iris_clustering_example():
    """Clustering on the Iris dataset with ground truth comparison."""
    print("🌸 Running Iris Clustering Example...")
    
    # Load Iris dataset
    iris = load_iris()
    X, y_true = iris.data, iris.target
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_pca)
    
    # Create comprehensive visualization
    visualizer = vl.ClusteringVisualizer(X_pca, "K-means on Iris", true_labels=y_true)
    visualizer.update(kmeans, iteration=0)
    
    plt.show()
    print("✅ Iris clustering example completed!")

if __name__ == "__main__":
    print("🎨 VisualLearn Clustering Examples")
    print("=" * 40)
    
    # Run all examples
    kmeans_clustering_example()
    print()
    
    dbscan_clustering_example()
    print()
    
    hierarchical_clustering_example()
    print()
    
    gaussian_mixture_example()
    print()
    
    clustering_comparison_example()
    print()
    
    iris_clustering_example()
    
    print("\n🎉 All clustering examples completed!")
    print("Features demonstrated:")
    print("- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical, GMM)")
    print("- Live clustering updates and iterations")
    print("- Cluster quality metrics (Silhouette, Davies-Bouldin)")
    print("- Comparison with ground truth labels")
    print("- Side-by-side algorithm comparison")
    print("- High-dimensional data visualization with PCA")
    print("- Different dataset types (blobs, moons, circles)")