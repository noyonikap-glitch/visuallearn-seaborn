"""
Regression Visualization Example for VisualLearn

Demonstrates visualization of regression models including 1D curve fitting,
2D surface regression, and multi-dimensional regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import visuallearn as vl

def simple_1d_regression():
    """Simple 1D regression with curve fitting visualization."""
    print("📈 Running 1D Regression Example...")
    
    # Generate 1D regression data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X.squeeze() ** 2 + 3 * X.squeeze() + np.random.normal(0, 2, 100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train different models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.2),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    # Create individual visualizers for each model
    visualizers = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Create visualizer
        visualizer = vl.RegressionVisualizer(X, y, name)
        visualizer.update(model, epoch=0)
        visualizers[name] = visualizer
    
    # Create comparison plot
    comparator = vl.MultiModelRegressionComparator(X, y, list(models.keys()))
    for name, model in models.items():
        comparator.update_model(name, model, 0)
    
    comparison_fig = comparator.create_comparison_plot()
    comparison_fig.suptitle('1D Regression Model Comparison', fontsize=16)
    plt.show()
    
    print("✅ 1D regression example completed!")

def surface_2d_regression():
    """2D regression with surface visualization."""
    print("🌊 Running 2D Surface Regression Example...")
    
    # Generate 2D regression data
    X, y = make_regression(n_samples=300, n_features=2, noise=10, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_scaled, y)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    vl.plot_decision_boundary(model, X_scaled, y, ax, task_type='regression')
    ax.set_title('2D Regression Surface (SVR with RBF kernel)')
    ax.set_xlabel('Feature 1 (scaled)')
    ax.set_ylabel('Feature 2 (scaled)')
    
    plt.show()
    print("✅ 2D surface regression example completed!")

def high_dimensional_regression():
    """High-dimensional regression with dimensionality reduction."""
    print("📊 Running High-dimensional Regression Example...")
    
    # Load diabetes dataset (10 features)
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Visualize (will use PCA reduction)
    fig, ax = plt.subplots(figsize=(12, 10))
    vl.plot_decision_boundary(model, X, y, ax, task_type='regression')
    ax.set_title('High-dimensional Regression (10D → 2D via PCA)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    print(f"  Model R² Score: {model.score(X_test, y_test):.3f}")
    plt.show()
    print("✅ High-dimensional regression example completed!")

def regression_with_training_visualization():
    """Regression with live training visualization."""
    print("🎯 Running Live Training Regression Example...")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
    
    # Create a simple neural network for demonstration
    model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1, warm_start=True, random_state=42)
    
    # Create visualizer
    visualizer = vl.RegressionVisualizer(X, y, "Neural Network Training")
    
    # Simulate training with live updates
    plt.ion()  # Enable interactive mode
    
    for epoch in range(0, 100, 10):  # Update every 10 epochs
        model.max_iter = epoch + 10
        model.fit(X, y)
        
        visualizer.update(model, epoch + 10)
        plt.pause(0.5)  # Small pause to see the animation
        
        print(f"  Epoch {epoch + 10:3d}: Training...")
    
    plt.ioff()  # Disable interactive mode
    plt.show()
    print("✅ Live training visualization completed!")

def polynomial_regression_comparison():
    """Compare different polynomial degrees."""
    print("📐 Running Polynomial Regression Comparison...")
    
    # Generate polynomial-like data
    np.random.seed(42)
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 0.5 * X.squeeze() ** 3 - 2 * X.squeeze() ** 2 + X.squeeze() + np.random.normal(0, 0.3, 80)
    
    # Create polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    
    degrees = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, degree in enumerate(degrees):
        print(f"  Training polynomial degree {degree}...")
        
        # Create polynomial pipeline
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        poly_model.fit(X, y)
        
        # Create visualizer
        visualizer = vl.RegressionVisualizer(X, y, f"Polynomial Degree {degree}")
        visualizer.fig, visualizer.axes = fig, axes
        visualizer.ax_fit = axes[i]
        visualizer.initialized = True
        
        y_pred = poly_model.predict(X)
        visualizer._update_fit_plot(poly_model, y_pred)
        axes[i].set_title(f'Polynomial Degree {degree}')
    
    plt.suptitle('Polynomial Regression Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("✅ Polynomial comparison completed!")

if __name__ == "__main__":
    print("📈 VisualLearn Regression Examples")
    print("=" * 40)
    
    # Run all examples
    simple_1d_regression()
    print()
    
    surface_2d_regression()
    print()
    
    high_dimensional_regression()
    print()
    
    regression_with_training_visualization()
    print()
    
    polynomial_regression_comparison()
    
    print("\n🎉 All regression examples completed!")
    print("Key features demonstrated:")
    print("- 1D curve fitting with residuals analysis")
    print("- 2D surface regression visualization")  
    print("- High-dimensional data with PCA reduction")
    print("- Live training progress visualization")
    print("- Model comparison capabilities")
    print("- Comprehensive regression metrics")