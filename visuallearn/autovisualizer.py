from visuallearn.modelanalyzer import classify_model
from visuallearn.combinedplot import CombinedPlotCoordinator
from visuallearn.cnnvisualizer import CNNFeatureMapVisualizer

def auto_visualize(model, X, y=None, input_tensor=None, **kwargs):
    model_type = classify_model(model)
    print(f"[auto_visualize] Detected model type: {model_type}")

    if model_type == "MLP":
        vis = CombinedPlotCoordinator(X, y, **kwargs)
        return vis
    elif model_type == "CNN":
        if input_tensor is None:
            raise ValueError("input_tensor is required for CNN visualization")
        vis = CNNFeatureMapVisualizer(model, input_tensor)
        vis.register_hooks()
        return vis
    else:
        raise NotImplementedError(f"No visualization available for model type: {model_type}")
