import torch.nn as nn

def classify_model(model):
    """
    Analyze a PyTorch model and return a high-level string classification.

    Returns:
        One of: "MLP", "CNN", "Transformer", "RNN", or "Unknown"
    """
    has_linear = False
    has_conv = False
    has_attention = False
    has_recurrent = False

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            has_linear = True
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            has_conv = True
        if "attention" in module.__class__.__name__.lower():
            has_attention = True
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            has_recurrent = True

    if has_conv:
        return "CNN"
    elif has_attention:
        return "Transformer"
    elif has_recurrent:
        return "RNN"
    elif has_linear:
        return "MLP"
    else:
        return "Unknown"
