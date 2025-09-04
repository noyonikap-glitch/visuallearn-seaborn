import matplotlib.pyplot as plt
import torch

class ActivationTracker:
    def __init__(self):
        self.activations = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def attach_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(self.hook_fn(name))