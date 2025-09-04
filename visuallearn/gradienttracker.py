import torch

class GradientTracker:
    def __init__(self):
        self.gradients = {}

    def hook_fn(self, name):
        def hook(module, grad_input, grad_output):
            # grad_output is a tuple of outputs; we take the first
            self.gradients[name] = grad_output[0].detach().cpu()
        return hook

    def attach_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.register_full_backward_hook(self.hook_fn(name))
