class CNNFeatureMapVisualizer:
    def __init__(self, model, input_tensor):
        self.model = model
        self.input_tensor = input_tensor
        self.activations = {}

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.__class__.__name__.startswith("Conv"):
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def forward_pass(self):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.input_tensor)

    def visualize(self):
        import matplotlib.pyplot as plt
        for name, act in self.activations.items():
            fig, axes = plt.subplots(1, min(8, act.shape[1]), figsize=(15, 3))
            for i, ax in enumerate(axes):
                ax.imshow(act[0, i].numpy(), cmap='viridis')
                ax.set_title(f'{name} | Channel {i}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()
