import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from visuallearn.visualizer import plot_decision_boundary
from visuallearn.liveplotter import LivePlotter
import io
import imageio
from PIL import Image

class CombinedPlotCoordinator:
    def __init__(self, X, y, activation_tracker=None, gradient_tracker=None):
        self.X = X
        self.y = y
        self.activation_tracker = activation_tracker
        self.gradient_tracker = gradient_tracker
        self.frames = []
        self.capture_frames = False
        self.fig = None
        self.ax_boundary = None
        self.ax_loss = None
        self.ax_activations = []
        self.loss_plotter = None
        self.initialized = False

    def initialize(self):
        # Delay until activations are populated
        if self.activation_tracker:
            if not self.activation_tracker.activations:
                raise ValueError("ActivationTracker has no activations. Run one forward pass before init.")
            self.activation_names = list(self.activation_tracker.activations.keys())
        else:
            self.activation_names = []

        num_activations = len(self.activation_tracker.activations) if self.activation_tracker else 0
        num_gradients = len(self.gradient_tracker.gradients) if self.gradient_tracker else 0
        num_cols = max(2, num_activations, num_gradients)

        self.fig = plt.figure(figsize=(5 * num_cols, 12))
        gs = GridSpec(3, num_cols, figure=self.fig)

        # Row 0
        self.ax_boundary = self.fig.add_subplot(gs[0, 0])
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.loss_plotter = LivePlotter(self.ax_loss, ylabel="Loss")

        # Row 1: activations
        self.ax_activations = [self.fig.add_subplot(gs[1, i]) for i in range(num_activations)]

        # Row 2: gradients
        self.ax_gradients = [self.fig.add_subplot(gs[2, i]) for i in range(num_gradients)]

        plt.ion()
        plt.tight_layout()
        plt.show()
        self.initialized = True

    def update(self, model, epoch, loss):
        if not self.initialized:
            self.initialize()

        plot_decision_boundary(model, self.X, self.y, self.ax_boundary)
        self.loss_plotter.update(epoch, loss)

        #Plot activations
        if self.activation_tracker:
            for ax, (name, act) in zip(self.ax_activations, self.activation_tracker.activations.items()):
                ax.clear()
                ax.hist(act.numpy().flatten(), bins=30, alpha=0.7)
                ax.set_title(f"{name} (Epoch {epoch})")
        
        # Plot gradients
        if self.gradient_tracker:
            for ax, (name, grad) in zip(self.ax_gradients, self.gradient_tracker.gradients.items()):
                ax.clear()
                ax.hist(grad.numpy().flatten(), bins=30, alpha=0.7, color="orange")
                ax.set_title(f"Grad: {name} (Epoch {epoch})")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.capture_frames:
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            self.frames.append(img)
            buf.close()

    
    def enable_recording(self):
        self.capture_frames = True
        self.frames = []

    def export(self, filename="output.gif", format="gif", fps=5):
        if not self.frames:
            print("No frames captured. Did you call enable_recording()? Skipping export.")
            return

        if format == "gif":
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
        elif format == "mp4":
            imageio.mimsave(filename, self.frames, fps=fps)
        else:
            raise ValueError("Unsupported format. Use 'gif' or 'mp4'.")

        #print(f"Exported training visualization to {filename}")
