import matplotlib.pyplot as plt

class LivePlotter:
    def __init__(self, ax, ylabel="Value"):
        self.epochs = []
        self.values = []
        self.ax = ax
        self.ax.set_title("Loss Curve")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(ylabel)
        self.line, = self.ax.plot([], [], marker='o')

    def update(self, epoch, value):
        self.epochs.append(epoch)
        self.values.append(value)

        self.line.set_data(self.epochs, self.values)
        self.ax.relim()
        self.ax.autoscale_view()
