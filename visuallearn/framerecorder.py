import io
import imageio
from PIL import Image
from datetime import datetime

class FrameRecorder:
    def __init__(self, fig, dpi=150):
        self.fig = fig
        self.frames = []
        self.dpi = dpi

    def capture(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        self.frames.append(img)
        buf.close()

    def export(self, filename="output.gif", format="gif", fps=10):
        if not self.frames:
            print("No frames captured. Skipping export.")
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

        print(f"Exported training animation to {filename}")

    def export_auto(self, format="gif", fps=10, prefix="mlvis"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_{timestamp}.{format}"
        self.export(filename, format, fps)
