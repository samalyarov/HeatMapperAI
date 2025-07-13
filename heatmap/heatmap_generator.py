import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeatmapGenerator:
    def __init__(self, background_image, alpha=0.4, cmap='hot'):
        """
        Initialize heatmap generator.

        Args:
            background_image (numpy.ndarray): RGB background image.
            alpha (float): Transparency of heatmap overlay.
            cmap (str): Colormap to use for heatmap.
        """
        self.background_image = background_image
        self.alpha = alpha
        self.cmap = cmap

        if self.background_image is None:
            raise ValueError("Background image cannot be None.")

        logger.info(f"HeatmapGenerator initialized with alpha={alpha}, cmap={cmap}")

    def generate_heatmap(self, points, output_path=None, show=False):
        """
        Generates and saves (or shows) a heatmap overlay.

        Args:
            points (list of tuples): (x, y) vehicle detection centers.
            output_path (str): Path to save heatmap PNG. If None, won't save.
            show (bool): Whether to display the heatmap inline.
        """
        if not points:
            logger.warning("No detection points provided for heatmap.")
            return

        xs, ys = zip(*points)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.background_image)

        # Plot the KDE heatmap overlay
        sns.kdeplot(
            x=xs,
            y=ys,
            fill=True,
            cmap=self.cmap,
            bw_adjust=0.5,
            levels=100,
            thresh=0.05,
            alpha=self.alpha
        )

        ax.set_title('Vehicle Heatmap Overlay')
        ax.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            logger.info(f"Saved heatmap overlay to {output_path}")

        if show:
            plt.show()

        plt.close(fig)
