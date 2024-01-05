import itertools
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf

from shared.action_space import ActionSpace
from shared.data_types import AIModel
from simulations.infrastructure.model_repository import ModelRepository

MODEL_NAME = "model_ep321"
MODELS_PATH = Path("models")
FIGURES_PATH = Path("figures")


class StatisticsService:

    @staticmethod
    def run(models_path: Optional[Path] = MODELS_PATH, model_name: Optional[str] = None):

        # Gathering the model
        repo = ModelRepository(models_path)
        model = repo.load(model_name)

        StatisticsService._plot_probabilities(model)
        StatisticsService._plot_actions()

    @staticmethod
    def _plot_probabilities(model: AIModel, export_path: Optional[Path] = FIGURES_PATH / "probabilities.png"):
        # Create mesh grid
        x = np.linspace(0, 1, 75)
        y = np.linspace(0, 1, 75)
        X, Y = np.meshgrid(x, y)

        # Create the input mesh
        inputs = np.column_stack([X.ravel(), Y.ravel()])[:, np.newaxis, :]

        # Make predictions
        predictions = model.predict(inputs)
        z = tf.nn.softmax(predictions, axis=-1).numpy().reshape(len(x), len(y), -1)

        # Plotting the results
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        fake_legend = []
        legend_labels = []
        n_planes = z.shape[2]
        for i in range(n_planes):
            surface = z[:, :, i]

            # Plot surface
            ax.plot_surface(X, Y, surface, alpha=0.5, color=plt.get_cmap('tab10').colors[i])

            # Create legend
            fake_legend.append(plt.Line2D([0], [0], linestyle="none", c=plt.get_cmap('tab10').colors[i], marker='o'))
            legend_labels.append(f"Action {i}")

        # Reverse the direction of the Y-axis
        ax.set_ylim(ax.get_ylim()[::-1])

        # Set labels and title
        ax.set_xlabel('Normalized x-coordinate')
        ax.set_ylabel('Ratio of area occupied')
        ax.set_zlabel('Probability')
        ax.set_title('Probabilities for each action')
        ax.legend(fake_legend, legend_labels, numpoints=1, loc='upper left')

        # Add an arrow outside the axis to represent the viewing/front perspective
        arrow_start = (0.5, -0.75, 0.3)  # Adjust the start position of the arrow
        arrow_direction = (0, 1, 0)  # Adjust the direction of the arrow
        ax.quiver(*arrow_start, *arrow_direction, length=0.2, color='black', linewidth=2.5, arrow_length_ratio=0.1)

        # Set bounds for the x, y, and z axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Tilt rotation (adjust elevation and azimuth angles)
        ax.view_init(elev=20)

        plt.tight_layout()
        plt.savefig(export_path, dpi=125)

    @staticmethod
    def _plot_actions(export_path: Optional[Path] = FIGURES_PATH / "directions.png"):

        wheel_speeds = [x.motors_speeds for x in ActionSpace.get_instance().actions]

        # Calculate the resulting direction based on wheel speeds
        directions = []
        for left, right in wheel_speeds:
            direction = right - left            # Calculating the direction based on the speed difference
            directions.append(direction * 90)   # Normalizing towards 90º

        # Convert directions to radians for polar plot
        angles = np.radians(directions)

        # Plotting the directions
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        legend_labels = []
        fake_legend = []
        for i, angle in enumerate(angles):
            x = angle
            y = 1
            ax.annotate(f"", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=plt.get_cmap('tab10').colors[i]), fontsize=30)
            fake_legend.append(mpl.lines.Line2D([0], [0], linestyle="none", c=plt.get_cmap('tab10').colors[i],
                                                marker='o'))
            legend_labels.append(f"Action {i}")

        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')  # Set the zero direction to the north
        ax.set_title('Action space VS Directions')
        plt.tight_layout()
        ax.legend(fake_legend, legend_labels, numpoints=1)
        plt.savefig(export_path, dpi=125)


if __name__ == '__main__':
    # Actual application startup
    StatisticsService.run(model_name=MODEL_NAME)
