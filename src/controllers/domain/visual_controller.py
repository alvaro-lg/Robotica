from typing import Tuple

import numpy as np

from controllers.domain.image_processing_service import ImageProcessingService
from controllers.infrastructure.pioneer3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction
from shared.data_types import ActionT
from shared.state import State


class VisualController:

    # Static variables
    idle_speeds: Tuple[float, float] = -1.5, 1.5
    steps_for_idle: int = 30

    def __init__(self):
        # Only variable declaration
        self.__last_action: ActionT = None
        self.__useless_steps: int = 0

    def get_next_action(self, state: State) -> MovementAction:
        """
            Calculates the next action to perform based on the current state.
            :param state: Actual state of the robot.
            :return: The next action to perform.
        """
        # Getting contours
        img = state.camera_reading
        contours = ImageProcessingService.get_contours(img)

        if len(contours) > 0:
            # Resetting counter
            self.__useless_steps = 0

            # Extract x-coordinate of the circle center
            center, area = ImageProcessingService.get_shape(img)
            center_x, center_y = center
            len_x = img.shape[0]

            # Making the robot go slower when is visually closer to the ball
            screen_area = (img.shape[0] * img.shape[1]) * 0.85  # Adjusting to 85% of the screen area

            # Defining the interpolation function
            def interpolation(x: float) -> float:
                return max(1 + np.emath.logn(3, 1 - x), 0)  # Logarithmic interpolation

            # Interpolating
            new_max_speed = Pioneer3DXConnector.max_speed * interpolation(float(area / screen_area))

            # Calculating relative normalized distances among x-axis
            dl = center_x / len_x
            dr = 1 - (center_x / len_x)

            # Calculate the speed of each wheel
            left_speed = new_max_speed * max(dl / 0.5, 0.5)
            right_speed = new_max_speed * max(dr / 0.5, 0.5)

            # Storing last action
            self.__last_action = MovementAction((left_speed, right_speed))

            return self.__last_action
        else:
            # Increasing counter
            self.__useless_steps += 1

            # Returning last action if there is one or idle action if not
            if self.__last_action is None or self.__useless_steps >= VisualController.steps_for_idle:
                return MovementAction(VisualController.idle_speeds)
            else:
                return self.__last_action
