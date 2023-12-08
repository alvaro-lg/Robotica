from typing import Tuple

import numpy as np

from controllers.domain.image_processing_service import ImageProcessingService
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
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
        if state.is_ball_in_sight():

            self.__useless_steps = 0

            # Defining the interpolation function
            def interpolation(x: float) -> float:
                return np.sqrt(1 - x)  # Square root interpolation

            # Making the robot go slower when is visually closer to the ball
            new_max_speed = interpolation(state.area_norm)

            # Calculating relative normalized distances among x-axis
            dl = state.x_norm
            dr = 1 - dl

            # Calculate the speed_ratio of each wheel
            left_speed = new_max_speed * min(dl / 0.5, 1)
            right_speed = new_max_speed * min(dr / 0.5, 1)

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
