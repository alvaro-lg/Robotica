import math
import numpy as np

from shared.actions import MovementAction
from shared.state import State


class RewardService:

    @staticmethod
    def get_reward(curr_state: State, next_state: State) -> int:
        """
        Returns the reward for the given states, action and next states.
        :param curr_state: State object representing the current states.
        :param next_state: State object representing the next states.
        :return: integer representing the reward.
        """
        reward = 0

        # Penalizing lost of sight of the ball
        if curr_state.is_ball_in_sight() and not next_state.is_ball_in_sight():
            return -10

        # Reinforcing the robot to keep the ball centered
        def norm(x: float, mean: float = 0.5, std_dev: float = 0.2) -> float:
            if 0 <= x <= 1:
                # Normal distribution for interpolating the reward with the center of the x-axis
                return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
            else:
                return 0.

        def lin(x: float) -> float:
            # Linear distribution for interpolating the reward with the center of the x-axis
            return 1. - abs(0.5 - x) * 2.

        reward += lin(next_state.x_norm)

        return reward
