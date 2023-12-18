import math
import numpy as np

from shared.actions import MovementAction
from shared.state import State


class RewardService:

    @staticmethod
    def get_reward(curr_state: State, next_state: State) -> int:
        """
        Returns the reward for the given state, action and next state.
        :param curr_state: State object representing the current state.
        :param next_state: State object representing the next state.
        :return: integer representing the reward.
        """
        reward = 0

        # Reinforcing the robot to get closer
        if next_state.is_ball_in_sight() >= curr_state.is_ball_in_sight():
            reward += 2
        else:
            reward -= 1

        # Reinforcing the robot to keep the ball centered
        if abs(next_state.x_norm - 0.5) <= abs(curr_state.x_norm - 0.5):
            reward += 2
        else:
            reward -= 1

        # Penalizing lost of sight of the ball
        if not next_state.is_ball_in_sight():
            reward -= 10

        return reward

    # TODO Remove if a better implementation is achieved
    """reward = 0

    # Reinforcing the robot to get closer
    if next_state.area_norm >= curr_state.area_norm:
        reward += 1

    # Reinforcing the robot to keep the ball in sight
    if next_state.is_ball_in_sight():
        # Normal distribution for interpolating the reward with the center of the x-axis
        def norm(x: float, mean: float = 0.5, std_dev: float = 0.2) -> float:
            if 0 <= x <= 1:
                return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
            else:
                return 0.
        # Reinforcing the robot to keep the ball centered
        reward += norm(next_state.x_norm)
    else:
        reward -= 10

    return reward"""
