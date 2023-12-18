import math

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
        if curr_state.area_norm > next_state.area_norm:
            reward += 1

        # Reinforcing the robot to keep the ball centered
        if abs(next_state.x_norm - 0.5) > 0.15:  # 0.5 +- 15% of deviation
            reward += 1

        return 0
