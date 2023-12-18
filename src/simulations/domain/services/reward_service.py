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
        if next_state.is_ball_in_sight():
            reward += 1
            if curr_state.area_norm > next_state.area_norm:
                reward += 1

        # Reinforcing the robot to keep the ball centered
        if abs(next_state.x_norm - 0.5) > 0.15:  # 0.5 +- 15% of deviation
            reward += 1
            if abs(curr_state.x_norm - 0.5) > abs(next_state.x_norm - 0.5):
                reward += 1

        return reward
