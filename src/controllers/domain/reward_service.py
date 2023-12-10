import math

from shared.actions import MovementAction
from shared.state import State


class RewardService:

    # Static variables
    TARGET_AREA: float = 0.75

    @staticmethod
    def get_reward(state: State, action: MovementAction, next_state: State) -> int:
        """
        Returns the reward for the given state, action and next state.
        """
        if not state.is_ball_in_sight():
            return 0
        else:
            return 1 + math.ceil(5 * min(next_state.area_norm / RewardService.TARGET_AREA, 1))
