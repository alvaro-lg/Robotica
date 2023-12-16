import math

from shared.actions import MovementAction
from shared.state import State


class RewardService:
    # Static variables
    target_area: float = 0.75

    @staticmethod
    def get_reward(state: State, action: MovementAction, next_state: State) -> int:
        """
        Returns the reward for the given state, action and next state.
        """
        if not state.is_ball_in_sight():
            return 0
        else:
            if next_state.is_ball_in_sight():
                return math.ceil(20 * ((min(next_state.area_norm / RewardService.target_area, 1) +
                                        (abs(next_state.x_norm - 0.5) / 0.5))) / 2)
            else:
                return -10
