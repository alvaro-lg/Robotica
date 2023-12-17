import math

from shared.actions import MovementAction
from shared.state import State


class RewardService:
    # Static variables
    target_area: float = 0.75

    @staticmethod
    def get_reward(state: State) -> int:
        """
        Returns the reward for the given state, action and next state.
        """
        if not state.is_ball_in_sight():
            return 0
        else:
            return math.ceil(5 * ((min(state.area_norm / RewardService.target_area, 1) +
                                    (abs(state.x_norm - 0.5) / 0.5))) / 2)
