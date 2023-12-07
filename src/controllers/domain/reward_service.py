from shared.actions import MovementAction
from shared.state import State


class RewardService:

    @staticmethod
    def get_reward(state: State, action: MovementAction, next_state: State) -> int:
        """
        Returns the reward for the given state, action and next state.
        """
        if not state.is_ball_in_sight():
            if next_state.is_ball_in_sight():
                return 2
            else:
                return 0
        else:
            if next_state.is_ball_in_sight():
                return 1
            else:
                return -1
