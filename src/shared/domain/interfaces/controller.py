from abc import ABC, abstractmethod
from shared.data_types import StateT, ActionT


class Controller(ABC):

    @abstractmethod
    def get_next_action(self, state: StateT) -> ActionT:
        """
            Calculates the next action to perform based on the current states.
            :param state: Actual states of the robot.
            :return: The next action to perform.
        """
        pass
