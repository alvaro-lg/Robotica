from collections.abc import Sequence
from random import choice
from typing import List

import numpy as np

from shared.actions import MovementAction
from shared.infrastructure.exceptions import SingletonException

# Constants
N_X_STEPS: int = 5


class ActionSpace(Sequence):
    """
        Class that represents the action space of the robot.
    """

    # Static attributes
    _instance: 'ActionSpace' = None  # Singleton implementation

    def __init__(self):
        """
            Class constructor.
        """
        if ActionSpace._instance is not None:  # Singleton implementation
            raise SingletonException()

        self.__actions: List[MovementAction] = [
            MovementAction((float(min(i / 0.5, 1)), float(min((1 - i) / 0.5, 1))))
            for i in np.arange(0, 1, 1 / N_X_STEPS)] + [
            MovementAction((-float(min(i / 0.5, 1)), -float(min((1 - i) / 0.5, 1))))
            for i in np.arange(0, 1, 1 / N_X_STEPS)
        ]

    @classmethod
    def get_instance(cls) -> 'ActionSpace':
        """
            Singleton implementation.
            :return: the _instance of the class.
        """
        # Initializing the _instance if it is not initialized yet
        if cls._instance is None:
            cls._instance = ActionSpace()

        return cls._instance

    @property
    def actions(self) -> List[MovementAction]:
        """
            Getter for the __actions attribute.
            :return: the __actions attribute.
        """
        return self.__actions

    def random_action(self) -> MovementAction:
        """
            Getter for the __actions attribute.
            :return: the __actions attribute.
        """
        return choice(self.__actions)

    # Interface implementation
    def __getitem__(self, index):
        return self.get_instance().actions[index]

    def __len__(self):
        return len(self.get_instance().actions)
