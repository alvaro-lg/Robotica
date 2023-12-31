import random
from collections.abc import Sequence
from random import choice
from typing import List

import numpy as np

from shared.actions import MovementAction
from shared.infrastructure.exceptions import SingletonException
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX

# Constants
N_X_STEPS: int = 3


class ActionSpace(Sequence):
    """
        Class that represents the action space of the robot.
    """

    # Static attributes
    _rotating_difference: float = 2
    _instance: 'ActionSpace' = None  # Singleton implementation
    __rnd = np.random.default_rng(2024)

    def __init__(self):
        """
            Class constructor.
        """
        if ActionSpace._instance is not None:  # Singleton implementation
            raise SingletonException()

        random.seed(10)

        def interpolation(x: float) -> float:
            return min((2. * ActionSpace._rotating_difference * x + Pioneer3DX.max_speed -
                        ActionSpace._rotating_difference) / Pioneer3DX.max_speed, 1.)

        self.__actions: List[MovementAction] = [
            MovementAction((interpolation(i), interpolation(1 - i)))
            for i in np.linspace(0, 1, N_X_STEPS, endpoint=True)
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
        return ActionSpace.__rnd.choice(self.__actions)

    # Interface implementation
    def __getitem__(self, index):
        return self.get_instance().actions[index]

    def __len__(self):
        return len(self.get_instance().actions)
