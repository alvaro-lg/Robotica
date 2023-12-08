from collections.abc import Sequence
from random import choice
from typing import Tuple, List

import numpy as np


class MovementAction(Sequence):

    def __init__(self, motors_speeds: Tuple[float, float]):
        # Class attributes
        self.__motors_speeds = motors_speeds

    # Properties
    @property
    def motors_speeds(self) -> Tuple[float, float]:
        """
            Getter for the motors_speeds private object.
        """
        return self.__motors_speeds

    def __len__(self):
        return len(self.__motors_speeds)

    def __getitem__(self, index):
        return self.__motors_speeds[index]

    def __iter__(self):
        return iter(self.__motors_speeds)


class EnumeratedMovementAction(MovementAction):

    def __init__(self, idx: int):
        super().__init__(MovementActionFactory.get_n_ith_action(idx).motors_speeds)
        self.__idx: int = idx

    @property
    def motors_speeds(self) -> Tuple[float, float]:
        """
            Getter for the motors_speeds private object.
        """
        return MovementActionFactory.get_n_ith_action(self.__idx).motors_speeds

    @property
    def idx(self) -> int:
        """
            Getter for the idx private object.
        """
        return self.__idx


class MovementActionFactory:

    # Static attributes
    N_X_STEPS: int = 8
    N_ACTIONS: int = N_X_STEPS
    _action_space: List[MovementAction] = None

    @classmethod
    def create_action_space(cls):
        cls._action_space = [
            MovementAction((float(min(i / 0.5, 1)), float(min((1 - i) / 0.5, 1))))
            for i in np.arange(0, 1, 1 / cls.N_X_STEPS)]

    @staticmethod
    def get_random_enum_action() -> EnumeratedMovementAction:
        """
            Returns a random action.
            :return: a random action.
        """
        return EnumeratedMovementAction(choice(range(len(MovementActionFactory._action_space))))

    @staticmethod
    def get_random_action() -> MovementAction:
        """
            Returns a random action.
            :return: a random action.
        """
        return choice(MovementActionFactory._action_space)

    @staticmethod
    def get_n_ith_action(n: int) -> MovementAction:
        """
            Returns the action space for the given number of points.
            :param n: index of the action to return.
            :return: the action space.
        """
        return MovementActionFactory._action_space[n]