from collections.abc import Sequence
from typing import Tuple


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
