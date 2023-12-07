from collections.abc import Sequence
from typing import Tuple, List


class MovementAction(Sequence):

    def __init__(self, motors_speeds: Tuple[float, float]):

        # Class attributes
        self.__motors_speeds = motors_speeds

    # Properties
    def motors_speeds(self) -> Tuple[float, float]:
        """
            Getter for the motors_speeds private object.
        """
        return self.__motors_speeds

    def _motors_speeds(self, motors_speeds: Tuple[float, float]) -> None:
        """
            Setter for the motors_speeds private object.
            :param motors_speeds: new motors_speeds object to store.
        """
        self.__motors_speeds = motors_speeds

    property(fget=motors_speeds, fset=_motors_speeds)

    def __len__(self):
        return len(self.__motors_speeds)

    def __getitem__(self, index):
        return self.__motors_speeds[index]

    def __iter__(self):
        return iter(self.__motors_speeds)

    @staticmethod
    def get_action_space(len_x: int) -> List[Tuple[float, float]]:
        """
            Returns the action space for the given number of points.
            :param len_x: number of points.
            :return: the action space.
        """
        return [(i / len_x, j / len_x) for i in range(-len_x, len_x) for j in range(-len_x, len_x)]
