from typing import Tuple


class MovementAction:

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
