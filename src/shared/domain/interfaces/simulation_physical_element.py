from abc import ABC, abstractmethod
from typing import Tuple


class SimulationPhysicalElement(ABC):
    """
        Class that represents a physical element in the simulation.
    """

    @abstractmethod
    def set_orientation(self, rotation: Tuple[float, float, float]) -> None:
        """
            Sets the orientation of the object.
        """
        pass

    @abstractmethod
    def set_position(self, rotation: Tuple[float, float, float]) -> None:
        """
            Sets the position of the object.
        """
        pass

    @abstractmethod
    def reset(self, shuffle: bool = False) -> None:
        """
            Resets the object to its initial state.
        """
        pass
