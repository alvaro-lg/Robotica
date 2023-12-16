from abc import ABC, abstractmethod
from typing import Tuple

from shared.domain.interfaces.simulation_physical_element import SimulationPhysicalElement


class SimulationLogicalElement(SimulationPhysicalElement, ABC):
    """
        Class that represents a logical element in the simulation, premising this to provide more operations over it.
    """

    @abstractmethod
    def get_orientation(self) -> Tuple[float, float, float]:
        """
            Retrieves the orientation of the object.
        """
        pass

    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        """
            Retrieves the position of the object.
        """
        pass
