import logging
import random
from typing import Tuple, List, Optional

from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from shared.infrastructure.data_types import ObjectHandler, ObjectId
from shared.domain.interfaces.simulation_logical_element import SimulationLogicalElement


class Pathway(SimulationLogicalElement):
    """
        Class that represents the pathway in the simulation.
    """

    # Class-static attributes
    num_points: int = 16
    _points_ids: str = '/{}/ctrlPt[{}]'  # TODO Implement a proper handler for path points

    def __init__(self, sim_connector: CoppeliaSimConnector, path_id: ObjectId):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
            :param path_id: string containing the filename of the pathway in the simulation.
        """
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info(f"Getting handles (Pathway {path_id})")

        # Attributes initialization
        self.__sim_connector: CoppeliaSimConnector = sim_connector
        self.__path_handler: ObjectHandler = self.__sim_connector.get_object_handler(f"/{path_id}")
        self.__points_handlers: List[ObjectHandler] = []
        for id_ in range(Pathway.num_points):
            self.__points_handlers.append(
                self.__sim_connector.get_object_handler(Pathway._points_ids.format(path_id, id_))
            )

    # Interface implementation
    def get_orientation(self) -> Tuple[float, float, float]:
        """
            Retrieves the orientation of the path object.
            :return: a tuple containing the orientation of the object on the three axis.
        """
        return self.__sim_connector.get_object_orientation(self.__path_handler)

    def get_position(self) -> Tuple[float, float, float]:
        """
            Retrieves the position of the path object.
            :return: a tuple containing the position of the object on the three axis.
        """
        return self.__sim_connector.get_object_position(self.__path_handler)

    def set_orientation(self, orientation: Tuple[float, float, float]) -> None:
        """
            Sets the orientation of the path object.
            :param orientation: a tuple containing the orientation of the object on the three axis.
        """
        return self.__sim_connector.set_object_orientation(self.__path_handler, orientation)

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
            Sets the position of the path object.
            :param position: a tuple containing the position of the object on the three axis.
        """
        return self.__sim_connector.set_object_position(self.__path_handler, position)

    def reset(self, shuffle: Optional[bool] = False) -> None:
        """
            Resets the object to its initial states.
            :param shuffle: boolean indicating whether the points should be shuffled.
        """
        if shuffle:
            # Getting the points and their positions
            positions = []
            for point_handler in self.__points_handlers:
                positions.append(self.__sim_connector.get_object_position(point_handler))

            # Shuffling the points
            print(positions)
            random.shuffle(positions)
            print(positions)

            # Setting the new positions
            for point_handler, pos in zip(self.__points_handlers, positions):
                self.__sim_connector.set_object_position(point_handler, pos)

            positions = []
            for point_handler in self.__points_handlers:
                positions.append(self.__sim_connector.get_object_position(point_handler))
            print(positions)
