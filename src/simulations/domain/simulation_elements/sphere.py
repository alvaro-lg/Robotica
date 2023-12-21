import logging
from typing import Tuple

from shared.domain.interfaces.simulation_physical_element import SimulationPhysicalElement
from shared.infrastructure.data_types import ObjectId, ObjectHandler
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector


class Sphere(SimulationPhysicalElement):
    """
        Class that represents the sphere in the simulation a wraps functions to use it
    """

    def __init__(self, sim_connector: CoppeliaSimConnector, ball_id: ObjectId):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
            :param sim_connector: connector class to the CoppeliaSim simulation.
            :param ball_id: string containing the filename of the robot in the simulation.
        """
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info(f"Getting handles ({ball_id})")

        # Attributes initialization
        self.__sim_connector: CoppeliaSimConnector = sim_connector
        self.__ball_handler: ObjectHandler = self.__sim_connector.get_object_handler(f'/{ball_id}')

    def set_orientation(self, orientation: Tuple[float, float, float]) -> None:
        """
            Sets the position of the robot.
            :param orientation: a tuple containing the orientation of the robot on the three axis.
        """
        self.__sim_connector.set_object_orientation(self.__ball_handler, orientation)

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
            Sets the position of the robot.
            :param position: a tuple containing the position of the robot on the three axis.
        """
        self.__sim_connector.set_object_position(self.__ball_handler, position)

    def reset(self, shuffle: bool = False) -> None:
        """
            Resets the ball to its initial position.
            :param shuffle:
        """
        self.__sim_connector.set_object_position(self.__ball_handler, (3.5, 0., 0.15))
