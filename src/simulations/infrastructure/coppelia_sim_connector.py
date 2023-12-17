import logging
import time
from typing import Tuple, List, Optional

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from shared.data_types import CameraReadingData, LidarReadingData
from shared.infrastructure.data_types import ObjectHandler, ObjectId
from shared.infrastructure.exceptions import SingletonException
from shared.infrastructure.interfaces.simulation_connector import SimulationConnector
from shared.domain.interfaces.simulation_physical_element import SimulationPhysicalElement


class CoppeliaSimConnector(SimulationConnector):
    """
        Class that connects to CoppeliaSim and provides the basic functions for starting, stopping and checking if the
        simulation is still running.
    """
    # Class-static attributes
    simulation_dims: Tuple[Tuple[float, float], Tuple[float, float]] = ((-4., 4.), (-4., 4.))
    _connector_instance: 'CoppeliaSimConnector' = None  # Singleton implementation

    def __init__(self):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
        """
        if CoppeliaSimConnector._connector_instance is not None:  # Singleton implementation
            raise SingletonException()

        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info("Connecting to CoppeliaSim...")

        # Class attributes initialization
        self.__client = RemoteAPIClient()
        self.__sim = self.__client.require('sim')
        self.__idle_fps: int = self.__sim.getInt32Param(self.__sim.intparam_idle_fps)
        self.__sim_elements: List[SimulationPhysicalElement] = []

    # Singleton implementation
    @classmethod
    def get_instance(cls):
        """
            Singleton implementation.
            :return: the _instance of the class.
        """

        # Initializing the _instance if it is not initialized yet
        if cls._connector_instance is None:
            cls._connector_instance = CoppeliaSimConnector()

        return cls._connector_instance

    # Interface implementation
    def start_simulation(self, stepping: bool = False) -> None:
        """
            Starts the simulation in CoppeliaSim.
        """
        if stepping:
            self.__client.setStepping(True)
        self.__logger.debug("Starting simulation...")
        self.__sim.setInt32Param(self.__idle_fps, 0)
        self.__sim.startSimulation()

    def stop_simulation(self) -> None:
        """
            Stops the simulation.
        """
        self.__sim.stopSimulation()
        while self.is_running():
            time.sleep(0.1)
        self.__sim.setInt32Param(self.__sim.intparam_idle_fps, self.__idle_fps)
        self.__logger.debug("Simulation stopped...")

    def step(self) -> None:
        """
            Steps the simulation.
        """
        self.__client.step()

    def reset_simulation(self, shuffle: Optional[bool] = False) -> None:
        """
            Starts the simulation in CoppeliaSim.
        """
        self.__logger.debug("Resetting simulation...")
        for sim_element in self.__sim_elements:
            sim_element.reset(shuffle=shuffle)

    def is_running(self) -> bool:
        """
            Checks whether the simulation is still running.
            :return: boolean: True if it is still running, False otherwise.
        """
        return self.__sim.getSimulationState() != self.__sim.simulation_stopped

    def get_time(self) -> float:
        """
            Returns the current simulation time.
            :return: float: current simulation time.
        """
        return self.__sim.getSimulationTime()

    # Proper methods
    def add_sim_element(self, sim_element: SimulationPhysicalElement) -> None:
        """
            Adds a simulation element to the simulation.
            :param sim_element: simulation element to be added.
        """
        self.__sim_elements.append(sim_element)

    def add_sim_elements(self, sim_elements: List[SimulationPhysicalElement]) -> None:
        """
            Adds simulation elements to the simulation.
            :param sim_elements: listo of simulation elements to be added.
        """
        for sim_element in sim_elements:
            self.add_sim_element(sim_element)

    # Observer methods
    def get_object_handler(self, object_id: ObjectId) -> ObjectHandler:
        """
            Returns the handler of the object with the given filename.
            :param object_id: string containing the filename of the object.
            :return: integer containing the handler of the object.
        """
        return self.__sim.getObject(object_id)

    def get_sensor_reading(self, sensor_handler: ObjectHandler) -> float:
        """
            Returns the reading of the given sensor.
            :param sensor_handler: handler of the sensor.
            :return: float containing the reading of the sensor.
        """
        return self.__sim.readProximitySensor(sensor_handler)

    def get_camera_reading(self, camera_handler: ObjectHandler) -> CameraReadingData:
        """
            Returns the reading of the given camera.
            :param camera_handler: handler of the camera.
            :return: float containing the reading of the camera.
        """

        # Getting raw data of the image
        img, resX, resY = self.__sim.getVisionSensorCharImage(camera_handler)

        # Processing image data
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        return img

    def get_lidar_reading(self, lidar_handler: ObjectHandler) -> LidarReadingData:
        """
            Returns the reading of the given lidar.
            :param lidar_handler: handler of the lidar.
            :return: float containing the reading of the lidar.
        """
        # Getting and unpacking the readings of the lidar
        data = self.__sim.getStringSignal(lidar_handler)
        if data is None:
            return []
        else:
            return self.__sim.unpackFloatTable(data)

    def get_joint_velocity(self, joint_handler: ObjectHandler) -> float:
        """
            Returns the velocity of the given joint.
            :param joint_handler: handler of the joint.
            :return: float containing the velocity of the joint.
        """
        return self.__sim.getJointTargetVelocity(joint_handler)

    def get_object_orientation(self, object_handler: ObjectHandler) -> Tuple[float, float, float]:
        """
            Gets the orientation of the given object.
            :param object_handler: handler of the object.
            :return: a tuple containing the orientation of the object on the three axis.
        """
        return self.__sim.getObjectOrientation(object_handler)

    def get_object_position(self, object_handler: ObjectHandler) -> Tuple[float, float, float]:
        """
            Gets the position of the given object.
            :param object_handler: handler of the object.
            :return: a tuple containing the position of the object on the three axis.
        """
        return self.__sim.getObjectPosition(object_handler)

    # Setter (in the simulation) methods
    def set_joint_velocity(self, joint_handler: ObjectHandler, velocity: float) -> None:
        """
            Sets the velocity of the given joint.
            :param joint_handler: handler of the joint.
            :param velocity: velocity to be set.
        """
        self.__sim.setJointTargetVelocity(joint_handler, velocity)

    def set_object_position(self, object_handler: ObjectHandler, position: Tuple[float, float, float]) -> None:
        """
            Sets the position of the given object.
            :param object_handler: handler of the object.
            :param position: position to be set.
        """
        # Checking if the position is valid
        if not isinstance(position, Tuple):
            TypeError("Invalid positions types")
        if not all(isinstance(p, (float, int)) for p in position):
            TypeError("Invalid position type")

        self.__sim.setObjectPosition(object_handler, position)

    def set_object_orientation(self, object_handler: ObjectHandler, orientation: Tuple[float, float, float]) -> None:
        """
            Sets the orientation of the given object.
            :param object_handler: handler of the object.
            :param orientation: orientation to be set.
        """
        # Checking if the orientation is valid
        if not isinstance(orientation, Tuple):
            TypeError("Invalid orientations types")
        if not all(isinstance(o, (float, int)) for o in orientation):
            TypeError("Invalid orientation type")

        self.__sim.setObjectOrientation(object_handler, orientation)
