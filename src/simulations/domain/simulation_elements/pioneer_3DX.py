import logging
import numpy as np

from typing import Tuple, List, Optional

from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from shared.actions import MovementAction
from shared.data_types import CameraReadingData, SonarsReadingsData, LidarReadingData, RobotControllerT, ActionT
from shared.infrastructure.data_types import ObjectHandler, ObjectId
from shared.domain.interfaces.simulation_physical_element import SimulationPhysicalElement
from shared.state import State


class Pioneer3DX(SimulationPhysicalElement):
    """
        Class that represents the Pioneer 3DX robot in the simulation a wraps functions to use the .
    """

    # Class-static attributes
    num_sonar: int = 16
    max_sonar: int = 1
    max_speed: float = 5.
    _distance_for_collision: float = 0.08

    def __init__(self, sim_connector: CoppeliaSimConnector, controller: RobotControllerT, robot_id: ObjectId):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
            :param sim_connector: connector class to the CoppeliaSim simulation.
            :param robot_id: string containing the filename of the robot in the simulation.
            :param controller: controllers class that will control the robot.
        """
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info(f"Getting handles ({robot_id})")

        # Attributes initialization
        self.__sim_connector: CoppeliaSimConnector = sim_connector
        self.__controller: RobotControllerT = controller
        self.__robot_handler: ObjectHandler = self.__sim_connector.get_object_handler(f'/{robot_id}')
        self.__left_motor_handler: ObjectHandler = self.__sim_connector.get_object_handler(f'/{robot_id}/leftMotor')
        self.__right_motor_handler: ObjectHandler = self.__sim_connector.get_object_handler(f'/{robot_id}/rightMotor')
        self.__camera_handlers: ObjectHandler = self.__sim_connector.get_object_handler(f'/{robot_id}/camera')
        self.__sonars_handlers: List[ObjectHandler] = []
        for i in range(self.num_sonar):
            self.__sonars_handlers.append(self.__sim_connector.get_object_handler(f'/{robot_id}/ultrasonicSensor[{i}]'))

    # Interface implementation
    def set_orientation(self, orientation: Tuple[float, float, float]) -> None:
        """
            Sets the position of the robot.
            :param orientation: a tuple containing the position of the robot on the three axis.
        """
        self.__sim_connector.set_object_orientation(self.__robot_handler, orientation)

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
            Sets the position of the robot.
            :param position: a tuple containing the position of the robot on the three axis.
        """
        self.__sim_connector.set_object_position(self.__robot_handler, position)

    def reset(self, shuffle: Optional[bool] = False) -> None:
        """
            Resets the robot to its initial state.
        """
        if shuffle:
            '''x_bounds, y_bounds = self.__sim_connector.simulation_dims
            x = np.random.uniform(x_bounds[0], x_bounds[1]) * 0.8
            y = np.random.uniform(y_bounds[0], y_bounds[1]) * 0.8
            self.__sim_connector.set_object_position(self.__robot_handler, (x, y, 0.15))'''
            # Only shuffling orientation
            self.__sim_connector.set_object_position(self.__robot_handler, (0, 0, 0.15))
            self.__sim_connector.set_object_orientation(self.__robot_handler,
                                                        (0, 0, np.random.uniform(0, np.pi)))
        else:
            self.__sim_connector.set_object_position(self.__robot_handler, (0., 0., 0.15))
            self.__sim_connector.set_object_orientation(self.__robot_handler,
                                                        (0, 0, 0))

    # Observer methods
    def get_motors_speeds_ratio(self) -> Tuple[float, float]:
        """
            Gets the speed_ratio of the two motors on the robot wheels on the simulation.
            :return: a tuple of two floating point numbers, containing the actual values of each motor_handler.
        """
        left_speed = self._get_motor_speed_ratio(self.__left_motor_handler)
        right_speed = self._get_motor_speed_ratio(self.__right_motor_handler)
        return left_speed, right_speed

    def _get_motor_speed_ratio(self, motor_handler: ObjectHandler) -> float:
        """
            Gets the speed_ratio of the left motor_handler on the robot wheels on the simulation.
            :return: floating point number representing actual speed_ratio of the left motor_handler.
        """
        return self.__sim_connector.get_joint_velocity(motor_handler)

    def get_sonars_readings(self) -> SonarsReadingsData:
        """
            Retrieves a list with the readings of the sonars_readings associated with ths robot.
            :return: a list containing the reading for each sensor.
        """
        # List to store readings
        readings = list()

        # Getting each reading
        for sonar in self.__sonars_handlers:
            res, dist, _, _, _ = self.__sim_connector.get_sensor_reading(sonar)
            readings.append(dist if res == 1 else self.max_sonar)

        return readings

    def get_camera_reading(self) -> CameraReadingData:
        """
            Retrieves the image captured by the robot's camera at execution time.
            :return: a ndarray containing the data of the image captured.
        """
        return self.__sim_connector.get_camera_reading(self.__camera_handlers)

    def get_state(self) -> State:
        """
            Retrieves the state of the robot.
            :return: a State object containing the state of the robot.
        """
        return State(self.get_camera_reading())

    # Illegal methods if a real robot is used
    def _get_orientation(self) -> Tuple[float, float, float]:
        """
            Retrieves the orientation of the robot.
            :return: a tuple containing the orientation of the robot on the three axis.
        """
        # TODO Implement using a sensor
        return self.__sim_connector.get_object_orientation(self.__robot_handler)

    def _get_position(self) -> Tuple[float, float, float]:
        """
            Retrieves the position of the robot.
            :return: a tuple containing the position of the robot on the three axis.
        """
        # TODO Implement using a sensor
        return self.__sim_connector.get_object_position(self.__robot_handler)

    # Setter (in the robot of the simulation) methods
    def set_motors_speeds_ratio(self, speeds_ratios: Tuple[float, float]) -> None:
        """
            Sets the speed_ratio of the two motors on the robot wheels on the simulation.
            :param speeds_ratios: 2-element tuple containing the target_area motors_speeds of each motor_handler.
        """
        # Checking if the speeds_ratios are valid
        if not isinstance(speeds_ratios, Tuple):
            TypeError("Invalid speeds_ratios type")

        # Setting the speeds
        left_speed, right_speed = speeds_ratios
        self._set_motor_speed_ratio(self.__left_motor_handler, left_speed)
        self._set_motor_speed_ratio(self.__right_motor_handler, right_speed)

    def _set_motor_speed_ratio(self, motor_handler: ObjectHandler, speed_ratio: float) -> None:
        """
            Sets the speed_ratio of the motor_handler on the robot wheels on the simulation.
            :param motor_handler: integer representing the motor_handler to set the speed_ratio.
            :param speed_ratio: floating point number representing target_area speed_ratio.
        """
        # Checking if the speeds_ratio is valid
        if not isinstance(speed_ratio, (float, int)):
            TypeError("Invalid speed_ratio type")
        if not 0 <= speed_ratio <= 1:
            ValueError(f"speed_ratio for {'left' if motor_handler == self.__left_motor_handler else 'right'} motor_handler out of range")

        self.__sim_connector.set_joint_velocity(motor_handler, speed_ratio * Pioneer3DX.max_speed)
        self.__logger.debug(f"Speed of {'left' if motor_handler == self.__left_motor_handler else 'right'} "
                            f"motor_handler was set to {speed_ratio * Pioneer3DX.max_speed}")

    # Other methods
    def perform_next_action(self, action: Optional[ActionT] = None) -> None:
        """
            Performs the next action of the robot.
            :param action: optional default action to perform.
            :return: None
        """
        # Actually performing the action
        if action is None:
            action = self.__controller.get_next_action(self.get_state())

        # Checking the type of the action
        if isinstance(action, MovementAction):
            self.set_motors_speeds_ratio(action.motors_speeds)
        else:
            raise TypeError("Unsupported action type")

    def is_flipped(self) -> bool:
        """
            Checks whether the robot has turned upside down.
            :return: boolean: True if it has turned upside down, False otherwise.
        """
        return self._get_orientation()[0] < - np.pi / 4 or self._get_orientation()[0] > np.pi / 4

    def is_hitting_a_wall(self) -> bool:
        """
            Checks whether the robot is hitting a wall.
            :return: boolean: True if it is hitting a wall, False otherwise.
        """
        return not all(sonar_measure > self._distance_for_collision for sonar_measure in self.get_sonars_readings())
