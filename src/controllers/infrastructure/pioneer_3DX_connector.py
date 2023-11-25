import cv2
import logging
import numpy as np

from typing import Tuple, List
from shared.actions import MovementAction
from shared.data_types import CameraReadingData, SonarsReadingsData, LidarReadingData, RobotControllerT, ActionT
from shared.exceptions import FlippedRobotException
from shared.state import State


class Pioneer3DXConnector:
    """
        Class that represents the Pioneer 3DX robot in the simulation.
    """

    # Class-static attributes
    num_sonar: int = 16
    max_sonar: int = 1
    max_speed: float = 3.

    def __init__(self, sim, robot_id: str, controller: RobotControllerT, use_camera: bool = False,
                 use_lidar: bool = False):
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info(f"Getting handles ({robot_id})")

        # Attributes initialization
        self.__sim = sim
        self.__robot_id: str = robot_id
        self.__left_motor: int = self.__sim.getObject(f'/{robot_id}/leftMotor')
        self.__right_motor: int = self.__sim.getObject(f'/{robot_id}/rightMotor')
        self.__controller: RobotControllerT = controller

        self.__sonars: List[int] = []
        for i in range(self.num_sonar):
            self.__sonars.append(self.__sim.getObject(f'/{robot_id}/ultrasonicSensor[{i}]'))

        if use_camera:
            self.__camera: int = self.__sim.getObject(f'/{robot_id}/camera')

        if use_lidar:
            self.__lidar: int = self.__sim.getObject(f'/{robot_id}/lidar_reading')

    def _set_motors_speeds(self, speeds: Tuple[float, float]):
        """
            Sets the speed of the two motors on the robot wheels on the simulation.
            :param speeds: 2-element tuple containing the target motors_speeds of each motor.
        """
        # Checking if the speeds are valid
        if all(isinstance(s, (float, int)) for s in speeds):
            left_speed, right_speed = speeds
            self.__sim.setJointTargetVelocity(self.__left_motor, left_speed)
            self.__sim.setJointTargetVelocity(self.__right_motor, right_speed)
            self.__logger.debug(f"Speeds of left and right motors were set to {left_speed} and {right_speed}")

    def _set_lmotor_speeds(self, speed: float):
        """
            Sets the speed of the left motor on the robot wheels on the simulation.
            :param speed: floating point number representing target speed.
        """
        self.__sim.setJointTargetVelocity(self.__left_motor, speed)
        self.__logger.debug(f"Speed of left motor was set to {speed}")

    def _set_rmotor_speeds(self, speed: float):
        """
            Sets the speed of the right motor on the robot wheels on the simulation.
            :param speed: floating point number representing target speed.
        """
        self.__sim.setJointTargetVelocity(self.__right_motor, speed)
        self.__logger.debug(f"Speed of right motor was set to {speed}")

    def get_motors_speeds(self) -> Tuple[float, float]:
        """
            Gets the speed of the two motors on the robot wheels on the simulation.
            :return: a tuple of two floating point numbers, containing the actual values of each motor.
        """
        left_speed = self.__sim.getJointTargetVelocity(self.__left_motor)
        right_speed = self.__sim.getJointTargetVelocity(self.__right_motor)
        return left_speed, right_speed

    def get_lmotor_speeds(self) -> float:
        """
            Gets the speed of the left motor on the robot wheels on the simulation.
            :return: floating point number representing actual speed of the left motor.
        """
        return self.__sim.getJointTargetVelocity(self.__left_motor)

    def get_rmotor_speeds(self) -> float:
        """
            Gets the speed of the right motor on the robot wheels on the simulation.
            :return: floating point number representing actual speed of the right motor.
        """
        return self.__sim.getJointTargetVelocity(self.__right_motor)

    def get_sonars_readings(self) -> SonarsReadingsData:
        """
            Retrieves a list with the readings of the sonars_readings associated with ths robot.
            :return: a list containing the reading for each sensor.
        """
        # List to store readings
        readings = list()

        # Getting each reading
        for sonar in self.__sonars:
            res, dist, _, _, _ = self.__sim.readProximitySensor(sonar)
            readings.append(dist if res == 1 else self.max_sonar)

        return readings

    def get_camera_reading(self) -> CameraReadingData:
        """
            Retrieves the image captured by the robot's camera at execution time and also the contour.
            :return: a ndarray containing the data of the image captured (and optionally another one containing the
            contours).
        """
        if hasattr(self, "_Pioneer3DXConnector__camera"):
            # Getting raw data of the image
            img, resX, resY = self.__sim.getVisionSensorCharImage(self.__camera)

            # Processing image data
            img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

            return img
        else:
            raise AttributeError("Undefined self.__camera attribute in __init__() method")

    def get_lidar_reading(self) -> LidarReadingData:
        """
            Retrieving data of the lidar_reading sensor on the robot.
        """
        if hasattr(self, "_Pioneer3DX__lidar"):
            data = self.__sim.getStringSignal('PioneerP3dxLidarData')
            if data is None:
                return []
            else:
                return self.__sim.unpackFloatTable(data)
        else:
            raise AttributeError("Undefined self.__lidar attribute in __init__() method")

    def perform_next_action(self) -> None:
        """
            Builds up the state and performs the next action in the controllers.
        """
        # Getting the rotation of the robot and checking if it has turned upside down
        if self.get_rotation()[0] < - np.pi / 4 or self.get_rotation()[0] > np.pi / 4:
            raise FlippedRobotException()

        # Building the state
        curr_state = State(self.get_camera_reading())

        # Actually performing the action
        self._perform_action(self._controller.get_next_action(curr_state))

    def _perform_action(self, action: ActionT) -> None:
        """
            Performs the action received as parameter.
            :param action: integer representing the action to perform.
        """
        if isinstance(action, MovementAction):
            self._set_motors_speeds(action.motors_speeds())
        else:
            raise TypeError("Unsupported action type")

    def get_rotation(self) -> Tuple[float, float, float]:
        """
            Retrieves the rotation of the robot.
            :return: a tuple containing the rotation of the robot on the three axis.
        """
        return self.__sim.getObjectOrientation(self.__sim.getObject(f'/{self.__robot_id}'))

    def get_position(self) -> Tuple[float, float, float]:
        """
            Retrieves the position of the robot.
            :return: a tuple containing the position of the robot on the three axis.
        """
        return self.__sim.getObjectPosition(self.__sim.getObject(f'/{self.__robot_id}'))

    # Properties
    @property
    def _sim(self):
        """
            Getter for the sim private object.
        """
        return self.__sim

    @_sim.setter
    def _sim(self, sim) -> None:
        """
            Setter for the sim private object.
            :param sim: new sim object to store.
        """
        self.__sim = sim

    @property
    def robot_id(self):
        """
            Getter for the sim private object.
        """
        return self.__robot_id

    @property
    def _controller(self) -> RobotControllerT:
        """
            Getter for the controllers private object.
        """
        return self.__controller

    @_controller.setter
    def _controller(self, controller: RobotControllerT) -> None:
        """
            Setter for the controllers private object.
            :param controller: new controllers object to store.
        """
        self.__controller = controller
