import logging
from typing import Tuple, List

import numpy as np
import cv2


class Pioneer3DX:
    """
        Class that represents the Pioneer 3DX robot in the simulation.
    """

    # Class-static attributes
    num_sonar: int = 16
    sonar_max: int = 1.0

    def __init__(self, sim, robot_id: str, use_camera: bool = False, use_lidar: bool = False):
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info(f"Getting handles ({robot_id})")

        # Attributes initialization
        self.__sim = sim
        self.__left_motor: int = self.__sim.getObject(f'/{robot_id}/leftMotor')
        self.__right_motor: int = self.__sim.getObject(f'/{robot_id}/rightMotor')

        self.__sonars: List[int] = []
        for i in range(self.num_sonar):
            self.__sonars.append(self.__sim.getObject(f'/{robot_id}/ultrasonicSensor[{i}]'))

        if use_camera:
            self.__camera: int = self.__sim.getObject(f'/{robot_id}/camera')

        if use_lidar:
            self.__lidar: int = self.__sim.getObject(f'/{robot_id}/lidar')

    def _set_motors_speeds(self, speeds: Tuple[float, float]):
        """
            Sets the speed of the two motors on the robot wheels on the simulation.
            :param speeds: 2-element tuple containing the target speeds of each motor.
        """
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

    def get_sonars_readings(self) -> List[float]:
        """
            Retrieves a list with the readings of the sonars associated with ths robot.
            :return: a list containing the reading for each sensor.
        """
        # List to store readings
        readings = list()

        # Getting each reading
        for sonar in self.__sonars:
            res, dist, _, _, _ = self.__sim.readProximitySensor(sonar)
            readings.append(dist if res == 1 else self.sonar_max)

        return readings

    def get_camera_frame(self, contours: bool = False) -> np.ndarray:
        """
            Retrieves the image captured by the robot's camera at execution time.
            :return: a ndarray containing the data of the image captured.
        """
        if hasattr(self, "_Pioneer3DX__camera"):
            # Getting raw data of the image
            img, resX, resY = self.__sim.getVisionSensorCharImage(self.__camera)

            # Processing image data
            img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

            if not contours:
                return img
            else:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Gen lower mask (0-5) and upper mask (175-180) of RED
                mask1 = cv2.inRange(img_hsv, np.array([0, 50, 20]), np.array([5, 255, 255]))
                mask2 = cv2.inRange(img_hsv, np.array([175, 50, 20]), np.array([180, 255, 255]))

                # Merge the mask and crop the red regions
                mask = cv2.bitwise_or(mask1, mask2)
                cropped = cv2.bitwise_and(img, img, mask=mask)

                img_r = cropped[:, :, 0]  # Getting only red channel
                contours, _ = cv2.findContours(img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_final = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

                return img_final
        else:
            raise AttributeError("Undefined self.__camera attribute in __init__() method")

    def get_lidar_reading(self) -> List[float]:
        """
            Retrieving data of the lidar sensor on the robot.
        """
        if hasattr(self, "_Pioneer3DX__lidar"):
            data = self.__sim.getStringSignal('PioneerP3dxLidarData')
            if data is None:
                return []
            else:
                return self.__sim.unpackFloatTable(data)
        else:
            raise AttributeError("Undefined self.__lidar attribute in __init__() method")

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
