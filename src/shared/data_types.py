from typing import TypeVar, List
import numpy as np

RobotT = TypeVar('RobotT', bound='Pioneer3DXConnector')
ActionT = TypeVar('ActionT', bound='MovementAction')
RobotControllerT = TypeVar('RobotControllerT', bound='VisualController')

CameraReadingData = np.ndarray
SonarsReadingsData = List[float]
LidarReadingData = List[float]
