from typing import TypeVar, List
import numpy as np

# Typing
RobotT = TypeVar('RobotT', bound='Pioneer3DXConnector')
ActionT = TypeVar('ActionT', bound='MovementAction')
RobotControllerT = TypeVar('RobotControllerT', bound='VisualController')

# Custom types
CameraReadingData = np.ndarray
SonarsReadingsData = List[float]
LidarReadingData = List[float]
