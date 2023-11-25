from typing import TypeVar, List
import numpy as np
import tensorflow as tf

# Typing
RobotT = TypeVar('RobotT', bound='Pioneer3DXConnector')
ActionT = TypeVar('ActionT', bound='MovementAction')
RobotControllerT = TypeVar('RobotControllerT', bound='VisualController')
MaskT = TypeVar('MaskT', bound='RedBallMask')

# Custom types
AIModel = tf.keras.models.Model
CameraReadingData = np.ndarray
SonarsReadingsData = List[float]
LidarReadingData = List[float]
