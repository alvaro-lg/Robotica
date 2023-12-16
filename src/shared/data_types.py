from typing import TypeVar, List, Tuple
import numpy as np
import tensorflow as tf


# Type hinting
RobotT = TypeVar('RobotT', bound='Pioneer3DX')
ActionT = TypeVar('ActionT', bound='MovementAction')
StateT = TypeVar('StateT', bound='State')
RobotControllerT = TypeVar('RobotControllerT', bound='VisualController')
MaskT = TypeVar('MaskT', bound='RedBallMask')
TransitionT = Tuple[StateT, ActionT, float, StateT, bool]

# Custom types
AIModel = tf.keras.models.Model
CameraReadingData = np.ndarray
SonarsReadingsData = List[float]
LidarReadingData = List[float]
