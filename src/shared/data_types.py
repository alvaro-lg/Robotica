from typing import TypeVar, List, Union
import numpy as np
import tensorflow as tf
import keras

from shared.actions import MovementAction

# Type hinting
RobotT = TypeVar('RobotT', bound='Pioneer3DX')
ActionT = TypeVar('ActionT', bound='MovementAction')
StateT = TypeVar('StateT', bound='State')
RobotControllerT = TypeVar('RobotControllerT', bound='VisualController')
MaskT = TypeVar('MaskT', bound='RedBallMask')

# Custom types
Transition = np.dtype([
    ('curr_state', np.float64, (2,)),   # StateT
    ('action', MovementAction),         # MovementAction
    ('reward', np.float64),             # float
    ('next_state', np.float64, (2,)),   # StateT
    ('done',  np.bool_)                 # bool
])
AIModel = Union[keras.models.Model, bytes]
CameraReadingData = np.ndarray
SonarsReadingsData = List[float]
LidarReadingData = List[float]
