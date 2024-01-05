from typing import Sequence

from simulations.domain.services.image_processing_service import ImageProcessingService
from shared.data_types import CameraReadingData


class State(Sequence):

    # Class-static attributes
    n_features: int = 2
    __last_state: 'State' = None

    def __init__(self, camera_reading: CameraReadingData):

        # Getting contours
        img = camera_reading
        contours = ImageProcessingService.get_contours(img)

        if len(contours) > 0:  # Sphere in sight
            # Extract x-coordinate of the circle center
            (center_x, _), area = ImageProcessingService.get_shape(img)
            len_x, len_y = img.shape[0], img.shape[1]

            # Calculating values
            self.__x_norm: float = center_x / len_x
            self.__area_norm: float = area / (len_x * len_y)

            self.__ball_in_sight: bool = True  # For speeding up checkings
        else:  # Sphere out of sight

            # Values for ball out of sight
            self.__x_norm: float = 0.0 if (State.__last_state is not None and
                                           State.__last_state.x_norm < 0.5) else 1.0  # Closer x boind to previous state
            self.__area_norm: float = 0.0

            self.__ball_in_sight: bool = False  # For speeding up checkings

        # Updating last state
        State.__last_state = self

    @property
    def x_norm(self) -> float:
        """
            Returns the normalized x-coordinate of the ball.
            :return: float representing the normalized x-coordinate of the ball.
        """
        return self.__x_norm

    @property
    def area_norm(self) -> float:
        """
            Returns the normalized area of the ball.
            :return: float representing the normalized area of the ball.
        """
        return self.__area_norm

    def is_ball_in_sight(self) -> bool:
        return self.__ball_in_sight

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.__x_norm
        elif index == 1:
            return self.__area_norm

    def __iter__(self):
        return iter([self.__x_norm, self.__area_norm])
