from controllers.domain.image_processing_service import ImageProcessingService
from shared.data_types import CameraReadingData


class State:

    def __init__(self, camera_reading: CameraReadingData):

        # Getting contours
        img = camera_reading
        contours = ImageProcessingService.get_contours(img)

        if len(contours) > 0:  # Ball in sight
            # Extract x-coordinate of the circle center
            (center_x, center_y), area = ImageProcessingService.get_shape(img)
            len_x, len_y = img.shape[0], img.shape[1]

            # Calculating values
            self.__x_norm: float = center_x / len_x
            self.__y_norm: float = center_y / len_y
            self.__area_norm: float = area / (len_x * len_y)

            self.__ball_in_sight: bool = True  # For speeding up checkings
        else:  # Ball out of sight

            # Values for ball out of sight
            self.__x_norm: float = -1.0
            self.__y_norm: float = -1.0
            self.__area_norm: float = 0.0

            self.__ball_in_sight: bool = False  # For speeding up checkings

    @property
    def x_norm(self) -> float:
        return self.__x_norm

    @property
    def y_norm(self) -> float:
        return self.__y_norm

    @property
    def area_norm(self) -> float:
        return self.__area_norm

    def is_ball_in_sight(self) -> bool:
        return self.__ball_in_sight


