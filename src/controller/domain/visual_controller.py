from typing import Sequence, Tuple

import cv2
import numpy as np

from controller.infrastructure.pioneer3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction
from shared.data_types import CameraReadingData
from shared.state import State


class VisualController:

    @staticmethod
    def get_next_action(state: State) -> MovementAction:

        # Getting contours
        img = state.camera_reading
        contours = VisualController.get_contours(img)

        if len(contours) > 0:
            # Extract x-coordinate of the circle center
            center, _ = VisualController.get_min_circle(img)
            center_x = center[0]
            len_x = img.shape[0]

            # Calculate the speed of each wheel
            left_speed = Pioneer3DXConnector.max_speed * (center_x / len_x)
            right_speed = Pioneer3DXConnector.max_speed * ((len_x - center_x) / (len_x / 2))

            return MovementAction((left_speed, right_speed))
        else:
            return MovementAction((-Pioneer3DXConnector.max_speed / 2, Pioneer3DXConnector.max_speed / 2))

    @staticmethod
    def get_contours(img: CameraReadingData) -> Sequence[np.ndarray]:

        # To HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Gen lower mask (0-5) and upper mask (175-180) of RED
        mask1 = cv2.inRange(img_hsv, np.array([0, 50, 20]), np.array([5, 255, 255]))
        mask2 = cv2.inRange(img_hsv, np.array([175, 50, 20]), np.array([180, 255, 255]))

        # Merge the mask and crop the red regions
        mask = cv2.bitwise_or(mask1, mask2)
        cropped = cv2.bitwise_and(img, img, mask=mask)

        img_r = cropped[:, :, 0]  # Getting only red channel
        contours, _ = cv2.findContours(img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    @staticmethod
    def get_image_contours(img: CameraReadingData) -> CameraReadingData:

        # Getting actual contours
        contours = VisualController.get_contours(img)

        # Drawing contours on top of image
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        return img

    @staticmethod
    def get_min_circle(img: CameraReadingData) -> Tuple[Sequence[int], int]:

        # Getting actual contours
        points = VisualController.get_contours(img)

        if len(points) > 0:
            # Convert points to numpy array format
            points = np.concatenate(points).reshape(-1, 2)

            # Find minimum enclosing circle
            center, radius = cv2.minEnclosingCircle(points)

            return tuple(map(int, center)), int(radius)
        else:
            return (-10, -10), 0

    @staticmethod
    def get_image_min_circle(img: CameraReadingData) -> CameraReadingData:

        # Calculating and drawing minimum circle
        center, radius = VisualController.get_min_circle(img)
        cv2.circle(img, center, radius, (0, 255, 0), 2)

        return img
