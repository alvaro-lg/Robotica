from typing import Sequence, Tuple

import cv2
import numpy as np

from shared.data_types import CameraReadingData


class ImageProcessingService:

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
        contours = ImageProcessingService.get_contours(img)

        # Drawing contours on top of image
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        return img

    @staticmethod
    def get_min_circle(img: CameraReadingData) -> Tuple[Sequence[int], int]:

        # Getting actual contours
        points = ImageProcessingService.get_contours(img)

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
        center, radius = ImageProcessingService.get_min_circle(img)
        cv2.circle(img, center, radius, (0, 255, 0), 2)

        return img
