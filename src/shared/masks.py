import cv2
import numpy as np

from shared.data_types import CameraReadingData


class RedBallMask:

    @staticmethod
    def mask_image(img: CameraReadingData) -> CameraReadingData:
        # Gen lower mask (0-5) and upper mask (175-180) of RED
        mask1 = cv2.inRange(img, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(img, np.array([170, 70, 50]), np.array([180, 255, 255]))

        # Merge the mask
        return cv2.bitwise_or(mask1, mask2)
