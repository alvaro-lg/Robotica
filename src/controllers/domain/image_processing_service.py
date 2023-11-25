from typing import Sequence, Tuple, Union

import cv2
import numpy as np

from shared.data_types import CameraReadingData, MaskT
from shared.masks import RedBallMask


class ImageProcessingService:

    @staticmethod
    def get_contours(img: CameraReadingData, mask: MaskT = RedBallMask, ret_img_mask: bool = False) \
            -> Union[Sequence[np.ndarray], Tuple[Sequence[np.ndarray], CameraReadingData]]:

        # To HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masking the image
        img_masked = mask.mask_image(img_hsv)

        # Cropping the red regions
        contours, _ = cv2.findContours(img_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Getting the biggest contour
        if len(contours) > 0:
            ret = max(contours, key=cv2.contourArea)
        else:
            ret = []

        # Adding mask to return
        if ret_img_mask:
            ret = ret, cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)

        return ret

    @staticmethod
    def get_image_contours(img: CameraReadingData) -> CameraReadingData:

        # Getting actual contours
        contours = ImageProcessingService.get_contours(img)

        # Drawing contours on top of image
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        return img

    @staticmethod
    def get_shape(img: CameraReadingData) -> Tuple[Sequence[float], float]:

        # Getting actual contours
        contours = ImageProcessingService.get_contours(img)

        if len(contours) > 0:
            # Convert points to numpy array format
            contours = np.concatenate(contours).reshape(-1, 2)

            # Find center and area of the shape described by contours
            M = cv2.moments(contours)
            area = M['m00']

            # No circle
            if area <= 0:
                return (-10, -10), 0

            center = M['m10'] / area, M['m01'] / area
            return center, area

        else:
            # No circle
            return (-10, -10), 0
