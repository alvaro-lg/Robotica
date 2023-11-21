import logging
from copy import copy

import cv2
import numpy as np

from controller.domain.visual_controller import VisualController
from controller.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from controller.infrastructure.pioneer3DX_connector import Pioneer3DXConnector
from shared.exceptions import FlippedRobotException

# Constants
DEBUG = True
ROBOT_ID = "PioneerP3DX"
ROBOT_CONTROLLER = VisualController


class TrainingService:

    @staticmethod
    def run():

        # Variables initialization
        simulation: CoppeliaSimConnector = CoppeliaSimConnector()
        robot = Pioneer3DXConnector(simulation.sim, ROBOT_ID, ROBOT_CONTROLLER, use_camera=True)

        try:
            # Starting the simulation
            simulation.start_simulation()

            # Running the simulation
            while simulation.is_running():
                # Performing the next action
                robot.perform_next_action()

                # Getting the camera readings, contours and circle
                img = robot.get_camera_reading()
                img_contours = VisualController.get_image_contours(copy(img))
                img_circle = VisualController.get_image_min_circle(copy(img))

                # Stacking the images together
                img_final = cv2.hconcat([img, img_contours, img_circle])

                # Blending the image with the contours and displaying them
                img_final_s = cv2.resize(img_final, (768, 256), interpolation=cv2.INTER_AREA)
                cv2.imshow('Camera View', img_final_s)
                cv2.waitKey(1)

        except FlippedRobotException:
            # Stopping the simulation
            simulation.stop_simulation()

            # Restarting whole service
            TrainingService.run()

        except KeyboardInterrupt:
            # Stopping the simulation
            simulation.stop_simulation()

            # Variables destruction
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Setting logging format
    if DEBUG:
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('root')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    # Actual application startup
    TrainingService.run()
