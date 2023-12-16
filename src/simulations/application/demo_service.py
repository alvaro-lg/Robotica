import logging
from copy import copy
from pathlib import Path
from typing import Optional

import cv2

from shared.application.exceptions import WallHitException, FlippedRobotException
from simulations.domain.controllers.visual_controller import VisualController
from simulations.domain.services.image_processing_service import ImageProcessingService
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
MODEL_NAME = None
MODELS_PATH = Path("models/Trained/V2")


class DemoService:

    @staticmethod
    def run(models_path: Optional[Path] = MODELS_PATH, model_name: Optional[str] = None):

        # Variables initialization
        simulation: CoppeliaSimConnector = CoppeliaSimConnector.get_instance()
        if model_name is None:
            robot = Pioneer3DX(simulation, VisualController(), ROBOT_ID)
            simulation.add_sim_element(robot)
        else:
            # TODO Implement AI Controller demo
            pass

        try:
            # Starting the simulation
            simulation.start_simulation()
            simulation.reset_simulation(shuffle=False)

            # Running the simulation
            while simulation.is_running():
                try:
                    # Performing the next action
                    robot.perform_next_action()

                    if DISPLAY:
                        # Getting the camera readings, contours and circle
                        img = robot.get_camera_reading()
                        _, img_mask = ImageProcessingService.get_contours(copy(img), ret_img_mask=True)
                        img_contours = ImageProcessingService.get_image_contours(copy(img))

                        # Stacking the images together
                        img_final = cv2.hconcat([img, img_mask, img_contours])

                        # Blending the image with the contours and displaying them
                        img_final_s = cv2.resize(img_final, (768, 256), interpolation=cv2.INTER_AREA)
                        cv2.imshow('Camera View', img_final_s)
                        cv2.waitKey(1)

                except WallHitException:
                    pass

            # Stopping the simulation
            simulation.stop_simulation()

            # Variables destruction
            cv2.destroyAllWindows()

        except FlippedRobotException:
            # Stopping the simulation
            simulation.stop_simulation()

            # Logging
            logger.exception("Robot flipped, restarting simulation...")

            # Restarting whole service
            DemoService.run()

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
    DemoService.run(model_name=MODEL_NAME)
