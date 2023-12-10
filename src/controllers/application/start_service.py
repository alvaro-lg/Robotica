import logging
import os
from copy import copy
from pathlib import Path
from typing import Optional

import cv2

from controllers.domain.image_processing_service import ImageProcessingService
from controllers.domain.visual_AI_controller import VisualAIController
from controllers.domain.visual_controller import VisualController
from controllers.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from controllers.infrastructure.model_repository import ModelRepository
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
from shared.actions import MovementActionFactory
from shared.exceptions import FlippedRobotException

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
MODEL_NAME = "1000_90_bcde40f5-c3b4-4301-b32b-d5da0f54810f.keras"
MODELS_PATH = Path("models/Trained/V1")


class StartService:

    @staticmethod
    def run(models_path: Optional[Path] = MODELS_PATH, model_name: Optional[str] = None):

        # Variables initialization
        simulation: CoppeliaSimConnector = CoppeliaSimConnector()
        if model_name is None:
            robot = Pioneer3DXConnector(simulation.sim, ROBOT_ID, VisualController(), use_camera=True)
        else:
            robot = Pioneer3DXConnector(simulation.sim, ROBOT_ID,
                                        VisualAIController(ModelRepository(models_path).get_model(model_name)),
                                        use_camera=True)
            MovementActionFactory.create_action_space()

        try:
            # Starting the simulation
            simulation.start_simulation(shuffle_points=True)

            # Running the simulation
            while simulation.is_running():
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
            StartService.run()

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
    StartService.run(model_name=MODEL_NAME)
