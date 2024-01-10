import logging

from copy import copy
from pathlib import Path
from typing import Optional

import cv2

from shared.application.exceptions import WallHitException, FlippedRobotException
from simulations.domain.controllers.visual_DQN_agent import VisualDQNAgent
from simulations.domain.controllers.visual_agent import VisualAgent
from simulations.domain.services.image_processing_service import ImageProcessingService
from simulations.domain.simulation_elements.pathway import Pathway
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from simulations.infrastructure.model_repository import ModelRepository

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
PATH_ID = "Path"
MODEL_NAME = "model_ep321"
MODELS_PATH = Path("models")


class DemoService:

    @staticmethod
    def run(models_path: Optional[Path] = MODELS_PATH, model_name: Optional[str] = None):

        # Variables initialization
        simulation: CoppeliaSimConnector = CoppeliaSimConnector.get_instance()
        if model_name is None:
            robot = Pioneer3DX(simulation, VisualAgent(), ROBOT_ID)
        else:
            repo = ModelRepository(models_path)
            model = repo.load_lite(model_name)
            robot = Pioneer3DX(simulation, VisualDQNAgent(model), ROBOT_ID)

        pathway = Pathway(simulation, PATH_ID)
        simulation.add_sim_elements([robot, pathway])

        try:
            # Starting the simulation
            simulation.start_simulation()
            simulation.reset_simulation(shuffle=True)

            # Running the simulation
            while simulation.is_running():
                try:
                    # Performing the next action
                    robot.perform_next_action()

                    # Resetting if proceeds
                    if robot.is_hitting_a_wall() or robot.is_flipped():
                        simulation.reset_simulation(shuffle=True)

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
            if DISPLAY:
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
