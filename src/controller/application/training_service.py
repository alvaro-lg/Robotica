import logging
import cv2
import numpy as np

from controller.infrastructure.coppelia_sim_service import CoppeliaSimService
from shared.pioneer3DX import Pioneer3DX

ROBOT_ID = "PioneerP3DX"


class TrainingService:

    @staticmethod
    def run():

        # Variables initialization
        simulation: CoppeliaSimService = CoppeliaSimService()
        robot = Pioneer3DX(simulation._sim, ROBOT_ID, use_camera=True)

        # Starting the simulation
        simulation.start_simulation()

        # Running the simulation
        while simulation.is_running():
            img = robot.get_camera_frame(contours=True)
            cv2.imshow('opencv', img)
            cv2.waitKey(1)

        # Stopping the simulation
        simulation.stop_simulation()

        # Variables destruction
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Setting logging format
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Actual application startup
    TrainingService.run()
