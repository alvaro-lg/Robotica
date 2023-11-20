import logging
import cv2

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
        simulation._start_simulation()

        robot._set_motors_speeds((0, 0))
        # Running the simulation
        while simulation.is_running():
            img = robot._get_camera_frame()
            cv2.imshow('opencv', img)
            cv2.waitKey(1)

        # Stopping the simulation
        simulation._stop_simulation()

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
