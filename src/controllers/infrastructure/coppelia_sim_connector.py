import logging
import time
import random

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class CoppeliaSimConnector:
    """
        Class that connects to CoppeliaSim and provides the basic functions for starting, stopping and checking if the
        simulation is still running.
    """

    # Class-static attributes
    num_points: int = 16

    def __init__(self):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
        """

        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')
        self.__logger.info("Connecting to CoppeliaSim...")

        # Class attributes initialization
        client = RemoteAPIClient()
        self.__sim = client.getObject('sim')
        self.__idle_fps: int = self.__sim.getInt32Param(self.__sim.intparam_idle_fps)

    def start_simulation(self, shuffle: bool = True) -> None:
        """
            Starts the simulation in CoppeliaSim.
        """
        robot = self.__sim.getObject(f'/PioneerP3DX')
        self.__sim.setObjectPosition(robot, [0, 0, 0.15])
        self.__sim.setInt32Param(self.__idle_fps, 0)
        if shuffle:
            self.__logger.debug("Shuffling points...")
            self._shuffle_points()
        self.__logger.debug("Starting simulation...")
        self.__sim.startSimulation()

    def stop_simulation(self) -> None:
        """
            Stops the simulation.
        """
        self.__sim.stopSimulation()
        while self.is_running():
            time.sleep(0.1)
        self.__sim.setInt32Param(self.__sim.intparam_idle_fps, self.__idle_fps)
        self.__logger.debug("Simulation stopped...")

    def is_running(self) -> bool:
        """
            Checks whether the simulation is still running.
            :return: boolean: True if it is still running, False otherwise.
        """
        return self.__sim.getSimulationState() != self.__sim.simulation_stopped

    def _shuffle_points(self) -> None:
        """
            Shuffles the points of the path followed by the visual target in the simulation.
        """
        # Getting the points and their positions
        points = []
        positions = []
        for i in range(self.num_points):
            points.append(self.__sim.getObject(f'/Path/ctrlPt[{i}]'))
            positions.append(self.__sim.getObjectPosition(points[i]))

        # Shuffling the points
        random.shuffle(points)

        # Setting the new positions
        for point, pos in zip(points, positions):
            self.__sim.setObjectPosition(point, pos)

    # Properties
    def sim(self):
        """
            Getter for the sim private object.
        """
        return self.__sim

    def _sim(self, sim) -> None:
        """
            Setter for the sim private object.
            :param sim: new sim object to store.
        """
        self.__sim = sim

    sim = property(fget=sim, fset=_sim)
