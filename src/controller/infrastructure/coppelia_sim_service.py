import logging
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class CoppeliaSimService:
    """
        Class that connects to CoppeliaSim and provides the basic functions for starting, stopping and checking if the
        simulation is still running.
    """

    def __init__(self):
        """
            Performs initialization of class attributes and connection to CoppeliaSim.
        """
        # Debugging
        self.__logger = logging.getLogger('root')
        self.__logger.info("Connecting to CoppeliaSim...")

        # Class attributes initialization
        client = RemoteAPIClient()
        self.__sim = client.getObject('sim')
        self.__idle_fps: int = self.__sim.getInt32Param(self.__sim.intparam_idle_fps)

    def _start_simulation(self) -> None:
        """
            Starts the simulation in CoppeliaSim.
        """
        self.__logger.debug("Starting simulation...")
        self.__sim.setInt32Param(self.__idle_fps, 0)
        self.__sim.startSimulation()

    def _stop_simulation(self) -> None:
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

    # Properties
    @property
    def _sim(self):
        """
            Getter for the sim private object.
        """
        return self.__sim

    @_sim.setter
    def _sim(self, sim) -> None:
        """
            Setter for the sim private object.
            :param sim: new sim object to store.
        """
        self.__sim = sim
