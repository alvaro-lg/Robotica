from abc import ABC, abstractmethod


class SimulationConnector(ABC):
    """
        Class that represents the interface of a connector to the simulation.
    """

    @abstractmethod
    def start_simulation(self) -> None:
        """
            Starts the simulation.
        """
        pass

    @abstractmethod
    def reset_simulation(self) -> None:
        """
            Starts the simulation.
        """
        pass

    @abstractmethod
    def stop_simulation(self) -> None:
        """
            Stops the simulation.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
            Checks whether the simulation is still running.
            :return: boolean: True if it is still running, False otherwise.
        """
        pass

    @abstractmethod
    def get_time(self) -> float:
        """
            Returns the current simulation time.
            :return: float: current simulation time.
        """
        pass
