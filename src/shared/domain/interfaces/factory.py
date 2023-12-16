from abc import ABC, abstractmethod

from shared.domain.data_types import Buildable


class Factory(ABC):
    """
        Class that represents the interface of a factory.
    """

    @staticmethod
    @abstractmethod
    def new() -> Buildable:
        """
            Creates a new object.
            :return: Buildable built object.
        """
        pass
