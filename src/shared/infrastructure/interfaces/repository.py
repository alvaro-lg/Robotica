from abc import ABC, abstractmethod
from typing import Optional

from shared.infrastructure.data_types import Storable


class Repository(ABC):
    """
        Class that represents the interface of a repository for data.
    """

    @abstractmethod
    def load(self, filename: str) -> Storable:
        """
            Loads a model from a file.
            :param filename: string representing the filename of the model file.
            :return: AIModel object representing the loaded model.
        """
        pass

    @abstractmethod
    def store(self, data: Storable, filename: Optional[str] = None) -> None:
        """
            Stores data to a file.
            :param data: Storable object representing the data to store.
            :param filename: string representing the filename of the model file.
        """
        pass
