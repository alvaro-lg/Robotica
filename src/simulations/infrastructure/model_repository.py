import logging
import uuid
from pathlib import Path
from typing import Optional

import tensorflow as tf

from shared.data_types import AIModel
from shared.infrastructure.interfaces.repository import Repository


class ModelRepository(Repository):
    """
        Class that represents the repository for models.
    """

    def __init__(self, base_dir: Path):
        """
            Constructor method.
            :param base_dir: Path object representing the base directory of the repository.
        """
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')

        # Attribute checking
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory {base_dir} does not exist")
        if not base_dir.is_dir():
            raise FileExistsError(f"Base directory {base_dir} is not a directory")

        # Attributes initialization
        self.__base_dir: Path = base_dir.resolve()

    def load(self, filename: str) -> AIModel:
        """
            Loads a model from a file.
            :param filename: string representing the filename of the model file.
            :return: AIModel object representing the loaded model.
        """

        if (self.__base_dir / filename).is_dir() or not (self.__base_dir / filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist in {self.__base_dir}")

        # Loading the model from a file
        model = tf.keras.models.load_model(self.__base_dir / filename)
        return model

    def store(self, model: AIModel, name: Optional[str] = None) -> None:
        """
            Stores a model to a file.
            :param model: AIModel object representing the model to store.
            :param name: string representing the filename of the model file.
        """
        # Figuring out the filename of the file
        if name is None:
            name = f"{str(uuid.uuid4())}.keras"  # Random filename if not provided
        else:
            # Checking already existing model
            if name is not None and (self.__base_dir / name).exists():
                raise FileExistsError(f"File {self.__base_dir / name} already exists")

        # Saving the model to a file
        model.save(self.__base_dir / name)
