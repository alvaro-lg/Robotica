import logging
import uuid
from pathlib import Path
from typing import Optional

import tensorflow as tf

from shared.data_types import AIModel


class ModelRepository:

    def __init__(self, base_dir: Path):
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')

        # Attribute checking
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory {base_dir} does not exist")
        if not base_dir.is_dir():
            raise FileExistsError(f"Base directory {base_dir} is not a directory")

        # Attributes initialization
        self.__base_dir = base_dir.resolve()

    def new_model(self, file_save: bool = False, name: Optional[str] = None) -> AIModel:
        """
            Creates a new AI model.
            :param file_save: boolean value indicating whether to save the model to a file or not.
            :param name: string representing the name of the model.
            :return: AIModel object representing the new model.
        """

        # Checking already existing model
        if name is not None and (self.__base_dir / name).exists():
            raise FileExistsError(f"File {self.__base_dir / name} already exists")

        # Creating a new model
        model = self.__new_model()

        # Saving the model to a file
        if file_save:
            self.save_model(model, name)

        return model

    def get_model(self, filename: str) -> AIModel:
        """
            Loads a model from a file.
            :param filename: string representing the name of the model file.
            :return: AIModel object representing the loaded model.
        """

        if (self.__base_dir / filename).is_dir() or not (self.__base_dir / filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist in {self.__base_dir}")

        # Loading the model from a file
        model = tf.keras.models.load_model(self.__base_dir / filename)
        return model

    def save_model(self, model: AIModel, name: Optional[str] = None) -> None:

        if name is None:
            name = f"{str(uuid.uuid4())}.keras"  # Random name if not provided

        # Saving the model to a file
        model.save(self.__base_dir / name)

    def __new_model(self) -> AIModel:

        # Input layer of the model
        input_ = tf.keras.layers.Input(shape=(1, 3), dtype=tf.float64, name="input")

        # Appending all the layers of the model in a list
        model_layers = [tf.keras.layers.Dense(32, activation='relu', name="dense_1"),
                        tf.keras.layers.Dense(32, activation='relu', name="dense_2"),
                        tf.keras.layers.Dropout(.75, input_shape=(32, 1), name="dropout_1"),
                        tf.keras.layers.Dense(16, activation='relu', name="s_dense_1"),
                        tf.keras.layers.Dense(8, activation='relu', name="s_dense_2"),
                        tf.keras.layers.Dropout(.5, input_shape=(8, 1), name="dropout_2"),
                        tf.keras.layers.Dense(4, activation='relu', name="xs_dense"),
                        tf.keras.layers.Dense(2, activation='relu', name="output")]

        # Composing all the layers of the model
        out_ = input_
        for layer in model_layers:
            out_ = layer(out_)

        # Building the actual model
        model = tf.keras.models.Model(inputs=input_, outputs=out_, name="ai_controller_model")
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(),
                      metrics=tf.metrics.Accuracy())

        # Logging
        self.__logger.info(f"New model created: {model.summary()}")

        return model
