import logging
from datetime import datetime
from typing import Tuple

import keras
import tensorflow as tf

from shared.data_types import AIModel
from shared.domain.interfaces.factory import Factory
from shared.action_space import ActionSpace


class ModelFactory(Factory):
    """
        Class that represents the factory for AI models.
    """

    # Static attributes
    input_shape: Tuple[int] = (1, 2)
    __logger: logging.Logger = logging.getLogger('root')

    @staticmethod
    def new() -> AIModel:
        """
            Creates a new AI model.
            :return: AIModel object representing the new model.
        """

        # Input layer of the model
        input_ = keras.layers.Input(shape=ModelFactory.input_shape, dtype=tf.float64, name="input")

        # Appending all the layers of the model in a list
        model_layers = [keras.layers.Dense(8, activation='relu', name="dense_1"),
                        keras.layers.Dense(len(ActionSpace.get_instance().actions), activation='linear', name='output')]

        # Composing all the layers of the model
        out_ = input_
        for layer in model_layers:
            out_ = layer(out_)

        # Building the actual model
        model = keras.models.Model(inputs=input_, outputs=out_, name=datetime.now().strftime("%Y%m%d-%H%M%S"))
        model.compile(keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(),
                      metrics=keras.metrics.Accuracy())

        # Logging
        ModelFactory.__logger.info(f"New model created: {model.summary()}")

        return model
