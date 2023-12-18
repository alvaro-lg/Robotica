import logging
from datetime import datetime
from typing import Tuple
from uuid import uuid4

import tensorflow as tf

from shared.data_types import AIModel
from shared.domain.interfaces.factory import Factory
from shared.action_space import ActionSpace


class ModelFactory(Factory):
    """
        Class that represents the factory for AI models.
    """

    # Static attributes
    input_shape: Tuple[int] = (1, 3)
    __logger: logging.Logger = logging.getLogger('root')

    @staticmethod
    def new() -> AIModel:
        """
            Creates a new AI model.
            :return: AIModel object representing the new model.
        """

        # Input layer of the model
        input_ = tf.keras.layers.Input(shape=ModelFactory.input_shape, dtype=tf.float64, name="input")

        # Appending all the layers of the model in a list
        model_layers = [tf.keras.layers.Dense(64, activation='relu', name="dense_1"),
                        tf.keras.layers.Dense(64, activation='relu', name="dense_2"),
                        tf.keras.layers.Dropout(.75, input_shape=(64, 1), name="dropout_1"),
                        tf.keras.layers.Dense(32, activation='relu', name="s_dense_1"),
                        tf.keras.layers.Dense(32, activation='relu', name="s_dense_2"),
                        tf.keras.layers.Dropout(.5, input_shape=(32, 1), name="dropout_2"),
                        tf.keras.layers.Dense(16, activation='relu', name="xs_dense_1"),
                        tf.keras.layers.Dense(16, activation='relu', name="xs_dense_2"),
                        tf.keras.layers.Dense(len(ActionSpace.get_instance().actions), activation='linear',
                                              name='output')]

        # Composing all the layers of the model
        out_ = input_
        for layer in model_layers:
            out_ = layer(out_)

        # Building the actual model
        model = tf.keras.models.Model(inputs=input_, outputs=out_, name=datetime.now().strftime("%Y%m%d-%H%M%S"))
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(),
                      metrics=tf.metrics.Accuracy())

        # Logging
        ModelFactory.__logger.info(f"New model created: {model.summary()}")

        return model
