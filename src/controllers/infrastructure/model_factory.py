import logging
from typing import Optional

from shared.data_types import AIModel

import tensorflow as tf


class ModelFactory:

    def __init__(self):
        # Debugging
        self.__logger: logging.Logger = logging.getLogger('root')

    def new_model(self) -> AIModel:
        """
            Creates a new AI model.
            :return: AIModel object representing the new model.
        """

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
                        tf.keras.layers.Dense(2, activation='tanh', name="output")]

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
