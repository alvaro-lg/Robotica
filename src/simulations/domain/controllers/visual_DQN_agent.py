from collections import deque
from copy import copy, deepcopy
import random
from pathlib import Path
from typing import Optional, Any, Deque

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from shared.action_space import ActionSpace
from simulations.domain.controllers.visual_agent import VisualAgent
from shared.actions import MovementAction
from shared.data_types import AIModel, Transition
from shared.state import State
from simulations.infrastructure.model_repository import ModelRepository

# Constants
REPLAY_MEMORY_SIZE = 3000       # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000   # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128            # How many steps (samples) to use for training
DISCOUNT = 0.99                 # Discount rate
UPDATE_TARGET_EVERY = 20        # Terminal states (end of episodes)


class VisualDQNAgent(VisualAgent):
    """
        Class that implements the AI controller for the visual simulation.
    """

    def __init__(self, model: AIModel):
        """
            Constructor.
            :param model: The model to be used by the controller.
        """
        # Variables initialization
        super().__init__()
        self.__model: AIModel = model
        self.__target_model: AIModel = copy(self.__model)
        self.__replay_memory: Deque[Transition] = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.__target_update_counter: int = 0

    def update_replay_memory(self, transition: Transition) -> None:
        """
            Adds step's data to a memory replay array
            :param transition: Tuple of observation space, action, reward and new observation space
        """
        self.__replay_memory.append(transition)

    def get_prediction(self, state: State, model: Optional[AIModel] = None) -> Any:
        """
            Returns the next action to be performed by the robot.
            :param state: The current states of the robot.
            :param model: The model to be used by the controller.
            :return: The output of the model.
        """
        # Getting input for prediction
        x, area = state.x_norm, state.area_norm
        input_data = np.array([[[x, area]]]).astype(np.float64)

        # Predicting the output
        if model is None:
            model = self.__model

        if isinstance(model, keras.models.Model):
            outputs = model.predict(input_data, verbose=0)
        else:
            # Create a new interpreter with the same configuration
            interpreter = tf.lite.Interpreter(model_content=model)

            # Allocate tensors for the new interpreter
            interpreter.allocate_tensors()

            # For TensorFlow Lite model prediction
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Perform inference
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(input_data))
            interpreter.invoke()

            # Get the output tensor
            outputs = interpreter.get_tensor(output_details[0]['index'])

        # Returning the corresponding outputs
        return outputs[0][0]

    def _get_predictions(self, states: npt.NDArray, model: Optional[AIModel] = None) -> Any:
        """
            Returns the next actions to be performed by the robot.
            :param states: The states of the robot.
            :param model: The model to be used by the controller.
            :return: The outputs of the model.
        """
        # Predicting the output
        if model is None:
            model = self.__model

        if isinstance(model, keras.models.Model):
            outputs = model.predict(states, verbose=0)
        else:
            raise NotImplementedError("This method is not implemented for this model type")

        # Returning the outputs
        return outputs

    def train(self, terminal_state: bool, log_dir: str = None) -> None:
        """
            Trains main network every step during episode
            :param terminal_state: State of the robot when the episode ended.
            :param log_dir: The directory where the logs will be saved.
        """
        # Start training only if certain number of samples is already saved
        if len(self.__replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = np.vstack(random.sample(self.__replay_memory, MINIBATCH_SIZE))

        # Get current states from minibatch, then query NN model for Q values
        curr_states = minibatch[:]['curr_state']
        current_qs_list = self._get_predictions(curr_states)

        # Get future states from minibatch, then query NN model for Q values
        next_states = minibatch[:]['next_state']
        future_qs_list = self._get_predictions(next_states)

        # Calculating new Q values
        max_future_qs = np.max(future_qs_list, axis=2)
        new_qs_list = minibatch[:]['reward'] + np.vectorize(int)(minibatch[:]['done']) * (DISCOUNT * max_future_qs)
        actions_taken_idxs = np.vectorize(ActionSpace.get_instance().actions.index)(minibatch[:]['action'][0])

        # Indexing the current Q values with the actions taken
        rows = np.arange(current_qs_list.shape[0])[:, None]
        cols = np.zeros((current_qs_list.shape[0], 1), dtype=int)
        actions_idxs = actions_taken_idxs[:, None]

        # Updating the current Q values with the new Q values
        current_qs_list[rows, cols, actions_idxs] = new_qs_list

        if log_dir is not None:
            callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
        else:
            callbacks = []

        # Fit on all samples as one batch, logs only on terminal states
        self.__model.fit(x=curr_states, y=current_qs_list, batch_size=MINIBATCH_SIZE, verbose=0,
                         shuffle=False,
                         callbacks=callbacks)

        # Update target_area network counter every episode
        if terminal_state:
            self.__target_update_counter += 1

        # If counter reaches set value, update target_area network with weights of main network
        if self.__target_update_counter > UPDATE_TARGET_EVERY:
            self.__target_model.set_weights(self.__model.get_weights())
            self.__target_update_counter = 0

    def get_next_action(self, state: State) -> MovementAction:
        return ActionSpace.get_instance().actions[np.argmax(self.get_prediction(state))]

    # Properties
    @property
    def model(self) -> AIModel:
        return self.__model
