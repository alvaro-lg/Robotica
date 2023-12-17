from collections import deque
from copy import deepcopy
import random
from datetime import datetime
from typing import Optional, Any

import numpy as np
import tensorflow as tf

from shared.action_space import ActionSpace
from simulations.domain.controllers.visual_controller import VisualController
from shared.actions import MovementAction
from shared.data_types import AIModel, TransitionT
from shared.state import State

# Constants
REPLAY_MEMORY_SIZE = 3000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
DISCOUNT = 0.99  # Discount rate
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)


class VisualAIController(VisualController):
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
        self.__target_model: AIModel = deepcopy(self.__model)
        self.__replay_memory: deque = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.__target_update_counter: int = 0

    def update_replay_memory(self, transition: TransitionT) -> None:
        """
            Adds step's data to a memory replay array
            :param transition: Tuple of observation space, action, reward and new observation space
        """
        self.__replay_memory.append(transition)

    def get_prediction(self, state: State, model: Optional[AIModel] = None) -> Any:
        """
            Returns the next action to be performed by the robot.
            :param state: The current state of the robot.
            :param model: The model to be used by the controller.
            :return: The output of the model.
        """
        # Getting input for prediction
        x, y, area = state.x_norm, state.y_norm, state.area_norm
        input_data = np.array([[[x, y, area]]])

        # Predicting the output
        if model is None:
            outputs = self.__model.predict(input_data, verbose=0)
        else:
            outputs = model.predict(input_data, verbose=0)

        # Returning the corresponding outputs
        return outputs[0][0]

    def train(self, terminal_state: bool) -> None:
        """
            Trains main network every step during episode
            :param terminal_state: State of the robot when the episode ended.
        """
        # Start training only if certain number of samples is already saved
        if len(self.__replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.__replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([state for state, _, _, _, _ in minibatch])
        current_qs_list = np.array([self.get_prediction(state) for state in current_states])

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array([state for _, _, _, state, _ in minibatch])
        future_qs_list = np.array([self.get_prediction(state, model=self.__target_model)
                                   for state in new_current_states])

        X, y, = [], []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[ActionSpace.get_instance().actions.index(action)] = new_q

            # And append to our training data
            X.append([[current_state.x_norm, current_state.y_norm, current_state.area_norm]])
            y.append([current_qs])

        # Fit on all samples as one batch, log only on terminal state
        self.__model.fit(x=np.array(X), y=np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

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
