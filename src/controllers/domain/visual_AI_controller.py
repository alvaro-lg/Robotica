from collections import deque
from copy import deepcopy
import random
from typing import Tuple, Optional

import numpy as np

from controllers.domain.visual_controller import VisualController
from controllers.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction
from shared.data_types import AIModel, TransitionT
from shared.state import State

# Constants
REPLAY_MEMORY_SIZE = 3000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
DISCOUNT = 0.99  # Discount rate
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)


class VisualAIController(VisualController):

    def __init__(self, model: AIModel):
        super().__init__()

        # Model initialization
        self.__model: AIModel = model

        # Target network initialization
        self.__target_model: AIModel = deepcopy(self.__model)

        # Replay memory initialization
        self.__replay_memory: deque = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.__target_update_counter: int = 0

        # TODO: Tensorboard
    def update_replay_memory(self, transition: TransitionT) -> None:
        """
            Adds step's data to a memory replay array
            :param transition: Tuple of observation space, action, reward and new observation space
        """
        self.__replay_memory.append(transition)

    def get_next_action(self, state: State, model: Optional[AIModel] = None) -> MovementAction:
        """
            Returns the next action to be performed by the robot.
            :param state:
            :param model:
            :return:
        """

        # Getting input for prediction
        x, y, area = state.x_norm, state.y_norm, state.area_norm
        input_data = np.array([[[x, y, area]]])

        # Predicting the output
        if model is None:
            left_speed_ratio, right_speed_ratio = self.__model.predict(input_data)[0][0]
        else:
            left_speed_ratio, right_speed_ratio = model.predict(input_data)[0][0]
        left_speed, right_speed = (left_speed_ratio * Pioneer3DXConnector.max_speed,
                                   right_speed_ratio * Pioneer3DXConnector.max_speed)

        # Returning the corresponding action
        return MovementAction((left_speed, right_speed))

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
        current_qs_list = np.array([self.get_next_action(state) for state in current_states])

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array([state for _, _, _, state, _ in minibatch])
        future_qs_list = np.array([self.get_next_action(state, model=self.__target_model) for state in current_states])

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
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.__model.fit(x=np.array(X), y=np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.__target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.__target_update_counter > UPDATE_TARGET_EVERY:
            self.__target_model.set_weights(self.__model.get_weights())
            self.__target_update_counter = 0

    # Properties
    @property
    def model(self) -> AIModel:
        return self.__model
