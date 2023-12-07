import logging
from pathlib import Path
from random import choice

import numpy as np
from tqdm import tqdm

import cv2
import math
from controllers.domain.reward_service import RewardService
from controllers.domain.visual_AI_controller import VisualAIController
from controllers.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from controllers.infrastructure.model_repository import ModelRepository
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction
from shared.exceptions import FlippedRobotException
from shared.state import State

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
MODELS_PATH = Path("models")
MODEL_NAME = "model_1.keras"

# Training-specific constants
N_EPISODES = 1000
MEM_SIZE = 1000
MAX_EPSILON = 1      # Maximum epsilon value
MIN_EPSILON = 0.001    # Minimum epsilon value
EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * N_EPISODES))
AGGREGATE_STATS_EVERY = 20
SAVE_MODEL_EVERY = 1000


class TrainService:

    @staticmethod
    def run():

        # Variables initialization
        simulation: CoppeliaSimConnector = CoppeliaSimConnector()
        repo = ModelRepository(MODELS_PATH)
        model = repo.get_model(MODEL_NAME)
        controller = VisualAIController(model)
        robot = Pioneer3DXConnector(simulation.sim, ROBOT_ID, controller, use_camera=True)

        # Training-specific variables
        epsilon = MAX_EPSILON
        best_average = -math.inf
        best_score = -math.inf
        ep_rewards = [best_average]
        avg_reward_info = [[1, best_average, epsilon]]  # [episode_n, reward_n , epsilon_n]
        max_reward_info = [[1, best_score, epsilon]]
        eps_no_inc_counter = 0  # Counts episodes with no increment in reward

        # Iterate over episodes
        for episode in tqdm(range(1, N_EPISODES + 1), ascii=True, unit='episodes'):

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
            action = 0

            # Reset environment and get initial state
            simulation.start_simulation()
            current_state = State(robot.get_camera_reading())
            game_over = robot.is_flipped()

            while not game_over:
                try:
                    if np.random.random() > epsilon:
                        # Get action from Q table
                        action = controller.get_next_action(current_state)
                    else:
                        # Get random action
                        action = choice(MovementAction.get_action_space(robot.get_camera_reading().shape[0]))

                    robot.perform_next_action()
                    new_state = State(robot.get_camera_reading())
                    game_over = robot.is_flipped()
                    reward = RewardService.get_reward(current_state, action, new_state)

                    # Transform new continuous state to new discrete state and count reward
                    episode_reward += reward

                    # Every step we update replay memory and train main network
                    controller.update_replay_memory(tuple([current_state, action, reward, new_state, game_over]))
                    controller.train(game_over)

                    current_state = new_state
                    step += 1

                    ep_rewards.append(episode_reward)

                    if not episode % AGGREGATE_STATS_EVERY:
                        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

                        # Save models, but only when avg reward is greater or equal a set value
                        if not episode % SAVE_MODEL_EVERY:
                            # Save Agent :
                            repo.save_model(controller.model)

                        if average_reward > best_average:
                            best_average = average_reward
                            # update ECC variables:
                            avg_reward_info.append([episode, best_average, epsilon])
                            # Save Agent :
                            repo.save_model(controller.model)

                    if episode_reward > best_score:
                        best_score = episode_reward
                        max_reward_info.append([episode, best_score, epsilon])

                        # Save Agent :
                        repo.save_model(controller.model)

                        # Decay epsilon
                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)

                except FlippedRobotException:
                    continue

        # Stopping the simulation
        simulation.stop_simulation()

        # Variables destruction
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Setting logging format
    if DEBUG:
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('root')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    # Actual application startup
    TrainService.run()
