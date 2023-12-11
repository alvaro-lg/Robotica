import logging
from copy import copy
from pathlib import Path

import numpy as np
from tqdm import tqdm

import cv2
import math

from controllers.domain.image_processing_service import ImageProcessingService
from controllers.domain.reward_service import RewardService
from controllers.domain.visual_AI_controller import VisualAIController
from controllers.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from controllers.infrastructure.model_factory import ModelFactory
from controllers.infrastructure.model_repository import ModelRepository
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction, MovementActionFactory, EnumeratedMovementAction
from shared.exceptions import FlippedRobotException, WallHitException
from shared.state import State

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
MODEL_NAME = "model_1.keras"
MODELS_PATH = Path("models")

# Training-specific constants
N_EPISODES = 1000
MEM_SIZE = 3000
MAX_EPSILON = 1  # Maximum epsilon value
MIN_EPSILON = 0.001  # Minimum epsilon value
EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * N_EPISODES))
AGGREGATE_STATS_EVERY = 1
SAVE_MODEL_EVERY = 20
MAX_TIME = 90


class TrainService:

    @staticmethod
    def run():

        # Variables initialization
        simulation = CoppeliaSimConnector()
        repo = ModelRepository(MODELS_PATH)
        model = repo.get_model(MODEL_NAME)
        controller = VisualAIController(model)
        robot = Pioneer3DXConnector(simulation.sim, ROBOT_ID, controller, use_camera=True)
        MovementActionFactory.create_action_space()

        # Training-specific variables
        epsilon = MAX_EPSILON
        best_average = -math.inf
        best_score = -math.inf
        ep_rewards = [best_average]
        avg_reward_info = [[1, best_average, epsilon]]  # [episode_n, reward_n , epsilon_n]
        max_reward_info = [[1, best_score, epsilon]]  # [episode_n, reward_n , epsilon_n]

        def end_episode(n_episode: int) -> bool:
            """
                Checks whether the episode has ended.
                :return: boolean: True if it has ended, False otherwise.
            """
            return robot.is_flipped() or (simulation.get_time() > MAX_TIME * n_episode)

        simulation.start_simulation()

        try:
            # Iterate over episodes
            for episode in tqdm(range(1, N_EPISODES + 1), ascii=True, unit='episodes'):
                try:

                    # Reset environment and get initial state
                    simulation.reset_simulation(shuffle_points=False)
                    curr_state = robot.get_state()

                    # Restarting episode - reset episode reward and step number
                    episode_reward = 0
                    step = 1

                    while not end_episode(episode + 1):

                        if DISPLAY:
                            # Getting the camera readings, contours and circle
                            img = robot.get_camera_reading()
                            _, img_mask = ImageProcessingService.get_contours(copy(img), ret_img_mask=True)
                            img_contours = ImageProcessingService.get_image_contours(copy(img))

                            # Stacking the images together
                            img_final = cv2.hconcat([img, img_mask, img_contours])

                            # Blending the image with the contours and displaying them
                            img_final_s = cv2.resize(img_final, (768, 256), interpolation=cv2.INTER_AREA)
                            cv2.imshow('Camera View', img_final_s)
                            cv2.waitKey(1)

                        # Exploration-exploitation
                        if np.random.random() > epsilon:
                            # Get action from Q table
                            enum_action = EnumeratedMovementAction(np.argmax(controller.get_prediction(curr_state)))
                        else:
                            # Get random action
                            enum_action = MovementActionFactory.get_random_enum_action()

                        robot.perform_next_action(enum_action)
                        new_state = State(robot.get_camera_reading())
                        reward = RewardService.get_reward(curr_state, enum_action, new_state)

                        # Transform new continuous state to new discrete state and count reward
                        episode_reward += reward

                        # Every step we update replay memory and train main network
                        controller.update_replay_memory((curr_state, enum_action, reward, new_state, end_episode(episode + 1)))

                        curr_state = new_state
                        step += 1

                        ep_rewards.append(episode_reward)
                    else:
                        # Stopping the simulation
                        logger.info(f"Episode: {episode} - Reward: {episode_reward} - Epsilon: {epsilon}")

                        # TODO Ver si poner aqui
                        controller.train(end_episode(episode + 1))

                    # Saving stats and saving model if proceeds
                    if not episode % AGGREGATE_STATS_EVERY:
                        average_reward = (sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) /
                                          len(ep_rewards[-AGGREGATE_STATS_EVERY:]))

                        # Save models, but only when avg reward is greater or equal a set value
                        if not episode % SAVE_MODEL_EVERY:
                            repo.save_model(controller.model)

                        if average_reward > best_average:
                            best_average = average_reward
                            avg_reward_info.append([episode, best_average, epsilon])
                            repo.save_model(controller.model)

                    # Updating and saving if model improves
                    if episode_reward > best_score:
                        best_score = episode_reward
                        max_reward_info.append([episode, best_score, epsilon])
                        repo.save_model(controller.model)

                    # Decay epsilon
                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)

                except FlippedRobotException:
                    continue

                except WallHitException:
                    simulation.reset_simulation(shuffle_points=False)
                    pass

        except KeyboardInterrupt:
            pass

        logger.debug(ep_rewards)
        simulation.stop_simulation()

        # Variables destruction
        if DISPLAY:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Setting logging format
    if DEBUG:
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('root')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Actual application startup
    TrainService.run()
