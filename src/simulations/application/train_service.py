import logging
from copy import copy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shared.action_space import ActionSpace
from shared.state import State
from simulations.domain.controllers.visual_AI_controller import VisualAIController
from simulations.domain.factories.model_factory import ModelFactory
from simulations.domain.services.image_processing_service import ImageProcessingService
from simulations.domain.services.reward_service import RewardService
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from simulations.infrastructure.model_repository import ModelRepository

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
MODEL_NAME = "model_1.keras"
MODELS_PATH = Path("models")
LOG_DIR = 'log/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# Training-specific constants
N_EPISODES = 200
MEM_SIZE = 3000
MAX_EPSILON = 1  # Maximum epsilon value
MIN_EPSILON = 0.001  # Minimum epsilon value
EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * N_EPISODES))
STEPS_PER_EPISODE = 150


class TrainService:

    @staticmethod
    def run():

        # Variables initialization
        simulation = CoppeliaSimConnector.get_instance()
        repo = ModelRepository(MODELS_PATH)
        model = repo.load(MODEL_NAME)
        controller = VisualAIController(model)
        robot = Pioneer3DX(simulation, controller, ROBOT_ID)
        simulation.add_sim_element(robot)
        tb_writer = tf.summary.create_file_writer(LOG_DIR)
        tb_writer.set_as_default()

        # Training-specific variables
        epsilon = MAX_EPSILON
        best_reward = -np.inf
        action_space = ActionSpace.get_instance()

        # Starting the simulation
        simulation.start_simulation(stepping=True)

        try:

            for episode in tqdm(range(N_EPISODES), ascii=True, unit='episodes'):

                # Reset variables
                simulation.reset_simulation(shuffle=False)
                episode_total_reward = 0
                episode_max_reward = -np.inf
                episode_min_reward = np.inf
                n_step = 0  # Warning-avoidance

                for n_step in range(STEPS_PER_EPISODE):

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

                    # Getting actual state
                    curr_state = robot.get_state()

                    # Exploration-exploitation
                    if np.random.random() > epsilon:
                        # Get action from Q table
                        action = action_space.actions[np.argmax(controller.get_prediction(curr_state))]
                    else:
                        # Get random action
                        action = action_space.random_action()

                    # Performing the action, getting next state and calculating the reward
                    robot.perform_next_action(action=action)
                    simulation.step()  # Next step
                    next_state = State(robot.get_camera_reading())
                    reward = RewardService.get_reward(curr_state, next_state)

                    # Forcing end of episode if proceeds
                    force_end = robot.is_hitting_a_wall() or robot.is_flipped() or not next_state.is_ball_in_sight()

                    # Updating metrics
                    episode_total_reward += reward
                    episode_max_reward = reward if reward > episode_max_reward else episode_max_reward
                    episode_min_reward = reward if reward < episode_min_reward else episode_min_reward

                    # Every step we update replay memory and train main network
                    end_episode = (n_step == STEPS_PER_EPISODE) - 1 or force_end
                    controller.update_replay_memory((curr_state, action, reward, next_state, end_episode))
                    controller.train(end_episode)

                    # Quitting if end of episode
                    if force_end:
                        logger.info(f"Episode: {episode} - FORCING QUIT")
                        break

                # End of episode
                logger.info(f"Episode: {episode} - Avg. Reward: "
                            f"{float(episode_total_reward) / float(n_step + 1)} (min: {episode_min_reward}, "
                            f"max: {episode_max_reward}) - Epsilon: {epsilon}")

                # Saving stats and saving model if proceeds
                tf.summary.scalar('average_reward', float(episode_total_reward) / float(n_step + 1), step=episode)
                tf.summary.scalar('max_reward', episode_max_reward, step=episode)
                tf.summary.scalar('min_reward', episode_min_reward, step=episode)

                # Updating and saving if model improves
                if episode_total_reward > best_reward:
                    best_reward = episode_total_reward
                    repo.store(controller.model)

                # Decay epsilon
                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)

        except KeyboardInterrupt:
            pass

        # Variables destruction
        simulation.stop_simulation()
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
