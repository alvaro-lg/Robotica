import logging
import time
from copy import copy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shared.action_space import ActionSpace
from shared.data_types import Transition
from shared.state import State
from simulations.domain.controllers.visual_AI_controller import VisualAIController
from simulations.domain.services.image_processing_service import ImageProcessingService
from simulations.domain.services.reward_service import RewardService
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX
from simulations.domain.simulation_elements.sphere import Sphere
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from simulations.infrastructure.model_repository import ModelRepository

# Constants
DEBUG = True
DISPLAY = True
ROBOT_ID = "PioneerP3DX"
SPHERE_ID = "Sphere"
MODEL_NAME = "model"
MODELS_PATH = Path("models")
LOG_DIR = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# Training-specific constants
N_EPISODES = 300
MAX_EPSILON = 1                                                 # Maximum epsilon value
MIN_EPSILON = 0.001                                             # Minimum epsilon value
EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * N_EPISODES))
MAX_STEPS_PER_EPISODE = 1000


class TrainService:

    @staticmethod
    def run():

        # Variables initialization
        simulation = CoppeliaSimConnector.get_instance()
        repo = ModelRepository(MODELS_PATH)
        model = repo.load(MODEL_NAME)
        controller = VisualAIController(model)
        robot = Pioneer3DX(simulation, controller, ROBOT_ID)
        sphere = Sphere(simulation, SPHERE_ID)
        simulation.add_sim_elements([robot, sphere])
        tb_writer = tf.summary.create_file_writer(f'{LOG_DIR}/train')
        tb_writer.set_as_default()
        rng = np.random.default_rng(2024)

        # Training-specific variables
        epsilon = MAX_EPSILON
        best_reward = -np.inf
        action_space = ActionSpace.get_instance()

        try:

            # Warning-avoidance
            episode = 0

            for episode in tqdm(range(N_EPISODES), ascii=True, unit='episodes'):

                # Reset variables
                simulation.start_simulation(stepping=True)
                simulation.reset_simulation(shuffle=False)
                episode_total_reward = 0

                for n_step in range(MAX_STEPS_PER_EPISODE):

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

                    # Getting actual states
                    curr_state = robot.get_state()

                    # Exploration-exploitation
                    if rng.random() <= epsilon:
                        # Get random action
                        action = action_space.get_instance().random_action()
                    else:
                        # Get action from Q table
                        action = action_space.actions[np.argmax(controller.get_prediction(curr_state))]

                    # Performing the action, getting next states and calculating the reward
                    robot.perform_next_action(action=action)

                    simulation.step()  # Next step
                    next_state = State(robot.get_camera_reading())
                    reward = RewardService.get_reward(curr_state, next_state)

                    # Forcing end of episode if proceeds
                    has_lost_the_ball = curr_state.is_ball_in_sight() and not next_state.is_ball_in_sight()
                    force_end = robot.is_hitting_a_wall() or robot.is_flipped() or has_lost_the_ball

                    # Updating metrics
                    episode_total_reward += reward

                    # Every step we update replay memory and train main network
                    end_episode = (n_step == MAX_STEPS_PER_EPISODE - 1) or force_end
                    transition = np.empty(1, dtype=Transition)
                    transition['curr_state'] = np.array(curr_state)
                    transition['action'] = action
                    transition['reward'] = reward
                    transition['next_state'] = np.array(next_state)
                    transition['done'] = end_episode
                    controller.update_replay_memory(transition)
                    controller.train(end_episode, log_dir=LOG_DIR)

                    # Quitting if end of episode
                    if force_end:
                        logger.info(f"Episode: {episode} - FORCING QUIT")
                        break

                # End of episode
                logger.info(f"Episode: {episode} - Epsilon: {epsilon} - Total Reward: {episode_total_reward}")

                # Saving stats and saving model if proceeds
                tf.summary.scalar('total_reward', episode_total_reward, step=episode)

                # Updating and saving if model improves
                if episode_total_reward > best_reward:
                    best_reward = episode_total_reward
                    repo.store(controller.model, f"{MODEL_NAME}_ep{episode}")

                # Decay epsilon
                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)

                # Stopping the simulation
                simulation.stop_simulation()

            repo.store(controller.model, f"{MODEL_NAME}_ep{episode}")

        except (Exception, KeyboardInterrupt):
            # Stopping the simulation
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
