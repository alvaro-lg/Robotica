import logging

from pathlib import Path
from tf_agents.environments import suite_gym

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

# Training hyperparameters
NUM_ITERATIONS = 20000
INITIAL_COLLECT_STEPS = 100
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 100000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOG_INTERVAL = 200
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000


class TrainService:

    @staticmethod
    def run():
        pass  # TODO Not working yet

        # TODO Implement here this management
        """
        # Getting the orientation of the robot and checking if it has turned upside down
        if self.is_flipped():
            raise FlippedRobotException()

        if self.is_hitting_a_wall():
            raise WallHitException()
        """

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
