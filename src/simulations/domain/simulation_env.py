from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
from gym.vector.utils import spaces

from simulations.domain.action_space_factory import ActionSpaceFactory
from simulations.domain.controllers.visual_AI_controller import VisualAIController
from simulations.infrastructure.coppelia_sim_connector import CoppeliaSimConnector
from simulations.domain.model_factory import ModelFactory
from simulations.infrastructure.model_repository import ModelRepository
from simulations.domain.simulation_elements.pioneer_3DX import Pioneer3DX
from shared.actions import MovementAction

# Constants
ROBOT_ID = "PioneerP3DX"
MODEL_NAME = "model_1.keras"
MODELS_PATH = Path("models")
DEFAULT_FPS = 30
DEFAULT_CAMERA_RESOLUTION = (256, 256)


class SimulationEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": DEFAULT_FPS}

    def __init__(self, render_mode: Optional[str] = None, size: Tuple[int, int] = DEFAULT_CAMERA_RESOLUTION):

        # Attributes initialization
        self.size: Tuple[int, int] = size
        self.window_size: int = 512  # TODO Parameterize this
        self.observation_space = spaces.Box(0, 1, shape=ModelFactory.input_shape, dtype=np.float64)
        self.action_space = spaces.Discrete(ActionSpaceFactory.n_actions)
        self._action_to_direction: dict[int, MovementAction] = {
            idx: action for idx, action in enumerate(ActionSpaceFactory.actions)
        }
        self.window = None
        self.clock = None

        repo = ModelRepository(MODELS_PATH)
        model = repo.load(MODEL_NAME)
        controller = VisualAIController(model)
        self.__sim_connector: CoppeliaSimConnector = CoppeliaSimConnector()
        self.__robot: Pioneer3DX = Pioneer3DX(self.__sim_connector.sim, controller, ROBOT_ID)

        # Argument checking
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise AttributeError(f"Invalid render mode: {render_mode}")
        else:
            self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target_area's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # TODO
    def get_state(self) -> StateT:
        """
            Retrieves the state of the robot.
            :return: a tuple containing the position of the robot on the three axis.
        """
        return State(self.get_camera_reading())

