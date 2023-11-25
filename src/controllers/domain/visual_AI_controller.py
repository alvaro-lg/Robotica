import numpy as np

from controllers.domain.visual_controller import VisualController
from controllers.infrastructure.pioneer_3DX_connector import Pioneer3DXConnector
from shared.actions import MovementAction
from shared.data_types import AIModel
from shared.state import State


class VisualAIController(VisualController):

    def __init__(self, model: AIModel):
        super().__init__()

        self.__model = model

    def get_next_action(self, state: State) -> MovementAction:

        # Getting input for prediction
        x, y, area = state.x_norm, state.y_norm, state.area_norm
        input_data = np.array([[[x, y, area]]])

        # Predicting the output
        left_speed_ratio, right_speed_ratio = self.__model.predict(input_data)[0][0]
        left_speed, right_speed = (left_speed_ratio * Pioneer3DXConnector.max_speed,
                                   right_speed_ratio * Pioneer3DXConnector.max_speed)

        # Returning the corresponding action
        return MovementAction((left_speed, right_speed))
