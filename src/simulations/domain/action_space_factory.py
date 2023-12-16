from typing import List

import numpy as np

from shared.actions import MovementAction


class ActionSpaceFactory:

    # Static attributes
    n_x_steps: int = 5
    n_actions: int = 2 * n_x_steps
    actions: List[MovementAction] = None

    def __init__(self):
        self.actions = [
            MovementAction((float(min(i / 0.5, 1)), float(min((1 - i) / 0.5, 1))))
            for i in np.arange(0, 1, 1 / self.n_x_steps)] + [
            MovementAction((-float(min(i / 0.5, 1)), -float(min((1 - i) / 0.5, 1))))
            for i in np.arange(0, 1, 1 / self.n_x_steps)
        ]
