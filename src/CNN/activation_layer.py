import numpy as np


class ReLULayerFS:
    def __init__(self):
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)

    def __repr__(self):
        return "ReLULayerFS()"
