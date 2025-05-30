import numpy as np


class ReLULayerFS:
    """
    Implementasi fungsi aktivasi ReLU from scratch.
    """

    def __init__(self):
        """Konstruktor."""
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Mengaplikasikan max(0, x) elemen-wise.

        Args:
            input_data (np.ndarray): Input data.

        Returns:
            np.ndarray: input_data setelah aktivasi ReLU.
        """
        return np.maximum(0, input_data)

    def __repr__(self):
        return "ReLULayerFS()"
