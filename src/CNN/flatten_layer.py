import numpy as np


class FlattenLayerFS:
    """
    Implementasi layer Flatten from scratch.
    """

    def __init__(self):
        """Konstruktor."""
        self.input_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Mengubah input multidimensi menjadi vektor 1D per instance dalam batch
        Shape input (batch_size, H, W, C) menjadi (batch_size, H*W*C). 

        Args:
            input_data (np.ndarray): Input data (misalnya, output dari Conv/Pool layer).

        Returns:
            np.ndarray: Data yang sudah di-flatten. 
        """
        self.input_shape = input_data.shape
        if input_data.ndim == 1:  # Sudah flatten
            return input_data

        batch_size = input_data.shape[0]
        # Reshape menjadi (batch_size, -1) dimana -1 akan mengkalkulasi sisa dimensi
        return input_data.reshape(batch_size, -1)

    def __repr__(self):
        return "FlattenLayerFS()"
