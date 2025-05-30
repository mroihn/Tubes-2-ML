import numpy as np
from .activation_layer import ReLULayerFS  # Contoh


class DenseLayerFS:

    def __init__(self, num_units: int, activation_name: str = None):
        self.num_units = num_units
        self.activation_name = activation_name
        self.activation_fn = None

        if self.activation_name == 'relu':
            self.activation_fn = ReLULayerFS().forward
        elif self.activation_name == 'softmax':
            self.activation_fn = self._softmax

        self.weights: np.ndarray = None
        self.biases: np.ndarray = None   # (num_units,)
        self.input_shape = None
        self.output_shape = None
        self.z = None

    def load_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Implementasi fungsi softmax yang stabil secara numerik."""
        # axis=1 jika z adalah (batch_size, num_features), axis=0 jika 1D
        axis = z.ndim - 1
        exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
        return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Method internal untuk menerapkan fungsi aktivasi."""
        if self.activation_fn:
            return self.activation_fn(z)
        return z  # Tanpa aktivasi

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights have not been loaded into DenseLayerFS.")

        self.input_shape = input_data.shape

        # Matriks perkalian: input @ weights + biases
        self.z = np.dot(input_data, self.weights) + self.biases
        output_data = self._apply_activation(self.z)

        self.output_shape = output_data.shape
        return output_data

    def __repr__(self):
        return f"DenseLayerFS(num_units={self.num_units}, activation='{self.activation_name}')"
