import numpy as np


class Conv2DLayerFS:

    def __init__(self, num_filters: int, kernel_size: tuple, stride: tuple = (1, 1), padding: str = 'valid'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride_h, self.stride_w = stride
        self.padding = padding

        self.weights: np.ndarray = None
        self.biases: np.ndarray = None   # (num_filters,)

        self.input_shape = None  # Akan di-infer saat forward pass pertama atau saat load weights
        self.output_shape = None  # Akan di-infer
        self.padded_input = None  # Untuk debugging atau backward pass nantinya

    def load_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights  #
        self.biases = biases  #
        if self.weights is not None:
            if self.input_shape is None or self.input_shape[-1] != self.weights.shape[2]:
                print(
                    f"[Conv2DLayerFS] Info: Input channels inferred from weights: {self.weights.shape[2]}")

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim == 3:  # (H, W, C)
            input_data = np.expand_dims(input_data, axis=0)

        batch_size, prev_h, prev_w, prev_c = input_data.shape
        self.input_shape = input_data.shape

        if self.weights is None:
            raise ValueError(
                "Weights have not been loaded into Conv2DLayerFS.")

        # prev_c_weights should match prev_c
        kernel_h, kernel_w, _, _ = self.weights.shape

        # Padding
        if self.padding == 'same':
            pad_h = max((prev_h - 1) * self.stride_h + kernel_h - prev_h, 0)
            pad_w = max((prev_w - 1) * self.stride_w + kernel_w - prev_w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_input = np.pad(input_data, ((
                0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        elif self.padding == 'valid':
            padded_input = input_data
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        self.padded_input = padded_input
        _, padded_h, padded_w, _ = padded_input.shape

        out_h = (padded_h - kernel_h) // self.stride_h + 1
        out_w = (padded_w - kernel_w) // self.stride_w + 1

        output_data = np.zeros((batch_size, out_h, out_w, self.num_filters))

        for b_idx in range(batch_size):
            current_input_padded = padded_input[b_idx]
            for f_idx in range(self.num_filters):
                current_kernel = self.weights[:, :, :, f_idx]
                current_bias = self.biases[f_idx]
                for r_idx in range(out_h):
                    for c_idx in range(out_w):
                        r_start = r_idx * self.stride_h
                        r_end = r_start + kernel_h
                        c_start = c_idx * self.stride_w
                        c_end = c_start + kernel_w

                        input_slice = current_input_padded[r_start:r_end,
                                                           c_start:c_end, :]

                        output_data[b_idx, r_idx, c_idx, f_idx] = np.sum(
                            input_slice * current_kernel) + current_bias

        self.output_shape = output_data.shape
        return output_data[0] if input_data.ndim == 3 and batch_size == 1 else output_data

    def __repr__(self):
        return (f"Conv2DLayerFS(num_filters={self.num_filters}, kernel_size={self.kernel_size}, "
                f"stride=({self.stride_h},{self.stride_w}), padding='{self.padding}')")
