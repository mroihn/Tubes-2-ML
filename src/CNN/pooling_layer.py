import numpy as np


class MaxPooling2DLayerFS:
    """
    Implementasi Max Pooling 2D from scratch.
    """

    def __init__(self, pool_size: tuple, stride: tuple = None):
        """
        Konstruktor.

        Args:
            pool_size (tuple): Ukuran window pooling (pool_height, pool_width)
            stride (tuple, optional): Stride pooling (stride_y, stride_x). 
                                     Jika None, defaultnya sama dengan pool_size.
        """
        self.pool_h, self.pool_w = pool_size
        if stride is None:
            self.stride_h, self.stride_w = self.pool_h, self.pool_w
        else:
            self.stride_h, self.stride_w = stride

        self.input_shape = None
        self.output_shape = None
        self.arg_max_indices = None  # Untuk backward pass nantinya

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Melakukan operasi Max Pooling. 

        Args:
            input_data (np.ndarray): Input data (batch_size, height, width, channels).

        Returns:
            np.ndarray: Output setelah pooling. 
        """
        if input_data.ndim == 3:  # (H, W, C)
            input_data = np.expand_dims(input_data, axis=0)

        batch_size, prev_h, prev_w, num_channels = input_data.shape
        self.input_shape = input_data.shape

        out_h = (prev_h - self.pool_h) // self.stride_h + 1
        out_w = (prev_w - self.pool_w) // self.stride_w + 1

        output_data = np.zeros((batch_size, out_h, out_w, num_channels))
        # self.arg_max_indices = np.zeros_like(output_data, dtype=object)

        for b_idx in range(batch_size):
            for ch_idx in range(num_channels):
                for r_idx in range(out_h):
                    for c_idx in range(out_w):
                        r_start = r_idx * self.stride_h
                        r_end = r_start + self.pool_h
                        c_start = c_idx * self.stride_w
                        c_end = c_start + self.pool_w

                        input_slice = input_data[b_idx,
                                                 r_start:r_end, c_start:c_end, ch_idx]
                        output_data[b_idx, r_idx, c_idx,
                                    ch_idx] = np.max(input_slice)

        self.output_shape = output_data.shape
        return output_data[0] if input_data.ndim == 3 and batch_size == 1 else output_data

    def __repr__(self):
        return f"MaxPooling2DLayerFS(pool_size=({self.pool_h},{self.pool_w}), stride=({self.stride_h},{self.stride_w}))"


class AveragePooling2DLayerFS:
    """
    Implementasi Average Pooling 2D from scratch.
    """

    def __init__(self, pool_size: tuple, stride: tuple = None):
        """
        Args:
            pool_size (tuple): Ukuran window pooling (pool_height, pool_width).
            stride (tuple, optional): Stride pooling (stride_y, stride_x). 
                                     Jika None, defaultnya sama dengan pool_size.
        """
        self.pool_h, self.pool_w = pool_size
        if stride is None:
            self.stride_h, self.stride_w = self.pool_h, self.pool_w
        else:
            self.stride_h, self.stride_w = stride

        self.input_shape = None
        self.output_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Melakukan operasi Average Pooling.

        Args:
            input_data (np.ndarray): Input data (batch_size, height, width, channels).

        Returns:
            np.ndarray: Output setelah pooling.
        """
        if input_data.ndim == 3:  # (H, W, C)
            input_data = np.expand_dims(input_data, axis=0)

        batch_size, prev_h, prev_w, num_channels = input_data.shape
        self.input_shape = input_data.shape

        out_h = (prev_h - self.pool_h) // self.stride_h + 1
        out_w = (prev_w - self.pool_w) // self.stride_w + 1

        output_data = np.zeros((batch_size, out_h, out_w, num_channels))

        for b_idx in range(batch_size):
            for ch_idx in range(num_channels):
                for r_idx in range(out_h):
                    for c_idx in range(out_w):
                        r_start = r_idx * self.stride_h
                        r_end = r_start + self.pool_h
                        c_start = c_idx * self.stride_w
                        c_end = c_start + self.pool_w

                        input_slice = input_data[b_idx,
                                                 r_start:r_end, c_start:c_end, ch_idx]
                        output_data[b_idx, r_idx, c_idx,
                                    ch_idx] = np.mean(input_slice)

        self.output_shape = output_data.shape
        return output_data[0] if input_data.ndim == 3 and batch_size == 1 else output_data

    def __repr__(self):
        return f"AveragePooling2DLayerFS(pool_size=({self.pool_h},{self.pool_w}), stride=({self.stride_h},{self.stride_w}))"
