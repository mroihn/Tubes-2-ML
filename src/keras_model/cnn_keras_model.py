import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, ReLU
)
from tensorflow.keras.optimizers import Adam  # Updated import for Adam optimizer


def build_cnn_keras(
    input_shape: tuple,
    num_classes: int,
    conv_blocks_params: list,
    pooling_type: str,
    pooling_size: tuple,
    use_global_pooling: str = None,  # 'avg', 'max', or None for Flatten
    dense_layers_params: list = None
) -> tf.keras.Model:
    """
    Membangun model CNN Keras yang dapat dikonfigurasi.

    Args:
        input_shape (tuple): Dimensi input gambar (misal, (32, 32, 3)).
        num_classes (int): Jumlah kelas output.
        conv_blocks_params (list): List of dictionaries. Setiap dict mendefinisikan 
                                   satu blok konvolusi dan dapat berisi:
                                   {'filters': int, 'kernel_size': tuple, 
                                    'num_conv_layers': int (jumlah Conv2D + ReLU dalam blok ini)}
                                   Contoh: [{'filters': 32, 'kernel_size': (3,3), 'num_conv_layers': 2},
                                            {'filters': 64, 'kernel_size': (3,3), 'num_conv_layers': 2}]
                                   Setelah setiap blok konvolusi, akan ada pooling layer.
        pooling_type (str): Jenis pooling layer ('max' atau 'average'). 
        pooling_size (tuple): Ukuran window pooling (misal, (2,2)).
        use_global_pooling (str, optional): Jenis global pooling ('avg' untuk GlobalAveragePooling2D,
                                           'max' untuk GlobalMaxPooling2D). 
                                           Jika None, Flatten layer akan digunakan. Defaults to None.
        dense_layers_params (list, optional): List of dictionaries untuk Dense layer sebelum output layer. 
                                              Setiap dict berisi {'units': int}. 
                                              Contoh: [{'units': 128}, {'units': 64}]
                                              Defaults to None (tidak ada hidden dense layer tambahan).

    Returns:
        tf.keras.Model: Model CNN Keras yang sudah di-compile.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Convolutional Blocks
    for i, block_param in enumerate(conv_blocks_params):
        filters = block_param.get('filters', 32)
        kernel_size = block_param.get('kernel_size', (3, 3))
        num_conv_layers_in_block = block_param.get('num_conv_layers', 1)

        for _ in range(num_conv_layers_in_block):
            model.add(Conv2D(filters=filters, kernel_size=kernel_size,
                      padding='same'))  # 
            # Menggunakan ReLU sebagai aktivasi standar setelah Conv2D
            model.add(ReLU())

        # Tambahkan Pooling layer setelah setiap blok konvolusi 
        if pooling_type == 'max':
            model.add(MaxPooling2D(pool_size=pooling_size))
        elif pooling_type == 'average':
            model.add(AveragePooling2D(pool_size=pooling_size))
        else:
            raise ValueError("pooling_type harus 'max' atau 'average'")

    # Flatten atau Global Pooling Layer 
    if use_global_pooling == 'avg':
        model.add(GlobalAveragePooling2D())
    elif use_global_pooling == 'max':
        model.add(GlobalMaxPooling2D())
    elif use_global_pooling is None:
        model.add(Flatten())
    else:
        raise ValueError("use_global_pooling harus 'avg', 'max', atau None")

    # Dense Layers (Hidden)
    if dense_layers_params:
        for dense_param in dense_layers_params:
            units = dense_param.get('units', 128)
            model.add(Dense(units=units))  # 
            # Umumnya ReLU juga digunakan di hidden dense layer
            model.add(ReLU())

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # 

    # Compile Model
    optimizer = Adam()  # 
    # 
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  # f1-score akan dihitung terpisah saat evaluasi akhir
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Contoh penggunaan fungsi build_cnn_keras
    input_s = (32, 32, 3)  # CIFAR-10 input shape
    n_classes = 10       # CIFAR-10 number of classes

    # Konfigurasi untuk eksperimen 1: Model dasar
    conv_config_1 = [
        {'filters': 32, 'kernel_size': (3, 3), 'num_conv_layers': 1},  # Blok 1
        {'filters': 64, 'kernel_size': (3, 3), 'num_conv_layers': 1}  # Blok 2
    ]
    dense_config_1 = [{'units': 128}]

    model1 = build_cnn_keras(
        input_shape=input_s,
        num_classes=n_classes,
        conv_blocks_params=conv_config_1,
        pooling_type='max',
        pooling_size=(2, 2),
        use_global_pooling=None,  # Menggunakan Flatten
        dense_layers_params=dense_config_1
    )
    print("Model 1 Summary (Flatten):")
    model1.summary()

    # Konfigurasi untuk eksperimen 2: Model dengan lebih banyak layer konvolusi per blok & Global Avg Pooling
    conv_config_2 = [
        {'filters': 32, 'kernel_size': (3, 3), 'num_conv_layers': 2},
        {'filters': 64, 'kernel_size': (3, 3), 'num_conv_layers': 2},
        {'filters': 128, 'kernel_size': (3, 3), 'num_conv_layers': 2}
    ]
    dense_config_2 = [{'units': 256}]

    model2 = build_cnn_keras(
        input_shape=input_s,
        num_classes=n_classes,
        conv_blocks_params=conv_config_2,
        pooling_type='average',
        pooling_size=(2, 2),
        use_global_pooling='avg',  # Menggunakan Global Average Pooling
        dense_layers_params=dense_config_2
    )
    print("\nModel 2 Summary (Global Average Pooling):")
    model2.summary()

    # Konfigurasi untuk eksperimen 3: Model tanpa hidden dense layer tambahan
    conv_config_3 = [
        {'filters': 32, 'kernel_size': (5, 5), 'num_conv_layers': 1}
    ]

    model3 = build_cnn_keras(
        input_shape=input_s,
        num_classes=n_classes,
        conv_blocks_params=conv_config_3,
        pooling_type='max',
        pooling_size=(3, 3),
        use_global_pooling='max',  # Menggunakan Global Max Pooling
        dense_layers_params=None  # Tidak ada hidden dense layer
    )
    print("\nModel 3 Summary (Global Max Pooling, No Hidden Dense):")
    model3.summary()
