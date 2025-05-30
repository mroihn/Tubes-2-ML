# src/models_from_scratch/cnn_fs.py

import numpy as np
import tensorflow as tf
import time

try:
    from CNN.conv_layer import Conv2DLayerFS
    from CNN.activation_layer import ReLULayerFS
    from CNN.pooling_layer import MaxPooling2DLayerFS, AveragePooling2DLayerFS
    from CNN.flatten_layer import FlattenLayerFS
    from CNN.dense_layer import DenseLayerFS
except ImportError:
    print("Peringatan: Gagal mengimpor layer CNN FS dengan path relatif standar.")


class CNNModelFS:
    """
    Kelas untuk model CNN yang dirakit dari layer-layer from scratch.
    """

    def __init__(self, layers_fs: list = None):
        """
        Konstruktor.

        Args:
            layers_fs (list, opsional): List berisi instance layer-layer from scratch yang membentuk model. Defaults to None.
        """
        self.layers = layers_fs if layers_fs is not None else []

    def add_layer(self, layer) -> None:
        """
        Menambahkan instance layer from scratch ke dalam model. 

        Args:
            layer: Objek dari salah satu kelas layer from scratch. 
        """
        self.layers.append(layer)

    def forward(self, input_data: np.ndarray, verbose: bool = False) -> np.ndarray:
        x = input_data
        if verbose:
            print(f"  Forward Pass FS - Input awal shape: {x.shape}")
        for i, layer_obj in enumerate(self.layers):
            start_layer_time = time.time()
            x = layer_obj.forward(x)  # Panggilan ke method forward layer
            end_layer_time = time.time()
            if verbose:
                print(
                    f"    Setelah Layer FS {i+1} ({layer_obj.__class__.__name__}): Output shape: {x.shape}, Time: {end_layer_time - start_layer_time:.4f}s")
        return x

    def predict_proba_batch(self, input_data: np.ndarray, batch_size_fs: int = 32, verbose_per_batch: bool = False) -> np.ndarray:
        """
        Melakukan prediksi (output probabilitas) dengan batching.
        Spesifikasi bonus menyebutkan forward propagation harus bisa menangani batch inference.

        Args:
            input_data (np.ndarray): Seluruh data input.
            batch_size_fs (int): Ukuran batch untuk inferensi.
            verbose_per_batch (bool): Jika True, cetak progres per batch.

        Returns:
            np.ndarray: Output probabilitas dari seluruh data input.
        """
        num_samples = input_data.shape[0]
        all_predictions = []

        if verbose_per_batch:
            print(
                f"Memulai prediksi batch FS untuk {num_samples} sampel dengan batch size {batch_size_fs}")

        for i in range(0, num_samples, batch_size_fs):
            batch_input = input_data[i:i + batch_size_fs]
            if verbose_per_batch:
                total_batches = (
                    num_samples + batch_size_fs - 1) // batch_size_fs
                print(
                    f"  Memproses batch FS {i//batch_size_fs + 1}/{total_batches} (sampel {i}-{i+len(batch_input)-1})")

            # Set verbose=False untuk forward internal agar tidak terlalu banyak output
            batch_output = self.forward(batch_input, verbose=False)
            all_predictions.append(batch_output)

        return np.vstack(all_predictions)

    def predict_classes_batch(self, input_data: np.ndarray, batch_size_fs: int = 32) -> np.ndarray:
        """
        Melakukan prediksi kelas dengan batching.

        Args:
            input_data (np.ndarray): Seluruh data input.
            batch_size_fs (int): Ukuran batch untuk inferensi.

        Returns:
            np.ndarray: Prediksi kelas (integer) dari seluruh data input.
        """
        probas = self.predict_proba_batch(
            input_data, batch_size_fs=batch_size_fs)
        return np.argmax(probas, axis=1)

    def load_keras_weights(self, keras_model: tf.keras.Model) -> None:
        print("\nMemulai pemuatan bobot Keras ke model FromScratch (versi optimasi)...")

        # 1. Dapatkan layer Keras yang memiliki bobot
        keras_layers_with_weights = [
            layer for layer in keras_model.layers if len(layer.get_weights()) > 0
        ]

        # 2. Dapatkan layer FromScratch yang memiliki method load_weights (berarti bisa menerima bobot)
        fs_layers_expecting_weights = [
            layer for layer in self.layers if hasattr(layer, 'load_weights')
        ]

        num_keras_trainable = len(keras_layers_with_weights)
        num_fs_trainable = len(fs_layers_expecting_weights)

        if num_fs_trainable == 0:
            print("Peringatan: Tidak ada layer FromScratch yang zmembutuhkan bobot.")
            return
        if num_keras_trainable == 0:
            print("Peringatan: Tidak ada layer Keras yang memiliki bobot untuk dimuat.")
            return

        print(
            f"  Ditemukan {num_fs_trainable} layer FS yang mengharapkan bobot.")
        print(
            f"  Ditemukan {num_keras_trainable} layer Keras yang memiliki bobot.")

        keras_idx = 0
        fs_idx = 0

        while fs_idx < num_fs_trainable and keras_idx < num_keras_trainable:
            fs_layer = fs_layers_expecting_weights[fs_idx]
            keras_layer = keras_layers_with_weights[keras_idx]

            print(
                f"  Mencoba memuat: FS L{fs_idx+1} ({fs_layer.__class__.__name__}) <- Keras L{keras_idx+1} ({keras_layer.name}, {keras_layer.__class__.__name__})")

            weights_successfully_loaded = False
            try:
                # Cocokkan berdasarkan tipe layer utama yang berbobot
                if isinstance(fs_layer, Conv2DLayerFS) and isinstance(keras_layer, tf.keras.layers.Conv2D):
                    keras_w = keras_layer.get_weights()
                    if len(keras_w) == 2:  # bobot dan bias
                        fs_layer.load_weights(keras_w[0], keras_w[1])
                        weights_successfully_loaded = True
                    elif len(keras_w) == 1:  # hanya bobot (use_bias=False di Keras)
                        fs_layer.load_weights(keras_w[0], np.zeros(
                            keras_w[0].shape[-1]))  # Asumsi bias nol
                        weights_successfully_loaded = True
                elif isinstance(fs_layer, DenseLayerFS) and isinstance(keras_layer, tf.keras.layers.Dense):
                    keras_w = keras_layer.get_weights()
                    if len(keras_w) == 2:
                        fs_layer.load_weights(keras_w[0], keras_w[1])
                        weights_successfully_loaded = True
                    elif len(keras_w) == 1:
                        fs_layer.load_weights(
                            keras_w[0], np.zeros(keras_w[0].shape[-1]))
                        weights_successfully_loaded = True
                # Tambahkan penanganan untuk layer berbobot lain jika ada (misal, EmbeddingLayerFS)
                # else:
                #     print(f"    Tipe layer tidak cocok untuk pemuatan bobot: FS {fs_layer.__class__.__name__} vs Keras {keras_layer.__class__.__name__}")

                if weights_successfully_loaded:
                    print(f"    Bobot berhasil dimuat.")
                    fs_idx += 1
                    keras_idx += 1
                else:
                    # Jika tipe tidak cocok untuk pasangan saat ini, coba majukan pointer Keras
                    # Ini mengasumsikan mungkin ada layer Keras berbobot yang tidak punya padanan langsung di FS
                    # (misal, BatchNormalization jika FS tidak mengimplementasikannya tapi Keras punya bobotnya)
                    print(
                        f"    Tidak ada kecocokan pemuatan bobot untuk FS {fs_layer.__class__.__name__} dengan Keras {keras_layer.name}. Mencoba Keras layer berikutnya.")
                    keras_idx += 1

            except Exception as e:
                print(
                    f"    ERROR saat memuat bobot untuk FS {fs_layer.__class__.__name__} dari Keras {keras_layer.name}: {e}")
                # Putuskan bagaimana menangani error ini, misal skip layer Keras ini
                keras_idx += 1

        if fs_idx < num_fs_trainable:
            print(
                f"Peringatan: Tidak semua layer FromScratch yang membutuhkan bobot ({num_fs_trainable}) mendapatkan bobot ({fs_idx} dimuat).")
            for i in range(fs_idx, num_fs_trainable):
                print(
                    f"  Layer FS yang tidak mendapat bobot: {fs_layers_expecting_weights[i].__class__.__name__}")
        if keras_idx < num_keras_trainable:
            print(
                f"Peringatan: Tidak semua layer Keras yang memiliki bobot ({num_keras_trainable}) digunakan ({keras_idx} digunakan).")
            for i in range(keras_idx, num_keras_trainable):
                print(
                    f"  Layer Keras berbobot yang tidak terpakai: {keras_layers_with_weights[i].name}")

        print("Selesai memuat bobot Keras.")

    def __repr__(self):
        layer_reprs = "\n  ".join([repr(layer) for layer in self.layers])
        return f"CNNModelFS(layers=[\n  {layer_reprs}\n])"


# Contoh penggunaan:
if __name__ == '__main__':
    # Bagian ini hanya untuk ilustrasi jika file ini dijalankan langsung
    # Anda perlu mengimplementasikan layer-layer FS di folder CNN terlebih dahulu

    # Dummy layers untuk contoh
    # from CNN.conv_layer import Conv2DLayerFS # Ganti dengan path impor yang benar
    # ...dan seterusnya untuk layer lain

    # Misal kita punya implementasi layer FS
    # model_fs = CNNModelFS()
    # model_fs.add_layer(Conv2DLayerFS(num_filters=32, kernel_size=(3,3), padding='same'))
    # model_fs.add_layer(ReLULayerFS())
    # # ... tambahkan layer lain sesuai arsitektur
    # model_fs.add_layer(DenseLayerFS(num_units=10, activation_name='softmax'))

    # print(model_fs)

    # Untuk load_keras_weights, Anda perlu model Keras yang sudah dilatih
    # (keras_model_instance = tf.keras.models.load_model('path/to/your/keras_model.h5'))
    # (model_fs.load_keras_weights(keras_model_instance))

    # Buat dummy input
    # dummy_input = np.random.rand(1, 32, 32, 3) # Misal (batch, height, width, channels)
    # Jika layer sudah punya bobot (misal dari load_keras_weights atau di-set manual)
    # output = model_fs.forward(dummy_input, verbose=True)
    # print("Output shape dari dummy input:", output.shape)
    pass
