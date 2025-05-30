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
        """
        Melakukan forward pass input melalui semua layer secara berurutan. 

        Args:
            input_data (np.ndarray): Input data.
            verbose (bool): Jika True, cetak bentuk output setelah setiap layer.

        Returns:
            np.ndarray: Output final dari model. 
        """
        x = input_data
        if verbose:
            print(f"  Forward Pass FS - Input awal shape: {x.shape}")
        for i, layer_obj in enumerate(self.layers):
            start_layer_time = time.time()
            # Memanggil method forward dari objek layer
            x = layer_obj.forward(x)
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
        """
        Memuat bobot dari model Keras yang sudah dilatih ke layer-layer from scratch yang sesuai. 
        Ini adalah bagian yang paling krusial dan mungkin memerlukan penyesuaian tergantung arsitektur persis dan bagaimana Keras menyimpan bobot.

        Args:
            keras_model (tf.keras.Model): Model Keras yang bobotnya akan diambil.
        """
        keras_layer_idx = 0
        fs_layer_idx = 0

        print("\nMemulai pemuatan bobot Keras ke model FromScratch...")

        # Loop selama masih ada layer di kedua model yang belum diproses
        while fs_layer_idx < len(self.layers) and keras_layer_idx < len(keras_model.layers):
            fs_layer = self.layers[fs_layer_idx]
            keras_layer = keras_model.layers[keras_layer_idx]

            # Debugging:
            # print(f"  Mencocokkan FS Layer {fs_layer_idx}: {fs_layer.__class__.__name__} | Keras Layer {keras_layer_idx}: {keras_layer.name} ({keras_layer.__class__.__name__})")

            weights_loaded_for_fs_layer = False

            # Mencocokkan layer FS dengan layer Keras yang memiliki bobot
            if isinstance(fs_layer, Conv2DLayerFS) and isinstance(keras_layer, tf.keras.layers.Conv2D):
                if len(keras_layer.get_weights()) == 2:  # weights and biases
                    w, b = keras_layer.get_weights()
                    fs_layer.load_weights(w, b)
                    print(
                        f"    Bobot dimuat: FS {fs_layer.__class__.__name__} <- Keras {keras_layer.name}")
                    weights_loaded_for_fs_layer = True
                    keras_layer_idx += 1  # Keras layer ini sudah digunakan
                elif len(keras_layer.get_weights()) == 1:  # weights only (use_bias=False)
                    w = keras_layer.get_weights()[0]
                    b = np.zeros(w.shape[-1])  # Asumsi bias nol jika tidak ada
                    fs_layer.load_weights(w, b)
                    print(
                        f"    Bobot (tanpa bias eksplisit) dimuat: FS {fs_layer.__class__.__name__} <- Keras {keras_layer.name}")
                    weights_loaded_for_fs_layer = True
                    keras_layer_idx += 1
                else:  # Format bobot tidak sesuai
                    print(
                        f"    Peringatan: Keras Conv2D {keras_layer.name} memiliki format bobot tak terduga ({len(keras_layer.get_weights())} elemen). Dilewati.")
                    # Anggap Keras layer ini tidak relevan atau coba Keras layer berikutnya
                    keras_layer_idx += 1

            elif isinstance(fs_layer, DenseLayerFS) and isinstance(keras_layer, tf.keras.layers.Dense):
                if len(keras_layer.get_weights()) == 2:
                    w, b = keras_layer.get_weights()
                    fs_layer.load_weights(w, b)
                    print(
                        f"    Bobot dimuat: FS {fs_layer.__class__.__name__} <- Keras {keras_layer.name}")
                    weights_loaded_for_fs_layer = True
                    keras_layer_idx += 1
                elif len(keras_layer.get_weights()) == 1:  # weights only (use_bias=False)
                    w = keras_layer.get_weights()[0]
                    b = np.zeros(w.shape[-1])
                    fs_layer.load_weights(w, b)
                    print(
                        f"    Bobot (tanpa bias eksplisit) dimuat: FS {fs_layer.__class__.__name__} <- Keras {keras_layer.name}")
                    weights_loaded_for_fs_layer = True
                    keras_layer_idx += 1
                else:
                    print(
                        f"    Peringatan: Keras Dense {keras_layer.name} memiliki format bobot tak terduga. Dilewati.")
                    keras_layer_idx += 1

            # Jika layer FS tidak punya bobot (ReLU, Pooling, Flatten)
            # atau layer FS adalah layer berbobot tapi Keras layer saat ini tidak cocok
            if not weights_loaded_for_fs_layer:
                # Cek apakah Keras layer saat ini adalah layer non-trainable yang bisa dilewati
                # seperti InputLayer, Activation (jika ReLU FS sudah ada), atau Pooling (jika FS pooling sudah ada)
                if isinstance(keras_layer, (tf.keras.layers.InputLayer, tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                    # print(f"    Melewati Keras layer non-trainable/utility: {keras_layer.name}")
                    keras_layer_idx += 1
                    continue  # Coba lagi dengan Keras layer berikutnya untuk FS layer yang sama
                elif isinstance(fs_layer, ReLULayerFS) and isinstance(keras_layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation)):
                    # print(f"    FS ReLULayerFS cocok dengan Keras {keras_layer.name}. Tidak ada bobot.")
                    keras_layer_idx += 1  # Keduanya maju karena ada kecocokan konseptual
                elif isinstance(fs_layer, MaxPooling2DLayerFS) and isinstance(keras_layer, tf.keras.layers.MaxPooling2D):
                    # print(f"    FS MaxPooling2DLayerFS cocok dengan Keras {keras_layer.name}.")
                    keras_layer_idx += 1
                elif isinstance(fs_layer, AveragePooling2DLayerFS) and isinstance(keras_layer, tf.keras.layers.AveragePooling2D):
                    # print(f"    FS AveragePooling2DLayerFS cocok dengan Keras {keras_layer.name}.")
                    keras_layer_idx += 1
                elif isinstance(fs_layer, FlattenLayerFS) and isinstance(keras_layer, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalMaxPooling2D)):
                    # print(f"    FS FlattenLayerFS cocok dengan Keras {keras_layer.name}.")
                    keras_layer_idx += 1
                # Jika tidak ada kecocokan sama sekali, FS layer akan maju di akhir loop ini.
                # Keras layer tidak maju di sini kecuali ada alasan spesifik (seperti di atas).
                # Jika FS layer adalah layer berbobot tapi tidak cocok, Keras layer akan dilewati di blok 'else' berikutnya.

            # Majukan FS layer index jika bobotnya sudah dimuat atau jika itu layer tanpa bobot
            if weights_loaded_for_fs_layer or isinstance(fs_layer, (ReLULayerFS, MaxPooling2DLayerFS, AveragePooling2DLayerFS, FlattenLayerFS)):
                fs_layer_idx += 1
            elif not isinstance(fs_layer, (Conv2DLayerFS, DenseLayerFS)):
                # Jika FS layer adalah tipe lain tanpa bobot yang belum ditangani di atas
                # print(f"    FS Layer {fs_layer.__class__.__name__} adalah tipe lain tanpa bobot. FS maju.")
                fs_layer_idx += 1
            # else: Keras layer akan dicoba lagi dengan fs_layer yang sama jika tidak ada bobot yang dimuat dan fs_layer adalah trainable

        if fs_layer_idx < len(self.layers):
            print(
                f"Peringatan: Tidak semua layer FromScratch ({len(self.layers)}) mendapatkan bobot atau diproses ({fs_layer_idx} diproses).")
            for i in range(fs_layer_idx, len(self.layers)):
                print(
                    f"  Layer FS yang tidak diproses: {self.layers[i].__class__.__name__}")
        if keras_layer_idx < len(keras_model.layers):
            print(
                f"Peringatan: Tidak semua layer Keras ({len(keras_model.layers)}) digunakan untuk bobot ({keras_layer_idx} diproses).")
            for i in range(keras_layer_idx, len(keras_model.layers)):
                print(
                    f"  Layer Keras yang tidak terpakai: {keras_model.layers[i].name}")

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
