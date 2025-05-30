import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoaderCIFAR10:
    """
    Mengelola dataset CIFAR-10, termasuk pemuatan, pembagian menjadi 
    set training (40k), validasi (10k), dan testing (10k), 
    serta pra-pemrosesan dasar.
    """

    def __init__(self, random_state_split: int = 42):
        """
        Konstruktor. Memuat dataset CIFAR-10, membagi data training menjadi 
        training baru dan set validasi (rasio 4:1), dan melakukan pra-pemrosesan.

        Args:
            random_state_split (int): Random state untuk train_test_split agar pembagian konsisten.
        """
        self.num_classes = 10
        self.random_state_split = random_state_split

        # Memuat data mentah
        (self.original_train_images, self.original_train_labels), \
            (self.test_images, self.test_labels) = self._load_data()  # type: ignore

        # Membagi set training asli menjadi training baru dan set validasi
        self._split_validation()

        # Pra-pemrosesan data
        self._preprocess_data()

        print("CIFAR-10 dataset loaded and preprocessed:")
        print(f"Training images shape: {self.train_images.shape}")
        print(f"Training labels shape: {self.train_labels.shape}")
        print(f"Validation images shape: {self.validation_images.shape}")
        print(f"Validation labels shape: {self.validation_labels.shape}")
        print(f"Test images shape: {self.test_images.shape}")
        print(f"Test labels shape: {self.test_labels.shape}")

    def _load_data(self) -> tuple:
        """
        Method internal untuk memuat data mentah CIFAR-10.
        """
        return tf.keras.datasets.cifar10.load_data()

    def _split_validation(self) -> None:
        """
        Method internal untuk membagi set training asli menjadi set training 
        yang lebih kecil dan set validasi.
        Spesifikasi: 40k train data, 10k validation data, dan 10k test data.
        Rasio pembagian training asli menjadi training baru dan validasi adalah 4:1.
        """
        self.train_images, self.validation_images, \
            self.train_labels, self.validation_labels = train_test_split(
                self.original_train_images,
                self.original_train_labels,
                test_size=0.2,  # 20% dari 50k adalah 10k untuk validasi
                random_state=self.random_state_split,
                stratify=self.original_train_labels  # Menjaga proporsi kelas
            )

    def _preprocess_data(self) -> None:
        """
        Method internal untuk melakukan pra-pemrosesan pada gambar.
        Normalisasi nilai piksel ke rentang [0, 1].
        Label tidak perlu di-one-hot encode jika menggunakan SparseCategoricalCrossentropy.
        """
        self.train_images = self.train_images.astype('float32') / 255.0
        self.validation_images = self.validation_images.astype(
            'float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0

        # Konversi label ke integer jika belum (load_data Keras sudah mengembalikan integer)
        self.train_labels = self.train_labels.astype('int32')
        self.validation_labels = self.validation_labels.astype('int32')
        self.test_labels = self.test_labels.astype('int32')

    def get_train_data(self) -> tuple:
        """
        Mengembalikan data training yang sudah diproses.
        """
        return self.train_images, self.train_labels

    def get_validation_data(self) -> tuple:
        """
        Mengembalikan data validasi yang sudah diproses.
        """
        return self.validation_images, self.validation_labels

    def get_test_data(self) -> tuple:
        """
        Mengembalikan data testing yang sudah diproses.
        """
        return self.test_images, self.test_labels

    def get_input_shape(self) -> tuple:
        """
        Mengembalikan shape dari satu sampel gambar input.
        """
        if self.train_images is not None and len(self.train_images) > 0:
            return self.train_images.shape[1:]
        elif self.test_images is not None and len(self.test_images) > 0:
            return self.test_images.shape[1:]
        else:
            # Default CIFAR-10 shape jika data belum dimuat sepenuhnya
            return (32, 32, 3)


# Contoh penggunaan:
if __name__ == '__main__':
    cifar_loader = DataLoaderCIFAR10()

    train_images, train_labels = cifar_loader.get_train_data()
    val_images, val_labels = cifar_loader.get_validation_data()
    test_images, test_labels = cifar_loader.get_test_data()

    print(f"\nContoh shape setelah diambil dari getter:")
    print(
        f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
    print(
        f"Validation images shape: {val_images.shape}, Validation labels shape: {val_labels.shape}")
    print(
        f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")
    print(f"Input shape for model: {cifar_loader.get_input_shape()}")
    print(f"Number of classes: {cifar_loader.num_classes}")

    # Verifikasi pembagian jumlah data
    assert len(train_images) == 40000, "Jumlah data training tidak sesuai"
    assert len(val_images) == 10000, "Jumlah data validasi tidak sesuai"
    assert len(test_images) == 10000, "Jumlah data test tidak sesuai"
    print("\nJumlah data pada setiap split sudah sesuai dengan spesifikasi.")
