# Tubes 2 ML: Forward Propagation Implementations (CNN, RNN, LSTM)

Repositori ini merupakan implementasi tugas besar kedua (Tubes 2) untuk mata kuliah Machine Learning. Fokus utama tugas ini adalah mengimplementasikan forward propagation dari beberapa arsitektur neural network (CNN, Simple RNN, dan LSTM), baik dengan library Keras maupun from scratch (tanpa library deep learning), serta melakukan analisis eksperimen hyperparameter pada tiap model.

## Daftar Isi

- [Deskripsi Singkat](#deskripsi-singkat)
- [Spesifikasi Tugas](#spesifikasi-tugas)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Simple Recurrent Neural Network (RNN)](#simple-recurrent-neural-network-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
- [Cara Setup & Menjalankan Program](#cara-setup--menjalankan-program)
- [Pembagian Tugas Anggota Kelompok](#pembagian-tugas-anggota-kelompok)

---

## Deskripsi Singkat

Repositori ini berisi:
- Implementasi forward propagation untuk CNN, Simple RNN, dan LSTM baik menggunakan Keras maupun from scratch.
- Eksperimen variasi hyperparameter dan analisis pengaruhnya terhadap performa model.
- Notebook interaktif untuk training, evaluasi, hingga pengujian forward propagation custom.
- Analisis hasil dalam bentuk grafik dan laporan singkat.

---

## Spesifikasi Tugas

### Convolutional Neural Network (CNN)

- **Dataset:** CIFAR-10 (split menjadi train: 40k, validasi: 10k, test: 10k)
- **Model minimal:** Conv2D, Pooling (Max/Average), Flatten/GlobalPooling, Dense
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Variasi eksperimen:**
  - Jumlah layer konvolusi (3 variasi)
  - Banyak filter per layer konvolusi (3 variasi)
  - Ukuran filter per layer konvolusi (3 variasi)
  - Jenis pooling layer (Max vs Average)
- **Evaluasi:** Macro F1-score, grafik training & validation loss per epoch
- **Forward Propagation From Scratch:**
  - Mampu membaca bobot hasil pelatihan Keras
  - Modular untuk setiap layer
  - Perbandingan hasil prediksi Keras vs from scratch menggunakan data test

### Simple Recurrent Neural Network (RNN)

- **Dataset:** NusaX-Sentiment (Bahasa Indonesia)
- **Preprocessing:** Tokenization (TextVectorization Keras), Embedding (Keras Embedding Layer)
- **Model minimal:** Embedding, Bidirectional/Unidirectional RNN, Dropout, Dense
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Variasi eksperimen:**
  - Jumlah layer RNN (3 variasi)
  - Banyak cell RNN per layer (3 variasi)
  - Jenis arah layer (bidirectional vs unidirectional)
- **Evaluasi:** Macro F1-score, grafik training & validation loss per epoch
- **Forward Propagation From Scratch:**
  - Mampu membaca bobot hasil pelatihan Keras
  - Modular untuk setiap layer
  - Perbandingan hasil prediksi Keras vs from scratch pada data test

### Long Short-Term Memory (LSTM)

- **Dataset:** NusaX-Sentiment (Bahasa Indonesia)
- **Preprocessing:** Tokenization dan Embedding (seperti pada RNN)
- **Model minimal:** Embedding, Bidirectional/Unidirectional LSTM, Dropout, Dense
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Variasi eksperimen:**
  - Jumlah layer LSTM (3 variasi)
  - Banyak cell LSTM per layer (3 variasi)
  - Jenis arah layer (bidirectional vs unidirectional)
- **Evaluasi:** Macro F1-score, grafik training & validation loss per epoch
- **Forward Propagation From Scratch:**
  - Mampu membaca bobot hasil pelatihan Keras
  - Modular untuk setiap layer
  - Perbandingan hasil prediksi Keras vs from scratch pada data test

---

## Cara Setup & Menjalankan Program

1. **Buka file notebook (ipynb) di Google Colab**
2. **Jalankan seluruh cell secara berurutan**
   - Semua dependensi akan otomatis terinstall jika belum tersedia.
   - Ikuti instruksi pada notebook untuk melakukan training, evaluasi, hingga pengujian forward propagation custom.
3. **Pastikan koneksi internet aktif untuk akses dataset dan dependensi**

**Catatan:** Tidak ada setup lokal khusus. Cukup buka notebook di Google Colab dan jalankan.

---

## Pembagian Tugas Anggota Kelompok

| Nama Anggota      | NIM           | Kontribusi Utama                                          |
|-------------------|---------------|-----------------------------------------------------------|
| [Muhammad Dzaki Arta]  | [13522149] | Implementasi & eksperimen CNN, forward propagation CNN    |
| [Samy Muhammad Haikal]  | [13522151] | Implementasi & eksperimen Simple RNN, forward propagation RNN |
| [Muhammad Roihan]  | [13522152] | Implementasi & eksperimen LSTM, forward propagation LSTM  |


---
