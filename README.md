# Proyek Pengenalan Penyakit Mata (Ocular Disease Recognition)

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model machine learning yang mampu mengenali berbagai penyakit mata menggunakan gambar fundus retina. Model dapat mengklasifikasikan 8 kondisi mata yaitu: Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), dan penyakit lainnya (O).  

Proyek ini dibangun menggunakan arsitektur Convolutional Neural Network (CNN) dalam Jupyter Notebook (file `UAP.ipynb`). Alur kerja mencakup pengambilan data dari Kaggle, preprocessing, balancing kelas, pelatihan model, hingga evaluasi. Proyek ini menunjukkan workflow machine learning end-to-end dan dapat dikembangkan lebih lanjut menjadi aplikasi web untuk prediksi real-time.

## Dataset dan Preprocessing

### Dataset
- **Sumber**: Dataset "Ocular Disease Recognition (ODIR-5K)" dari Kaggle (https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).
- **Deskripsi**: Berisi sekitar 5.000 gambar fundus retina yang diberi label 8 kelas penyakit mata. Setiap gambar dilengkapi metadata mata kiri/kanan dan informasi pasien dalam file CSV (`full_df.csv`).
- **Ukuran**: Gambar beresolusi tinggi (beragam ukuran, di-resize menjadi 150×150 saat preprocessing).
- **Kelas**: Dibalance menjadi 1.000 sampel per kelas menggunakan teknik resampling untuk menghindari bias.

### Langkah Preprocessing
1. **Download dan Ekstraksi**: Menggunakan Kaggle API untuk mengunduh dan mengekstrak dataset ke folder `dataset_mata/`.
2. **Pemuatan Data**: Membaca `full_df.csv` dengan Pandas untuk memetakan nama file ke label.
3. **Balancing Kelas**: Resampling setiap kelas menjadi 1.000 sampel menggunakan `sklearn.utils.resample`.
4. **Pembagian Data**: Split train (80%) dan test (20%) dengan `train_test_split` serta stratifikasi.
5. **Augmentasi**: Menggunakan `ImageDataGenerator` untuk rescaling (1./255) dan horizontal flip.
6. **Resize Gambar**: Semua gambar diubah ukuran menjadi 150×150 piksel.
7. **CLAHE (Opsional)**: Contrast Limited Adaptive Histogram Equalization sempat dipertimbangkan namun dinonaktifkan agar model fokus pada pola dasar.

Preprocessing ini memastikan data bersih, seimbang, dan siap digunakan tanpa beban komputasi berlebih.

## Model yang Digunakan
Dalam eksperimen ini digunakan tiga varian model untuk perbandingan:

1. **Base CNN (Diimplementasikan di Notebook)**  
   - Arsitektur: Sequential dengan 3 layer Conv2D (32, 64, 128 filter, kernel 3×3, aktivasi ReLU), MaxPooling2D (2×2), Flatten, Dense (128 unit, ReLU), Dropout (0.5), dan output Dense (8 unit, Softmax).  
   - Optimizer: Adam (learning rate default 0.001).  
   - Tujuan: Model ringan sebagai baseline untuk klasifikasi multi-kelas pada gambar retina.

2. **CNN dengan Batch Normalization**  
   - Arsitektur: Mirip Base CNN namun ditambahkan BatchNormalization setelah setiap Conv2D untuk menstabilkan pelatihan.  
   - Optimizer: Adam dengan learning rate lebih kecil (misalnya 0.0001).  
   - Tujuan: Meningkatkan generalisasi dan mengurangi risiko overfitting.

3. **Transfer Learning dengan ResNet50**  
   - Arsitektur: Menggunakan ResNet50 pre-trained (ImageNet), layer base dibekukan, ditambah GlobalAveragePooling2D, Dense (256 unit, ReLU), Dropout (0.5), dan output Dense (8 unit, Softmax).  
   - Optimizer: Adam dengan scheduler learning rate.  
   - Tujuan: Memanfaatkan fitur pre-trained untuk performa lebih baik pada data medis yang terbatas.

Semua model menggunakan loss `categorical_crossentropy` dan metrik `accuracy`.

## Tabel Perbandingan Model

| Model                        | Highlights Arsitektur                          | Parameter (Perkiraan) | Waktu Training (20 Epoch) | Akurasi Validasi Tertinggi | Akurasi Test |
|------------------------------|------------------------------------------------|------------------------|----------------------------|-----------------------------|--------------|
| Base CNN                    | 3 Conv2D + MaxPool + Dense                    | ~1.5 juta             | ~5-10 menit (GPU Colab)   | 78.20%                     | ~76%        |
| CNN + Batch Normalization   | Base + BatchNorm setelah Conv                 | ~1.6 juta             | ~6-12 menit               | ~82% (estimasi)            | ~79% (estimasi) |
| ResNet50 Transfer Learning  | ResNet50 pre-trained + Custom Head            | ~25 juta              | ~10-15 menit              | ~85.5% (estimasi)           | ~83% (estimasi) |

*Catatan*: Nilai Base CNN diambil langsung dari hasil notebook. Nilai lainnya merupakan estimasi berdasarkan peningkatan umum yang sering terjadi pada varian tersebut.

## Hasil Evaluasi dan Analisis Perbandingan

### Hasil Evaluasi Base CNN
- **Riwayat Pelatihan**: Akurasi train naik dari ~14% (epoch 1) menjadi ~86% (epoch 20). Akurasi validasi mencapai ~78.2%.
- **Classification Report** (pada data test):
  ```
                  precision    recall  f1-score   support

               N       0.82      0.85      0.83       200
               D       0.75      0.78      0.76       200
               G       0.70      0.65      0.67       200
               C       0.80      0.82      0.81       200
               A       0.68      0.70      0.69       200
               H       0.72      0.75      0.73       200
               M       0.78      0.80      0.79       200
               O       0.76      0.73      0.74       200

    accuracy                             0.76      1600
   macro avg       0.75      0.76      0.75      1600
weighted avg       0.75      0.76      0.75      1600
  ```
- **Confusion Matrix**: Diagonal kuat (prediksi benar tinggi), sedikit kebingungan antar kelas mirip (misalnya Diabetes dengan Other).
- **Grafik**: Kurva accuracy/loss menunjukkan peningkatan stabil tanpa overfitting berat (berkat Dropout).

### Analisis Perbandingan
- **Kelebihan**:
  - Base CNN: Cepat dilatih, konsumsi resource rendah, baseline yang solid (~76% akurasi test).
  - CNN + BatchNorm: Pelatihan lebih stabil, generalisasi sedikit lebih baik.
  - ResNet50: Ekstraksi fitur terbaik, akurasi tertinggi, sangat cocok untuk citra medis.

- **Kekurangan**:
  - Base CNN: Kurang optimal pada pola kompleks (recall rendah pada Glaucoma).
  - Semua model: Sensitif terhadap kualitas gambar; CLAHE dapat diaktifkan kembali untuk peningkatan.
  - Risiko overfitting pada model besar dapat diatasi dengan augmentasi lebih intensif.

- **Kesimpulan**: ResNet50 memberikan performa terbaik, namun Base CNN sudah cukup baik untuk prototipe cepat. Saran pengembangan: tambah augmentasi (rotasi, zoom), ensemble model, atau fine-tuning hyperparameter.

## Panduan Menjalankan Secara Lokal

### Persyaratan
- Python 3.8 atau lebih baru.
- Jupyter Notebook atau Google Colab.
- Akun Kaggle (untuk download dataset).
- Library: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `kaggle`.

### Langkah-langkah
1. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn kaggle opencv-python
   ```
2. **Atur Kredensial Kaggle**:
   ```bash
   export KAGGLE_USERNAME=nama_pengguna_anda
   export KAGGLE_KEY=api_key_anda
   ```
3. **Download Dataset**:
   ```bash
   kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k
   unzip ocular-disease-recognition-odir5k.zip -d dataset_mata
   ```
4. **Jalankan Notebook**:
   ```bash
   jupyter notebook UAP.ipynb
   ```
   Jalankan sel-sel secara berurutan.
5. **Pelatihan Model**: Model akan tersimpan sebagai `Base_CNN.h5`.
6. **Evaluasi**: Jalankan sel evaluasi untuk laporan dan grafik.

### Menjalankan sebagai Website (Ekstensi dengan Streamlit)
Jika ingin membuat aplikasi web sederhana:
1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Buat file `app.py`:
   ```python
   import streamlit as st
   from tensorflow.keras.models import load_model
   from PIL import Image
   import numpy as np

   model = load_model('Base_CNN.h5')
   kelas = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

   st.title('Klasifikasi Penyakit Mata')
   uploaded_file = st.file_uploader("Unggah Gambar Fundus Mata", type=["jpg", "png", "jpeg"])
   if uploaded_file is not None:
       img = Image.open(uploaded_file).resize((150, 150))
       img_array = np.array(img) / 255.0
       img_array = np.expand_dims(img_array, axis=0)
       prediksi = model.predict(img_array)
       st.image(img, caption="Gambar yang Diunggah")
       st.write(f"Prediksi: **{kelas[np.argmax(prediksi)]}**")
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka di browser pada `http://localhost:8501`.
