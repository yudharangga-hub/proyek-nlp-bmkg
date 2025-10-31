Dashboard Analisis Sentimen & Visualisasi NLP Ulasan Aplikasi BMKG

Dashboard analitik interaktif yang dibangun menggunakan Flask untuk membedah dan memvisualisasikan ulasan pengguna aplikasi Info BMKG. Proyek ini mengubah data teks mentah menjadi wawasan strategis melalui berbagai teknik NLP, machine learning, dan visualisasi data.

Proyek ini merupakan bagian dari Ujian Tengah Semester (UTS) mata kuliah Advanced NLP (Dosen: Dr. Sajarwo Anggai, S.ST., M.T) oleh Yudha Rangga Wulung Pura (241012000151), Mahasiswa S2 Teknik Informatika Universitas Pamulang.

Fitur Utama & Galeri Proyek

Aplikasi web ini menyajikan berbagai analisis dalam antarmuka multi-tab.

1. Ekstraksi Fitur Dasar (BoW, TF-IDF, Word Embeddings)

Tab ini menampilkan "tulang punggung" dari analisis NLP. Pengguna dapat melihat data teks yang telah diubah menjadi representasi numerik menggunakan Bag of Words (BoW), TF-IDF, dan Word Embeddings (Word2Vec) untuk menemukan kata-kata yang serupa secara kontekstual.

2. Topic Modeling Otomatis (LDA & BERTopic)

Untuk memahami "apa yang paling sering dibicarakan pengguna", dua model topic modeling diimplementasikan:

LDA (Latent Dirichlet Allocation): Metode statistik klasik untuk menemukan tema-tema umum.

BERTopic: Metode modern yang menggunakan embeddings SBERT untuk menemukan topik yang lebih akurat secara kontekstual.

3. Visualisasi Word Cloud

Visualisasi ringkas dari kata-kata yang paling sering muncul di seluruh dataset, memberikan gambaran instan tentang fokus utama ulasan.

4. Evaluasi Model & Analisis Sentimen (Supervised)

Ini adalah inti dari dashboard yang menunjukkan performa model Logistic Regression yang telah dilatih:

Laporan Evaluasi: Menampilkan Confusion Matrix visual untuk melihat kesalahan prediksi secara detail.

Simulasi Prediksi: Model yang sudah dilatih digunakan untuk memprediksi sentimen dari data yang diunggah, disajikan dalam bentuk grafik donat (terlihat di Sentimen.png).

5. Visualisasi Klastering & Distribusi (Unsupervised)

Tab Visualisasi Sentimen: Tab ini berfokus pada hasil analisis data .txt yang diunggah:

Silhouette Score (K-Means): Mengukur seberapa "alami" data terkelompok menggunakan gauge meter. Skor rendah menunjukkan data yang tumpang tindih—memvalidasi perlunya model supervised.

Distribusi Sentimen: Grafik batang horizontal modern yang menampilkan persentase dan jumlah pasti dari prediksi.

6. Upload Data

Pengguna dapat mengunggah dataset .txt mereka sendiri untuk dianalisis secara real-time oleh semua model.

Alur Kerja (Workflow) Proyek

Persiapan Data: Aplikasi ini memiliki dua alur data:

Data Latih (Offline/Lokal): File Data Komen Apps InfoBMKG - dataset_bmkg_reviews.csv harus ada di folder utama saat app.py dijalankan. File ini digunakan hanya untuk melatih model Logistic Regression.

Data Analisis (Online/Upload): File .txt diunggah oleh pengguna melalui interface web. Data inilah yang akan dianalisis oleh semua fitur.

Pra-Pemrosesan (NLP Pipeline): Semua teks melalui proses Cleaning, Tokenization, Stopword Removal (NLTK + Custom), dan Stemming (Sastrawi).

Pelatihan Model (Saat Startup): Saat app.py dijalankan, ia akan mencari file .csv dan melatih model Logistic Regression. Jika file .csv tidak ditemukan, server tetap berjalan, namun fitur sentimen akan dinonaktifkan.

Analisis & Ekstraksi Fitur (Saat Upload): Saat pengguna mengunggah file .txt, data tersebut diproses untuk menghasilkan: II, BoW, TF-IDF, Word2Vec, SBERT Embeddings, LDA, dan BERTopic.

Simulasi Prediksi (Saat Upload): Model Logistic Regression yang sudah ada di memori digunakan untuk memprediksi sentimen pada data .txt yang baru diunggah.

Visualisasi Flask: Semua hasil dikirim ke frontend dan ditampilkan secara interaktif di dashboard web.

Teknologi yang Digunakan

Backend: Flask, Pandas, Scikit-learn (Logistic Regression, TF-IDF, K-Means, Silhouette Score), NLTK, Sastrawi, Gensim (Word2Vec), Sentence-Transformers, BERTopic, WordCloud.

Frontend: HTML5, CSS3, JavaScript (untuk interaktivitas tab dan visualisasi dinamis).

Visualisasi: Matplotlib (untuk Confusion Matrix) & CSS/JavaScript (untuk grafik batang, gauge meter, dan efek "The Matrix").

Struktur Proyek

Berdasarkan struktur folder Anda yang sebenarnya:

Proyek_ProposalUTS_NLP_YudhaRWP-SentimenAnalisis/
├── app.py                 # File utama Flask (Backend)
├── requirements.txt       # Daftar library Python yang diperlukan
├── .gitignore             # File untuk mengabaikan folder environment
├── README.md              # File ini
├── Data Komen Apps InfoBMKG - dataset_bmkg_reviews.csv # (WAJIB ADA)
├── dataset_bmkg_reviews.txt                            # (Opsional, untuk diunggah)
├── screenshots/           # Folder untuk menyimpan tangkapan layar
│   ├── BoW.png
│   ├── Sentimen.png
│   ├── bertopic.png
│   ├── conf-matrix.png
│   ├── lda.png
│   ├── siluet-skor.png
│   ├── tf-idf.png
│   ├── upload.png
│   ├── word-cloud.png
│   └── Word-Embed.png
├── static/
│   ├── css/
│   │   └── style.css      # File styling
│   └── images/
│       ├── UNPAM_logo1.png
│       └── wordcloud_general.png
└── templates/
    └── index.html         # File utama HTML (Frontend)


Cara Menjalankan Proyek Secara Lokal

Clone Repositori:

git clone [https://github.com/yudharangga-hub/proyek-nlp-bmkg.git](https://github.com/yudharangga-hub/proyek-nlp-bmkg.git)
cd Proyek_ProposalUTS_NLP_YudhaRWP-SentimenAnalisis


Buat Virtual Environment:
(Direkomendasikan untuk menghindari konflik library)

# Menggunakan conda (jika Anda menggunakan Anaconda)
conda create -n nlp-bmkg python=3.9
conda activate nlp-bmkg

# Atau menggunakan venv (bawaan Python)
# python -m venv venv
# source venv/bin/activate  # (Di Mac/Linux)
# .\venv\Scripts\activate    # (Di Windows)


Install Library yang Dibutuhkan:
Gunakan file requirements.txt yang telah disediakan.

pip install -r requirements.txt


(Catatan: torch dan sentence-transformers mungkin memakan waktu untuk diunduh).

Siapkan Dataset Latih (WAJIB untuk Fitur Sentimen):

Pastikan file Data Komen Apps InfoBMKG - dataset_bmkg_reviews.csv ada di folder utama (sejajar dengan app.py).

Tanpa file ini, server akan tetap berjalan, tetapi semua fitur "Analisis Sentimen" dan "Visualisasi Sentimen" tidak akan berfungsi.

Jalankan Aplikasi Flask:

python app.py


(Aplikasi akan dimulai dalam keadaan "kosong".)

Buka di Browser dan Unggah Data:

Buka http://127.0.0.1:5000 di browser Anda.

Navigasi ke tab "Upload Dataset".

Unggah file .txt (seperti dataset_bmkg_reviews.txt jika ada) untuk memulai analisis.

Link Repositori

https://github.com/yudharangga-hub/proyek-nlp-bmkg.git