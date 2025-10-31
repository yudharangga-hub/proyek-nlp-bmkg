import os
import re
import string
import io
import time
import random
import nltk
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set backend non-interaktif
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, get_flashed_messages, session # Tambahkan session
)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# --- IMPORT BARU ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# --- AKHIR IMPORT BARU ---
import matplotlib.pyplot as plt # Import untuk plot confusion matrix
import base64 # Untuk mengirim gambar plot ke HTML
import traceback # Untuk debugging error

# --- Unduh Data NLTK jika diperlukan ---
print("Memeriksa kelengkapan data NLTK...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    print("Data NLTK lengkap.")
except LookupError:
    print("Data NLTK tidak lengkap, mengunduh...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("Unduhan NLTK selesai.")

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ganti_dengan_kunci_rahasia_yang_kuat_dan_unik' # GANTI INI!
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- Inisialisasi Preprocessing Tools ---
print("Menyiapkan tools preprocessing (Stopwords & Stemmer)...")
try:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    custom_stopwords = {'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo',
                        'kalo', 'amp', 'biar', 'bikin', 'bilang',
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah'}
    stop_words = set(stopwords.words('indonesian')).union(custom_stopwords)
    print("Tools preprocessing siap.")
except Exception as e:
    print(f"FATAL ERROR: Gagal menyiapkan tools preprocessing: {e}")
    exit()

# --- Inisialisasi Model Sentence Transformer ---
print("Memuat model Sentence Transformer (mungkin perlu download)...")
try:
    SEMANTIC_MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Model Sentence Transformer berhasil dimuat.")
except Exception as e:
    print(f"WARNING: Gagal memuat model Sentence Transformer: {e}")
    print("Fitur Pencarian Semantik dan BERTopic mungkin tidak berfungsi.")
    SEMANTIC_MODEL = None

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    """Membersihkan teks: lowercase, hapus angka & punct, tokenize, stopwords, stem."""
    if not isinstance(text, str): return []
    try:
        text = text.lower()
        text = re.sub(r'\d+', '', text) # Hapus angka
        text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
        text = text.strip()
        tokens = word_tokenize(text)
        processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(stemmer.stem(token)) > 1]
        processed_tokens = [token for token in processed_tokens if token] # Hapus token kosong
        return processed_tokens
    except Exception as e: print(f"Error preprocessing text: '{text[:50]}...' - {e}"); return []

def preprocess_text_for_tfidf(text):
    """Wrapper: Memanggil preprocess_text dan menggabungkan token menjadi string."""
    tokens = preprocess_text(text); return " ".join(tokens)

# --- Fungsi Pemrosesan Data ---
def process_data_from_stream(file_stream):
    """Memproses data dari file stream yang di-upload (.txt)."""
    print("Memproses data dari file stream yang di-upload...")
    documents = {}; processed_docs_as_tokens = []; processed_docs_as_strings = []
    doc_id_counter = 1
    try:
        stream_reader = io.TextIOWrapper(file_stream, encoding='utf-8', errors='ignore')
        for line in stream_reader:
            text = line.strip();
            if not text: continue
            doc_name = f"Komentar_{doc_id_counter}"; documents[doc_name] = text
            tokens = preprocess_text(text); processed_string = " ".join(tokens)
            processed_docs_as_tokens.append(tokens); processed_docs_as_strings.append(processed_string)
            doc_id_counter += 1
        print(f"Total {doc_id_counter - 1} komentar selesai diproses dari upload.")
        return documents, processed_docs_as_tokens, processed_docs_as_strings
    except Exception as e:
        print(f"Error saat memproses file stream: {e}")
        try: flash(f"Error saat memproses file: {e}", "danger")
        except RuntimeError: print("Tidak dalam konteks request untuk flash error.")
        return None, None, None

def process_data_from_path(file_path):
    """Memproses data dari path file .txt."""
    print(f"Membaca dan memproses dataset dari: {file_path}...")
    documents = {}; processed_docs_as_tokens = []; processed_docs_as_strings = []
    doc_id_counter = 1
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                text = line.strip()
                if not text: continue
                doc_name = f"Komentar_{doc_id_counter}"; documents[doc_name] = text
                tokens = preprocess_text(text); processed_string = " ".join(tokens)
                processed_docs_as_tokens.append(tokens); processed_docs_as_strings.append(processed_string)
                doc_id_counter += 1
        print(f"Total {doc_id_counter - 1} komentar selesai diproses dari file.")
        return documents, processed_docs_as_tokens, processed_docs_as_strings
    except FileNotFoundError:
        print(f"INFO: File dataset default tidak ditemukan di '{file_path}'. Melanjutkan tanpa data awal.")
        return None, None, None
    except Exception as e:
        print(f"Error saat membaca file {file_path}: {e}")
        return None, None, None

# --- Fungsi Pembangun Model ---
def build_inverted_index(token_lists):
    if not token_lists: return {}
    print("Membangun Inverted Index...")
    inverted_index = {}
    for i, tokens in enumerate(token_lists):
        if i < len(DOC_NAMES_LIST):
            doc_name = DOC_NAMES_LIST[i]
            unique_tokens = set(tokens)
            for token in unique_tokens:
                inverted_index.setdefault(token, []).append(doc_name)
    return dict(sorted(inverted_index.items()))

def calculate_bow(processed_strings):
    if not processed_strings: return None, None
    print("Melatih model Bag of Words (BoW)...")
    try:
        bow_vectorizer = CountVectorizer(max_features=1500, ngram_range=(1,1))
        bow_matrix = bow_vectorizer.fit_transform(processed_strings)
        print("Model BoW selesai dilatih.")
        return bow_vectorizer, bow_matrix
    except Exception as e: print(f"Error melatih BoW: {e}"); return None, None

def calculate_tfidf(processed_strings):
    if not processed_strings: return None, None
    print("Melatih model TF-IDF (Non-Sentimen)...")
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,1))
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_strings)
        print("Model TF-IDF (Non-Sentimen) selesai dilatih.")
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e: print(f"Error melatih TF-IDF (Non-Sentimen): {e}"); return None, None

def train_word_embeddings(token_lists):
    if not token_lists: return None
    print("Melatih model Word Embeddings (Word2Vec)...")
    try:
        w2v_model = Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=2, workers=os.cpu_count() or 1)
        if not hasattr(w2v_model, 'wv') or not w2v_model.wv.key_to_index:
             print("Warning: Model Word2Vec dilatih tetapi vocabulary kosong.")
             return None
        print(f"Model Word2Vec dilatih dengan {len(w2v_model.wv.key_to_index)} kata unik.")
        return w2v_model
    except Exception as e: print(f"Error melatih Word2Vec: {e}"); return None

def generate_general_wordcloud(all_text):
    img_path = "static/images/wordcloud_general.png"
    print("Membuat Word Cloud Umum...")
    try:
        wordcloud = WordCloud(width=1000, height=600, background_color='#0a0a0a',
                              colormap='cool', max_words=150, prefer_horizontal=0.9,
                              stopwords=stop_words
                              ).generate(all_text if all_text and all_text.strip() else "Tidak Ada Data")
        wordcloud.to_file(img_path)
        print(f"Word Cloud Umum disimpan di {img_path}")
    except Exception as e: print(f"Error membuat word cloud umum: {e}")

def train_lda_model(bow_matrix, n_topics=5):
    if bow_matrix is None or bow_matrix.shape[0] < n_topics:
        print("Data tidak cukup untuk melatih LDA, pelatihan dilewati.")
        return None
    print(f"Melatih model Topic Modeling (LDA) dengan {n_topics} topik...")
    try:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
        lda.fit(bow_matrix)
        print("Model LDA selesai dilatih.")
        return lda
    except Exception as e: print(f"Error melatih LDA: {e}"); return None

def generate_embeddings(texts):
    if SEMANTIC_MODEL is None or not texts:
        print("Model Semantic atau teks tidak tersedia, pembuatan embeddings dilewati.")
        return None
    print(f"Menghasilkan embeddings untuk {len(texts)} teks...")
    try:
        embeddings = SEMANTIC_MODEL.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        print("Embeddings selesai dibuat.")
        return embeddings.cpu().numpy()
    except Exception as e: print(f"Error saat membuat embeddings: {e}"); return None

def train_bertopic_model(documents_list, embedding_model):
    if not documents_list or len(documents_list) < 20:
        print("Data terlalu sedikit untuk BERTopic (<20), pelatihan dilewati.")
        return None, None, None
    if embedding_model is None:
        print("Model embedding tidak tersedia, pelatihan BERTopic dilewati.")
        return None, None, None
    print("Memulai pelatihan model BERTopic...")
    try:
        bertopic_model = BERTopic(embedding_model=embedding_model, language="multilingual",
                                  calculate_probabilities=True, verbose=True, min_topic_size=5)
        topics, probabilities = bertopic_model.fit_transform(documents_list)
        num_topics_found = len(bertopic_model.get_topic_info())
        print(f"Model BERTopic selesai dilatih. Ditemukan {num_topics_found} topik (termasuk outlier).")
        bertopic_model.doc_topics_ = topics; bertopic_model.doc_probs_ = probabilities
        return bertopic_model, topics, probabilities
    except Exception as e: print(f"Error melatih BERTopic: {e}"); return None, None, None

# --- Fungsi Pelatihan & Evaluasi Sentimen dari CSV ---
def train_and_evaluate_sentiment_model_from_csv(csv_path):
    """
    Melatih model sentimen dari CSV, mencetak laporan akurasi, menyimpan metrik,
    dan mengembalikan model serta vectorizer.
    """
    global SENTIMENT_EVAL_ACCURACY, SENTIMENT_EVAL_REPORT, SENTIMENT_EVAL_CM_IMG

    print(f"\n--- Memulai Pelatihan & Evaluasi Model Sentimen dari {csv_path} ---")
    sentiment_model = None
    sentiment_vectorizer = None
    SENTIMENT_EVAL_ACCURACY = None; SENTIMENT_EVAL_REPORT = None; SENTIMENT_EVAL_CM_IMG = None # Reset
    try:
        # 1. Muat CSV dengan delimiter koma (,)
        print(f"DEBUG: Mencoba memuat CSV: {csv_path} dengan delimiter ','...")
        df_labeled = pd.read_csv(csv_path, encoding='utf-8-sig', delimiter=',')
        df_labeled.columns = df_labeled.columns.str.strip() # Hapus spasi ekstra di header
        print(f"DEBUG: Berhasil memuat CSV. Kolom ditemukan: {df_labeled.columns.tolist()}")

        # 2. Validasi Kolom
        text_col = 'Komentar' # Pastikan sesuai nama kolom di CSV Anda
        label_col = 'Sentimen' # Pastikan sesuai nama kolom di CSV Anda
        if text_col not in df_labeled.columns or label_col not in df_labeled.columns:
            print(f"FATAL ERROR: Kolom '{text_col}' atau '{label_col}' tidak ditemukan di CSV.")
            return None, None
        print(f"DEBUG: Kolom '{text_col}' dan '{label_col}' ditemukan.")

        # 3. Cleaning Data Awal
        initial_rows = len(df_labeled)
        df_labeled = df_labeled.dropna(subset=[text_col, label_col])
        valid_labels = ['Positif', 'Negatif', 'Netral'] # Urutan penting untuk CM nanti
        df_labeled = df_labeled[df_labeled[label_col].isin(valid_labels)]
        cleaned_rows = len(df_labeled)
        print(f"DEBUG: Data setelah cleaning awal: {cleaned_rows} baris valid (dari {initial_rows}).")
        if cleaned_rows < 50: print(f"FATAL ERROR: Data berlabel valid < 50 ({cleaned_rows} baris)."); return None, None

        # 4. Preprocessing Teks
        print("DEBUG: Melakukan preprocessing data berlabel...")
        df_labeled['Processed_Text'] = df_labeled[text_col].apply(lambda x: preprocess_text_for_tfidf(x) if pd.notna(x) else "")
        df_labeled = df_labeled[df_labeled['Processed_Text'].str.strip().astype(bool)]
        processed_rows = len(df_labeled)
        print(f"DEBUG: Data setelah preprocessing: {processed_rows} baris valid.")
        if processed_rows < 50: print(f"FATAL ERROR: Data setelah preprocessing < 50 ({processed_rows} baris)."); return None, None

        X = df_labeled['Processed_Text']
        y = df_labeled[label_col]

        # 5. Validasi Jumlah Kelas
        unique_classes = sorted(y.unique(), key=lambda x: valid_labels.index(x) if x in valid_labels else 99) # Urutkan Neg, Net, Pos
        print(f"DEBUG: Kelas sentimen yang ditemukan setelah preprocessing: {unique_classes}")
        if len(unique_classes) < 2: print(f"FATAL ERROR: Hanya ditemukan {len(unique_classes)} kelas sentimen."); return None, None
        elif len(unique_classes) < 3: print(f"PERINGATAN: Hanya ditemukan {len(unique_classes)} kelas sentimen.")

        # 6. Bagi Data Latih & Uji
        print("DEBUG: Membagi data latih/uji...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"DEBUG: Data dibagi: {len(X_train)} latih, {len(X_test)} uji.")

        # 7. Vektorisasi TF-IDF
        print("DEBUG: Melakukan vektorisasi TF-IDF...")
        sentiment_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_tfidf = sentiment_vectorizer.fit_transform(X_train)
        X_test_tfidf = sentiment_vectorizer.transform(X_test)
        print("DEBUG: Vektorisasi TF-IDF (untuk sentimen) selesai.")

        # 8. Latih Model Logistic Regression
        print("DEBUG: Melatih model Logistic Regression...")
        sentiment_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=1.0)
        sentiment_model.fit(X_train_tfidf, y_train)
        print("DEBUG: Model sentimen (Logistic Regression) berhasil dilatih.")

        # --- Uji Akurasi ---
        print("\n--- LAPORAN EVALUASI MODEL SENTIMEN (DATA UJI 20%) ---")
        y_pred = sentiment_model.predict(X_test_tfidf)

        # Simpan Akurasi
        accuracy = accuracy_score(y_test, y_pred)
        SENTIMENT_EVAL_ACCURACY = round(accuracy * 100, 2)
        print(f"Akurasi (Accuracy): {accuracy:.4f}")

        # Simpan Laporan Klasifikasi (sebagai dictionary)
        report_dict = classification_report(y_test, y_pred, labels=unique_classes, zero_division=0, output_dict=True)
        SENTIMENT_EVAL_REPORT = report_dict
        print("\nLaporan Klasifikasi (Precision, Recall, F1-Score):")
        print(classification_report(y_test, y_pred, labels=unique_classes, zero_division=0))

        # Simpan Confusion Matrix (sebagai gambar base64)
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        print(f"Label (Sumbu): {unique_classes}")
        print(cm)
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            plt.title('Confusion Matrix (Data Uji)', color='cyan')
            plt.xticks(color='cyan'); plt.yticks(color='cyan')
            ax.xaxis.label.set_color('cyan'); ax.yaxis.label.set_color('cyan')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            SENTIMENT_EVAL_CM_IMG = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
            print("DEBUG: Gambar Confusion Matrix berhasil dibuat.")
        except Exception as plot_err:
            print(f"WARNING: Gagal membuat plot Confusion Matrix: {plot_err}")
            SENTIMENT_EVAL_CM_IMG = None

        print("---------------------------------------------------\n")
        print("DEBUG: Pelatihan dan evaluasi sentimen SUKSES.")
        return sentiment_model, sentiment_vectorizer

    except FileNotFoundError:
        print(f"FATAL ERROR: File data sentimen berlabel '{csv_path}' tidak ditemukan.")
        print("Harap letakkan file di folder yang sama dengan app.py.")
        return None, None
    except Exception as e:
        print(f"FATAL ERROR saat melatih model sentimen dari CSV: {e}")
        traceback.print_exc()
        return None, None

# --- Fungsi Helper Visualisasi ---
def get_topic_distribution_for_doc(doc_id):
    if LDA_MODEL is None or BOW_MATRIX is None or not DOC_NAMES_LIST: return {"error": "Model LDA atau BoW tidak tersedia."}
    try:
        doc_index = DOC_NAMES_LIST.index(doc_id)
        if doc_index >= BOW_MATRIX.shape[0]: return {"error": "Indeks dokumen di luar batas."}
        doc_bow = BOW_MATRIX[doc_index]; topic_distribution = LDA_MODEL.transform(doc_bow)[0]
        result = {f"Topik {i + 1}": round(prob * 100, 2) for i, prob in enumerate(topic_distribution)}
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    except ValueError: return {"error": "Dokumen ID tidak ditemukan."}
    except Exception as e: print(f"Error distribusi LDA: {e}"); return {"error": f"Error internal: {e}"}

def get_sparse_matrix_viz(vectorizer, matrix, doc_names, num_docs=5):
    if vectorizer is None or matrix is None: return []
    viz_data = []; feature_names = vectorizer.get_feature_names_out()
    docs_to_process = min(num_docs, matrix.shape[0], len(doc_names))
    for i in range(docs_to_process):
        doc_name = doc_names[i]; doc_vector = matrix.getrow(i)
        for col_idx, score in zip(doc_vector.indices, doc_vector.data):
            if col_idx < len(feature_names):
                term = feature_names[col_idx]
                viz_data.append({"doc": doc_name, "term": term, "score": round(score, 4)})
    return viz_data

def get_word_embeddings_viz(w2v_model, query_words):
    if w2v_model is None or not hasattr(w2v_model, 'wv') or not w2v_model.wv.key_to_index: return {"error": "Model Word2Vec tidak tersedia."}
    viz_data = {}
    for word in query_words:
        word = word.lower().strip(); processed_word = stemmer.stem(word)
        if not word: continue
        if processed_word in w2v_model.wv:
            try:
                similar_words = w2v_model.wv.most_similar(processed_word, topn=5)
                display_key = f"{word} (stem: {processed_word})" if word != processed_word else word
                viz_data[display_key] = ", ".join([f"{w} ({s:.2f})" for w, s in similar_words])
            except Exception as e: viz_data[word] = f"Error: {e}"
        else: viz_data[word] = f"Kata '{processed_word}' tidak ditemukan."
    return viz_data

def get_lda_topics_viz(lda_model, vectorizer, n_words=10):
    if lda_model is None or vectorizer is None or not hasattr(lda_model, 'components_'): return {}
    print("Mengekstrak visualisasi LDA..."); viz_data = {}
    try:
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            topic_name = f"Topik {topic_idx + 1}"
            top_words_indices = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices if i < len(feature_names)]
            viz_data[topic_name] = ", ".join(top_words)
    except Exception as e: print(f"Error mengekstrak topik LDA: {e}")
    return viz_data

def get_bertopic_topics_viz(bertopic_model):
    if bertopic_model is None: return {}
    print("Mengekstrak visualisasi BERTopic..."); viz_data = {}
    try:
        topic_info = bertopic_model.get_topic_info()
        for _, row in topic_info[topic_info.Topic != -1].iterrows():
            topic_id = row['Topic']; topic_name_parts = row['Name'].split('_')
            display_name = "_".join(topic_name_parts[1:min(4, len(topic_name_parts))])
            topic_name = f"Topik {topic_id} ({display_name})"
            keywords = ", ".join(row['Representation']) if isinstance(row['Representation'], list) else str(row['Representation'])
            viz_data[topic_name] = keywords
    except Exception as e: print(f"Error mengekstrak topik BERTopic: {e}")
    return viz_data

# =========================================================================
# === PERSIAPAN DATA GLOBAL ===
# =========================================================================
LABELED_DATASET_FILE = 'Data Komen Apps InfoBMKG - dataset_bmkg_reviews.csv' # Gunakan file CSV yang sudah diperbaiki
DEFAULT_DATASET_FILE = 'dataset_bmkg_reviews.txt' # File .txt untuk dimuat saat start (opsional)

DOCUMENTS, DOCS_TOKENS, DOCS_STRINGS = {}, [], []; ALL_PROCESSED_TEXT = ""; DOC_NAMES_LIST = []
INVERTED_INDEX = {}; BOW_VECTORIZER, BOW_MATRIX = None, None; TFIDF_VECTORIZER, TFIDF_MATRIX = None, None
W2V_MODEL = None; LDA_MODEL = None; DOC_EMBEDDINGS = None
BERTOPIC_MODEL, BERTOPIC_TOPICS, BERTOPIC_PROBS = None, None, None
SENTIMENT_MODEL, SENTIMENT_TFIDF_VECTORIZER = None, None
SENTIMENT_PREDICTIONS, SENTIMENT_DISTRIBUTION = [], {}
SENTIMENT_EVAL_ACCURACY = None; SENTIMENT_EVAL_REPORT = None; SENTIMENT_EVAL_CM_IMG = None

# === VARIABEL BARU UNTUK VISUALISASI ===
SENTIMENT_DISTRIBUTION_COUNTS = {} # Untuk menyimpan jumlah (count)
SENTIMENT_SILHOUETTE_SCORE = None # Untuk skor silhouette

# =========================================================================
# === DEFINISI FUNGSI load_and_train_all ===
# =========================================================================
def load_and_train_all(data_source, source_type='path', called_from_startup=False):
    """Memuat data .txt, melatih model non-sentimen, dan memprediksi sentimen."""
    global DOCUMENTS, DOCS_TOKENS, DOCS_STRINGS, ALL_PROCESSED_TEXT, DOC_NAMES_LIST
    global INVERTED_INDEX, BOW_VECTORIZER, BOW_MATRIX, TFIDF_VECTORIZER, TFIDF_MATRIX
    global W2V_MODEL, LDA_MODEL, DOC_EMBEDDINGS, BERTOPIC_MODEL, BERTOPIC_TOPICS, BERTOPIC_PROBS
    global SENTIMENT_PREDICTIONS, SENTIMENT_DISTRIBUTION, SENTIMENT_DISTRIBUTION_COUNTS, SENTIMENT_SILHOUETTE_SCORE # Tambahkan var baru

    # Reset variabel sebelum memuat data baru
    DOCUMENTS, DOCS_TOKENS, DOCS_STRINGS = {}, [], []; ALL_PROCESSED_TEXT = ""; DOC_NAMES_LIST = []
    INVERTED_INDEX = {}; BOW_VECTORIZER, BOW_MATRIX = None, None; TFIDF_VECTORIZER, TFIDF_MATRIX = None, None
    W2V_MODEL = None; LDA_MODEL = None; DOC_EMBEDDINGS = None
    BERTOPIC_MODEL, BERTOPIC_TOPICS, BERTOPIC_PROBS = None, None, None
    SENTIMENT_PREDICTIONS = []; SENTIMENT_DISTRIBUTION = {}; SENTIMENT_DISTRIBUTION_COUNTS = {}; SENTIMENT_SILHOUETTE_SCORE = None # Reset var baru
    generate_general_wordcloud("")

    # Muat dan proses data .txt
    if source_type == 'path': DOCUMENTS, DOCS_TOKENS, DOCS_STRINGS = process_data_from_path(data_source)
    elif source_type == 'stream': DOCUMENTS, DOCS_TOKENS, DOCS_STRINGS = process_data_from_stream(data_source)
    else: print(f"Error: source_type tidak dikenal: {source_type}"); return False

    # Cek apakah data .txt berhasil dimuat
    if DOCUMENTS is None or not DOCUMENTS:
        print("\n--- Gagal memuat data .txt atau data kosong. Model tidak dilatih/diprediksi. ---")
        if not called_from_startup: flash("Gagal memuat atau memproses data .txt.", 'danger')
        return False

    ALL_PROCESSED_TEXT = " ".join(DOCS_STRINGS)
    DOC_NAMES_LIST = sorted(DOCUMENTS.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    original_texts = [DOCUMENTS[doc_name] for doc_name in DOC_NAMES_LIST]

    print("\n--- Memulai Pelatihan Model (Non-Sentimen) ---")
    INVERTED_INDEX = build_inverted_index(DOCS_TOKENS)
    BOW_VECTORIZER, BOW_MATRIX = calculate_bow(DOCS_STRINGS)
    TFIDF_VECTORIZER, TFIDF_MATRIX = calculate_tfidf(DOCS_STRINGS)
    W2V_MODEL = train_word_embeddings(DOCS_TOKENS)
    generate_general_wordcloud(ALL_PROCESSED_TEXT)
    LDA_MODEL = train_lda_model(BOW_MATRIX, n_topics=5)
    DOC_EMBEDDINGS = generate_embeddings(original_texts)
    BERTOPIC_MODEL, BERTOPIC_TOPICS, BERTOPIC_PROBS = train_bertopic_model(original_texts, SEMANTIC_MODEL)

    print("\n--- Memprediksi Sentimen ---")
    if SENTIMENT_MODEL is None or SENTIMENT_TFIDF_VECTORIZER is None:
        print("WARNING: Model sentimen global tidak tersedia. Prediksi sentimen dilewati.")
        if not called_from_startup: flash("Model sentimen tidak dimuat.", 'warning')
    else:
        try:
            processed_sentiment_texts = [preprocess_text_for_tfidf(text) for text in original_texts]
            valid_indices = [i for i, text in enumerate(processed_sentiment_texts) if text.strip()]
            
            sentiment_tfidf_matrix = None # Definisikan di luar
            
            if not valid_indices:
                print("WARNING: Tidak ada teks valid setelah preprocessing untuk prediksi sentimen.")
                SENTIMENT_PREDICTIONS = ['Netral'] * len(original_texts)
            else:
                valid_processed_texts = [processed_sentiment_texts[i] for i in valid_indices]
                sentiment_tfidf_matrix = SENTIMENT_TFIDF_VECTORIZER.transform(valid_processed_texts) # Simpan matriks
                predictions_valid = SENTIMENT_MODEL.predict(sentiment_tfidf_matrix).tolist()
                SENTIMENT_PREDICTIONS = ['Netral'] * len(original_texts)
                for idx, pred in zip(valid_indices, predictions_valid):
                    SENTIMENT_PREDICTIONS[idx] = pred

            total_docs = len(SENTIMENT_PREDICTIONS)
            if total_docs > 0:
                pos_count = SENTIMENT_PREDICTIONS.count('Positif')
                neg_count = SENTIMENT_PREDICTIONS.count('Negatif')
                neu_count = SENTIMENT_PREDICTIONS.count('Netral')
                
                SENTIMENT_DISTRIBUTION_COUNTS = {
                    'Positif': pos_count,
                    'Negatif': neg_count,
                    'Netral': neu_count,
                    'Total': total_docs
                }
                SENTIMENT_DISTRIBUTION = {
                    'Positif': round((pos_count / total_docs) * 100, 1),
                    'Negatif': round((neg_count / total_docs) * 100, 1),
                    'Netral': round((neu_count / total_docs) * 100, 1)
                }
                print(f"Prediksi sentimen selesai. Distribusi (Counts): {SENTIMENT_DISTRIBUTION_COUNTS}")
            
            else: 
                SENTIMENT_DISTRIBUTION = {}
                SENTIMENT_DISTRIBUTION_COUNTS = {}

            # --- PERHITUNGAN SILHOUETTE SCORE (BARU) ---
            print("Menghitung Silhouette Score (K-Means)...")
            # Butuh setidaknya 3 sampel untuk K-Means (k=2) dan silhouette
            if sentiment_tfidf_matrix is not None and len(valid_indices) > 2:
                try:
                    # Tentukan jumlah cluster, min 2, maks 3 (atau jumlah sampel - 1)
                    num_samples = len(valid_indices)
                    n_clusters_to_try = min(3, num_samples - 1)

                    if n_clusters_to_try >= 2:
                        kmeans = KMeans(n_clusters=n_clusters_to_try, random_state=42, n_init='auto')
                        cluster_labels = kmeans.fit_predict(sentiment_tfidf_matrix)
                        
                        # Pastikan K-Means menghasilkan lebih dari 1 cluster
                        if len(np.unique(cluster_labels)) > 1:
                            score = silhouette_score(sentiment_tfidf_matrix, cluster_labels)
                            SENTIMENT_SILHOUETTE_SCORE = round(score, 4)
                            print(f"Silhouette Score (K={n_clusters_to_try}): {SENTIMENT_SILHOUETTE_SCORE}")
                        else:
                            print("K-Means hanya menemukan 1 cluster, silhouette score tidak dihitung.")
                            SENTIMENT_SILHOUETTE_SCORE = None # Atau 0
                    else:
                        print("Tidak cukup cluster (perlu min 2) untuk silhouette score.")
                        SENTIMENT_SILHOUETTE_SCORE = None
                except Exception as e:
                    print(f"Error saat menghitung Silhouette Score: {e}")
                    SENTIMENT_SILHOUETTE_SCORE = None
            else:
                print("Tidak cukup data (perlu > 2) untuk menghitung Silhouette Score.")
                SENTIMENT_SILHOUETTE_SCORE = None
            # --- AKHIR PERHITUNGAN SILHOUETTE ---

        except Exception as e:
            print(f"Error saat prediksi sentimen: {e}")
            if not called_from_startup: flash(f"Error saat prediksi sentimen: {e}", 'danger')
            SENTIMENT_PREDICTIONS = []; SENTIMENT_DISTRIBUTION = {}; SENTIMENT_DISTRIBUTION_COUNTS = {}; SENTIMENT_SILHOUETTE_SCORE = None

    print("\n--- Semua proses untuk data baru selesai! ---")
    return True

# =========================================================================
# === EKSEKUSI SAAT STARTUP ===
# =========================================================================
# Latih & Evaluasi Model Sentimen DULU
SENTIMENT_MODEL, SENTIMENT_TFIDF_VECTORIZER = train_and_evaluate_sentiment_model_from_csv(LABELED_DATASET_FILE)
if SENTIMENT_MODEL is None:
    print("\n*******************************************************")
    print("PERINGATAN: Model Sentimen GAGAL dimuat/dilatih.")
    print(f"Pastikan file '{LABELED_DATASET_FILE}' ada, format benar (delimiter ','),")
    print("dan memiliki kolom 'Komentar' serta 'Sentimen'.")
    print("Fitur Analisis Sentimen akan dinonaktifkan.")
    print("*******************************************************\n")

# Muat Dataset Default dan Latih Model Lainnya
print(f"\nMencoba memuat dataset default: {DEFAULT_DATASET_FILE}")
load_and_train_all(DEFAULT_DATASET_FILE, source_type='path', called_from_startup=True)
# =========================================================================


# --- Rute Web (Flask) ---
@app.route('/')
def home():
    messages = get_flashed_messages(with_categories=True)
    ii_viz = dict(list(INVERTED_INDEX.items())[:100])
    bow_viz = get_sparse_matrix_viz(BOW_VECTORIZER, BOW_MATRIX, DOC_NAMES_LIST, num_docs=5)
    tfidf_viz = get_sparse_matrix_viz(TFIDF_VECTORIZER, TFIDF_MATRIX, DOC_NAMES_LIST, num_docs=5)
    default_w2v_query = ['lokasi', 'gps', 'error'] if DEFAULT_DATASET_FILE and 'bmkg' in DEFAULT_DATASET_FILE.lower() else ['aqua', 'izin']
    w2v_viz = get_word_embeddings_viz(W2V_MODEL, default_w2v_query)
    lda_viz = get_lda_topics_viz(LDA_MODEL, BOW_VECTORIZER, n_words=10)
    bertopic_viz = get_bertopic_topics_viz(BERTOPIC_MODEL)
    sentiment_samples = []
    if DOCUMENTS and SENTIMENT_PREDICTIONS and len(DOC_NAMES_LIST) == len(SENTIMENT_PREDICTIONS):
        num_samples = min(10, len(DOC_NAMES_LIST))
        if DOC_NAMES_LIST:
            try:
                sampled_indices = random.sample(range(len(DOC_NAMES_LIST)), num_samples)
                for i in sampled_indices:
                     if i < len(DOC_NAMES_LIST) and i < len(SENTIMENT_PREDICTIONS):
                         doc_name = DOC_NAMES_LIST[i]; doc_text = DOCUMENTS[doc_name]; prediction = SENTIMENT_PREDICTIONS[i]
                         sentiment_samples.append({'text': doc_text[:150] + ('...' if len(doc_text) > 150 else ''), 'sentiment': prediction})
            except ValueError as e: print(f"Error sampling sentimen: {e}")
    cache_buster = int(time.time())
    # Kirim semua data, termasuk evaluasi
    return render_template('index.html', messages=messages, inverted_index=ii_viz, documents=DOCUMENTS,
                           doc_names_list=DOC_NAMES_LIST, bow_data=bow_viz, tfidf_data=tfidf_viz,
                           w2v_data=w2v_viz, lda_data=lda_viz, bertopic_data=bertopic_viz,
                           sentiment_distribution=SENTIMENT_DISTRIBUTION, sentiment_samples=sentiment_samples,
                           sentiment_accuracy=SENTIMENT_EVAL_ACCURACY, sentiment_report=SENTIMENT_EVAL_REPORT,
                           sentiment_cm_img=SENTIMENT_EVAL_CM_IMG, cache_buster=cache_buster)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dataset_file' not in request.files: flash('Tidak ada file terdeteksi.', 'danger'); return redirect(url_for('home'))
    file = request.files['dataset_file']
    if file.filename == '': flash('Tidak ada file dipilih.', 'warning'); return redirect(url_for('home'))
    if file and file.filename.lower().endswith('.txt'):
        try:
            success = load_and_train_all(file, source_type='stream', called_from_startup=False)
            if success: flash(f"File '{file.filename}' berhasil diproses!", 'success')
        except Exception as e: print(f"Error besar saat upload: {e}"); flash(f"Error pemrosesan unggahan: {e}", 'danger')
        return redirect(url_for('home'))
    else: flash('Format file tidak valid (.txt).', 'danger'); return redirect(url_for('home'))

@app.route('/get_similar_words', methods=['POST'])
def get_similar_words_route():
    if W2V_MODEL is None: return jsonify({"error": "Model Word2Vec tidak dimuat."}), 500
    query_word = request.form.get('word', ''); viz_data = get_word_embeddings_viz(W2V_MODEL, [query_word]); return jsonify(viz_data)

@app.route('/semantic_search', methods=['POST'])
def semantic_search_route():
    query = request.form.get('query', '').strip()
    if not query: return jsonify({"results": []})
    if SEMANTIC_MODEL is None: return jsonify({"error": "Model semantic tidak dimuat."}), 500
    if DOC_EMBEDDINGS is None or not DOCUMENTS: return jsonify({"error": "Embeddings dokumen belum dibuat."}), 500
    try:
        print(f"Mencari query semantik: '{query}'")
        query_embedding = SEMANTIC_MODEL.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        similarities = cosine_similarity(query_embedding, DOC_EMBEDDINGS)[0]
        top_n = 5; top_indices = np.argsort(similarities)[-top_n:][::-1]
        results = []
        for i in top_indices:
             if i < len(DOC_NAMES_LIST) and similarities[i] > 0.1:
                 doc_name = DOC_NAMES_LIST[i]; score = float(similarities[i])
                 text_snippet = DOCUMENTS[doc_name][:150] + ('...' if len(DOCUMENTS[doc_name]) > 150 else '')
                 results.append({"doc_name": doc_name, "score": round(score, 4), "text": text_snippet})
        return jsonify({"results": results})
    except Exception as e: print(f"Error semantic search: {e}"); return jsonify({"error": f"Error pencarian: {e}"}), 500

@app.route('/get_topic_distribution', methods=['POST'])
def get_topic_distribution_route():
    doc_id = request.form.get('doc_id', ''); distribution = get_topic_distribution_for_doc(doc_id); return jsonify(distribution)

@app.route('/get_bertopic_distribution', methods=['POST'])
def get_bertopic_distribution_route():
    doc_id = request.form.get('doc_id', '')
    if not doc_id: return jsonify({"error": "Parameter 'doc_id' tidak ada."}), 400
    if BERTOPIC_MODEL is None or BERTOPIC_PROBS is None or not DOC_NAMES_LIST: return jsonify({"error": "Model BERTopic tidak tersedia."}), 500
    try:
        doc_index = DOC_NAMES_LIST.index(doc_id)
        if doc_index >= len(BERTOPIC_PROBS): return jsonify({"error": "Indeks dokumen di luar batas."}), 500
        doc_probs_row = BERTOPIC_PROBS[doc_index]
        topic_info_df = BERTOPIC_MODEL.get_topic_info()
        topic_info_df = topic_info_df[topic_info_df.Topic != -1].reset_index(drop=True)
        result = {}
        for idx, row in topic_info_df.iterrows():
            if idx < len(doc_probs_row):
                topic_name_parts = row['Name'].split('_'); display_name = "_".join(topic_name_parts[1:min(4, len(topic_name_parts))])
                topic_name = f"Topik {row.Topic} ({display_name})"; probability = doc_probs_row[idx]
                if probability > 0.01: result[topic_name] = round(float(probability) * 100, 2)
        return jsonify(dict(sorted(result.items(), key=lambda item: item[1], reverse=True)))
    except ValueError: return jsonify({"error": "Dokumen ID tidak ditemukan."}), 404
    except Exception as e: print(f"Error distribusi BERTopic: {e}"); return jsonify({"error": f"Error internal: {e}"}), 500

# === RUTE BARU UNTUK VISUALISASI SENTIMEN ===
@app.route('/get_sentiment_visualization_data')
def get_sentiment_visualization_data():
    """Mengembalikan data distribusi sentimen (COUNTS) untuk chart baru."""
    global SENTIMENT_DISTRIBUTION_COUNTS # Gunakan var baru
    if not SENTIMENT_DISTRIBUTION_COUNTS or SENTIMENT_DISTRIBUTION_COUNTS.get('Total', 0) == 0:
        return jsonify({"error": "Data distribusi sentimen tidak ditemukan. Harap unggah data."}), 400
    return jsonify(SENTIMENT_DISTRIBUTION_COUNTS)

# === RUTE BARU UNTUK SILHOUETTE SCORE ===
@app.route('/get_silhouette_score')
def get_silhouette_score():
    """Mengembalikan Silhouette Score yang sudah dihitung."""
    global SENTIMENT_SILHOUETTE_SCORE
    if SENTIMENT_SILHOUETTE_SCORE is None:
        return jsonify({"error": "Skor Silhouette tidak tersedia. Data mungkin terlalu sedikit atau K-Means gagal."}), 400
    return jsonify({"score": SENTIMENT_SILHOUETTE_SCORE})
# === AKHIR RUTE BARU ===


# --- Menjalankan Aplikasi ---
if __name__ == '__main__':
    if not os.path.exists('static/images'):
        try: os.makedirs('static/images')
        except OSError as e: print(f"Error membuat direktori static/images: {e}")
    print("\n--- Menjalankan Server Flask ---")
    app.run(debug=True, host='0.0.0.0', port=5000)