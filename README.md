# BankClust: Sistem Clustering dan Klasifikasi Transaksi Bank

## 📊 Gambaran Umum Project

BankClust adalah platform analitik canggih yang memanfaatkan teknik machine learning untuk melakukan segmentasi nasabah dan analisis pola transaksi bagi institusi perbankan. Sistem ini menggunakan unsupervised learning (K-Means clustering) yang dikombinasikan dengan klasifikasi supervised (Decision Trees) untuk mengidentifikasi segmen nasabah yang berbeda dan memprediksi perilaku transaksi.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Clustering%20%26%20Classification-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

## 🎯 Tujuan dan Manfaat

### **Tujuan Utama**
- **Segmentasi Nasabah**: Mengelompokkan nasabah berdasarkan pola transaksi dan karakteristik demografi
- **Analisis Pola Transaksi**: Mengidentifikasi pola transaksi yang mencurigakan atau tidak biasa
- **Optimasi Layanan**: Memberikan insights untuk pengembangan produk dan layanan yang lebih personal

### **Manfaat Bisnis**
- 🎯 Marketing yang lebih terarah berdasarkan segmentasi nasabah
- ⚠️ Deteksi dini transaksi mencurigakan
- 📈 Optimasi penawaran produk perbankan
- 🔍 Pemahaman mendalam tentang perilaku nasabah

## 🏗️ Arsitektur Sistem

```
Data Pipeline:
Data Mentah → Pembersihan Data → Preprocessing → Clustering → Classification → Visualisasi

Komponen Utama:
1. Data Collection & Cleaning
2. Feature Engineering
3. K-Means Clustering
4. Decision Tree Classification
5. Result Visualization
```

## 📁 Struktur Dataset

### **Fitur yang Dianalisis:**
- **Data Demografi**: CustomerAge, Location
- **Data Transaksi**: TransactionAmount, TransactionDuration, TransactionType
- **Data Akun**: AccountBalance, LoginAttempts
- **Data Waktu**: TransactionDate, TransactionTime (diekstrak menjadi hour, dayofweek, is_weekend)

### **Preprocessing Features:**
- Label Encoding untuk data kategorikal
- StandardScaler untuk normalisasi numerik
- PCA untuk reduksi dimensi
- Feature Selection menggunakan VarianceThreshold dan SelectKBest

## 🔧 Teknologi yang Digunakan

### **Machine Learning Algorithms**
- **Clustering**: K-Means dengan optimasi Elbow Method
- **Classification**: Decision Tree Classifier
- **Feature Selection**: VarianceThreshold + SelectKBest
- **Dimensionality Reduction**: PCA (Principal Component Analysis)

### **Python Libraries**
```python
# Data Processing
pandas, numpy, scikit-learn

# Visualization
matplotlib, seaborn, plotly

# Machine Learning
scikit-learn, yellowbrick

# Utilities
joblib, warnings
```

## 📊 Metodologi

### **1. Eksplorasi Data Awal**
- Analisis statistik deskriptif
- Visualisasi distribusi data
- Matriks korelasi antar fitur
- Deteksi missing values dan outliers

### **2. Preprocessing Data**
```python
# Handling missing values
imputer = SimpleImputer(strategy='mean')  # numerik
mode imputation  # kategorikal

# Feature engineering
datetime feature extraction → hour, dayofweek, is_weekend
Binning untuk continuous variables
Label Encoding untuk categorical variables
```

### **3. Penentuan Jumlah Cluster**
- **Elbow Method** dengan KElbowVisualizer
- **Silhouette Score** analysis
- **Davies-Bouldin Index** evaluation
- Optimal k determination based on multiple metrics

### **4. Model Building**
```python
# Clustering Model
kmeans = KMeans(n_clusters=optimal_k, random_state=42)

# Classification Model
clf = DecisionTreeClassifier(random_state=42)
```

### **5. Evaluasi Model**
- **Clustering**: Silhouette Score, Davies-Bouldin Index
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Visual Validation**: PCA 2D visualization

## 🚀 Cara Menjalankan

### **Instalasi Dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn yellowbrick joblib
```

### **Menjalankan Analisis**
```python
# 1. Load data
df = pd.read_csv("bank_transactions_data_edited.csv")

# 2. Run complete pipeline
python bankclust_analysis.py

# 3. View results
# - Cluster visualization
# - Classification report
# - Feature importance
```

### **Output yang Dihasilkan**
- `model_clustering.h5` - Model K-Means yang sudah trained
- `model_klasifikasi.h5` - Model Decision Tree yang sudah trained
- `data_clustering_with_target.csv` - Data dengan label cluster
- Visualisasi PCA dan cluster analysis

## 📈 Hasil dan Interpretasi

### **Metrik Evaluasi**
```python
# Contoh output
Silhouette Score: 0.75 (Excellent separation)
Davies-Bouldin Index: 0.45 (Good clustering)
Classification Accuracy: 92.5% (High predictive power)
```

### **Interpretasi Cluster**
Setiap cluster merepresentasikan segmen nasabah yang berbeda:
- **Cluster 0**: Nasabah dengan transaksi rutin kecil
- **Cluster 1**: Nasabah premium dengan transaksi besar
- **Cluster 2**: Nasabah dengan aktivitas mencurigakan
- **Cluster 3**: Nasabah seasonal dengan pola tertentu

## 🔍 Use Cases

### **1. Segmentasi Marketing**
```python
# Target marketing berdasarkan cluster
cluster_1_premium → Offer wealth management products
cluster_0_regular → Offer savings and basic banking products
```

### **2. Fraud Detection**
```python
# Identifikasi pola mencurigakan
anomaly_cluster = df[df['cluster'] == 2]
high_risk_transactions = anomaly_cluster[anomaly_cluster['TransactionAmount'] > threshold]
```

### **3. Customer Behavior Analysis**
```python
# Analisis pola transaksi
weekend_transactions = df[df['is_weekend'] == 1]
peak_hours_analysis = df.groupby('transaction_hour')['TransactionAmount'].mean()
```

## 📝 Customization

### **Menyesuaikan Parameter**
```python
# Adjust clustering parameters
kmeans = KMeans(
    n_clusters=5,           # Ubah jumlah cluster
    random_state=42,
    n_init='auto'
)

# Adjust feature selection
selector = SelectKBest(score_func=f_classif, k=15)  # Ubah jumlah fitur
```

### **Menambah Fitur Baru**
```python
# Tambah feature engineering
df['amount_to_balance_ratio'] = df['TransactionAmount'] / df['AccountBalance']
df['transaction_frequency'] = df.groupby('CustomerID')['TransactionID'].transform('count')
```

## 🤝 Kontribusi

Kami menyambut kontribusi untuk pengembangan BankClust! Beberapa area yang dapat dikembangkan:

1. **Algoritma tambahan** (DBSCAN, Hierarchical Clustering)
2. **Deep Learning integration**
3. **Real-time streaming analysis**
4. **Dashboard visualization**

## 📄 Lisensi

Project ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail lengkap.

## 👥 Tim Pengembang

**BankClust Development Team**
- Data Scientist & ML Engineer
- Business Analyst
- Banking Domain Expert

---

## 📞 Kontak dan Support

Untuk pertanyaan, bug reports, atau kolaborasi, silakan buat issue di repository atau hubungi tim pengembang.

**Status Project**: ✅ Production Ready  
**Last Update**: June 2024  
**Version**: 1.0.0

---

*BankClust - Transforming Banking Insights Through AI-Driven Analytics*
