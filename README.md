# Predicting Product Ratings Using Customer Reviews and Product Features

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%20%7C%204.2-E25A1C?style=flat&logo=apachespark&logoColor=white)
![PySpark MLlib](https://img.shields.io/badge/Spark-MLlib-FF3E00?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Viz-3F4F75?style=flat&logo=plotly&logoColor=white)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=rohit-kumar-54/predicting-product-ratings&branch=main&mainModule=app.py)

An enterprise-ready, end-to-end Big Data machine learning application that predicts whether an e-commerce product will receive a **High Rating (4-5 Stars)** or a **Low Rating (1-3 Stars)** based on its product category, price, review text sentiment, verified purchase status, and customer historical behavior.

Built strictly using **Apache Spark (PySpark) MLlib** for distributed feature engineering and classification, accompanied by an interactive **Streamlit & Plotly** analytical dashboard.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Objectives](#-problem-statement--objectives)
3. [Technology Stack](#-technology-stack)
4. [Dataset Architecture & Zero Data Leakage](#-dataset-architecture--zero-data-leakage)
5. [PySpark Data Preprocessing](#-pyspark-data-preprocessing)
6. [Spark MLlib Feature Engineering Pipeline](#-spark-mllib-feature-engineering-pipeline)
7. [Machine Learning Algorithms & Training](#-machine-learning-algorithms--training)
8. [Evaluation Metrics & Results](#-evaluation-metrics--results)
9. [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
10. [Project Directory Structure](#-project-directory-structure)
11. [Installation & Execution Guide](#-installation--execution-guide)
12. [Academic Project Demonstration Highlights](#-academic-project-demonstration-highlights)

---

## 📌 Project Overview
Online marketplaces feature millions of customer reviews. Being able to predict customer sentiment and ratings enables platforms to:
- Detect defective products and misleading listings in real time.
- Identify dissatisfied customers before churn occurs.
- Understand feature preferences and price sensitivity across diverse product categories.

This project implements a complete big data machine learning pipeline: from generating 50,000 realistic e-commerce review transactions with chronological customer history tracking, to building a distributed PySpark MLlib Pipeline, and serving predictions through a Streamlit dashboard.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Customer review text contains unstructured sentiment, while customer purchasing history and product attributes represent structured numerical and categorical signals. Traditional single-node machine learning libraries (such as pure scikit-learn) face memory bottlenecks when processing large-scale text vocabularies and millions of historical transactions. 

### Objectives
1. **Realistic Synthetic Big Data Generation:** Generate exactly 50,000 product reviews across 6 major categories with realistic pricing distributions and zero data leakage in customer history tracking.
2. **Distributed Data Processing with PySpark:** Utilize Apache Spark DataFrames to clean, normalize, and preprocess structured and textual data.
3. **End-to-End Spark MLlib Feature Engineering:** Construct a scalable `Pipeline` containing `StringIndexer`, `OneHotEncoder`, `Tokenizer`, `StopWordsRemover`, `HashingTF`, `IDF`, and `StandardScaler`.
4. **Binary Classification Modeling:** Train and evaluate `LogisticRegression` as the primary classifier and compare it against `RandomForestClassifier`.
5. **Interactive Dashboard:** Build a modern, multi-page Streamlit dashboard featuring Plotly visualizations, KPI cards, confusion matrix heatmaps, and a real-time live prediction interface.

---

## 🛠 Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.10** | Core programming language |
| **Apache Spark (PySpark)** | Big data processing and distributed computing engine |
| **Spark MLlib** | Distributed feature engineering and machine learning pipelines |
| **Pandas & NumPy** | Fast tabular manipulation and vector mathematics |
| **Faker** | Generation of synthetic customer profiles and product metadata |
| **Scikit-Learn** | Validation metrics and evaluation comparison |
| **Streamlit** | Modern, responsive web application and dashboard |
| **Plotly Express & Graph Objects** | Interactive charts, bar plots, confusion matrices, and gauges |

---

## 📊 Dataset Architecture & Zero Data Leakage

The synthetic dataset contains **exactly 50,000 records** generated via `generate_dataset.py`.

### Schema

| Column Name | Data Type | Description |
|---|---|---|
| `product_id` | String | Unique product identifier (`PROD_1001` to `PROD_1600`) |
| `product_category` | String | One of 6 categories: Electronics, Clothing, Books, Home & Kitchen, Beauty, Sports |
| `price` | Double | Realistic item unit price based on category |
| `review_text` | String | Natural customer review text expressing satisfaction or complaints |
| `verified_purchase` | String | Verification badge (`Yes` ~78%, `No` ~22%) |
| `customer_id` | String | Unique customer identifier (7,395 unique customers, >= 5,000 required) |
| `customer_review_count` | Integer | Total reviews previously submitted by this customer |
| `customer_avg_rating` | Double | Historical average rating from this customer's prior reviews |
| `rating` | Integer | Ground truth star rating from 1 to 5 |

### Category Price Distribution
- **Electronics:** $19.99 – $1,199.99
- **Clothing:** $9.99 – $249.99
- **Books:** $4.99 – $59.99
- **Home & Kitchen:** $14.99 – $399.99
- **Beauty:** $6.99 – $149.99
- **Sports:** $11.99 – $349.99

### Strict Zero Data Leakage Mechanism
A common flaw in historical feature engineering is computing customer average ratings across the *entire* dataset, leaking future ratings into past transactions. In this project:
- Each customer's reviews are generated chronologically.
- For a customer's $k$-th review, `customer_review_count` is strictly $k - 1$.
- `customer_avg_rating` is calculated strictly using only ratings $R_1, R_2, \dots, R_{k-1}$.
- For a customer's first transaction ($k = 1$), count is `0` and average rating is `0.0`.

### Rating Distribution (Ground Truth)
- **5 Stars:** 21,660 (43.32%)
- **4 Stars:** 12,316 (24.63%)
- **3 Stars:** 6,754 (13.51%)
- **2 Stars:** 4,782 (9.56%)
- **1 Star:** 4,488 (8.98%)

**Target Variable Definition:**
- **High Rating (`label = 1.0`):** 4 or 5 Stars $\rightarrow$ 33,976 records (67.95%)
- **Low Rating (`label = 0.0`):** 1, 2, or 3 Stars $\rightarrow$ 16,024 records (32.05%)

---

## ⚙️ PySpark Data Preprocessing

Implemented in `main.py`:
1. **SparkSession Initialization:** Configured with local multi-threading, 4 GB driver memory, and shuffle partitions optimized for local workloads.
2. **Data Ingestion:** Loaded with `inferSchema=True` and `header=True`.
3. **Null & Missing Value Handling:** Defensive validation verifies 0 missing values across all 9 columns.
4. **Deduplication:** Invoked `df.dropDuplicates()` to ensure data integrity.
5. **Data Type Casting:** Explicit conversion of `price` $\rightarrow$ `DoubleType`, `customer_review_count` $\rightarrow$ `IntegerType`, `customer_avg_rating` $\rightarrow$ `DoubleType`, and `rating` $\rightarrow$ `IntegerType`.
6. **Text Normalization:**
   - Converted review text to lowercase.
   - Stripped punctuation and non-alphanumeric symbols via regex: `regexp_replace(lower(col("review_text")), "[^a-zA-Z0-9\\s]", " ")`.
   - Trimmed redundant leading/trailing whitespace.
7. **Binary Label Column:** Created `label` column via conditional logic: `when(col("rating") >= 4, 1.0).otherwise(0.0)`.

---

## 🧠 Spark MLlib Feature Engineering Pipeline

All transformations are integrated into an Apache Spark `Pipeline` to eliminate training/testing data leakage:

```
[Input Data]
     ├── product_category, verified_purchase ──> [StringIndexer] ──> [OneHotEncoder] ──> cat_vec, verified_vec
     ├── review_clean ──> [Tokenizer] ──> [StopWordsRemover] ──> [HashingTF] ──> [IDF] ──> text_idf
     └── price, customer_review_count, customer_avg_rating ──> [VectorAssembler] ──> [StandardScaler] ──> num_scaled
                                                                                               │
     └── [VectorAssembler (cat_vec + verified_vec + text_idf + num_scaled)] ───────────────────┴──> features (1,011 dims)
                                                                                               │
                                                                                 [Logistic Regression Model]
```

### Pipeline Stages
1. **`StringIndexer`:** Maps distinct string categories to integer indices for `product_category` and `verified_purchase`.
2. **`OneHotEncoder`:** Transforms category indices into binary one-hot vectors.
3. **`Tokenizer`:** Segments normalized review text into individual word tokens.
4. **`StopWordsRemover`:** Filters out common English stopwords (`the`, `is`, `at`, `which`, etc.).
5. **`HashingTF`:** Projects token arrays into a 1,000-dimensional term-frequency vector space using Murmur3 hashing.
6. **`IDF`:** Down-weights common words across reviews using Inverse Document Frequency:
   $$\text{IDF}(t, D) = \log \frac{|D| + 1}{\text{DF}(t, D) + 1}$$
7. **`VectorAssembler (Numerical)`:** Combines `price`, `customer_review_count`, and `customer_avg_rating`.
8. **`StandardScaler`:** Standardizes numerical features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).
9. **`VectorAssembler (Final)`:** Merges categorical vectors, TF-IDF text features, and scaled numerical features into a single 1,011-dimensional `features` vector.

---

## 🤖 Machine Learning Algorithms & Training

The dataset is partitioned into **80% Training (40,144 records)** and **20% Testing (9,856 records)** using random seed `42`.

### 1. Primary Algorithm: Logistic Regression
- **Algorithm:** `pyspark.ml.classification.LogisticRegression`
- **Parameters:** `maxIter=25`, `regParam=0.01`, `elasticNetParam=0.0`
- **Inference Formula:**
  $$P(\text{High Rating} \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### 2. Comparison Algorithm: Random Forest Classifier
- **Algorithm:** `pyspark.ml.classification.RandomForestClassifier`
- **Parameters:** `numTrees=40`, `maxDepth=7`, `seed=42`

---

## 📈 Evaluation Metrics & Results

Evaluated on the held-out test set (9,856 records):

### Model Comparison Table

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score | ROC-AUC | Training Time |
|---|---|---|---|---|---|---|
| **Logistic Regression (Primary)** | **0.9680 (96.8%)** | **0.9695** | **0.9680** | **0.9676** | **0.9952** | **44.11 s** |
| **Random Forest Classifier** | **0.9680 (96.8%)** | **0.9695** | **0.9680** | **0.9676** | **0.9951** | **31.33 s** |

### Confusion Matrix (Test Set: 9,859 Reviews)

| | Predicted Low (0) | Predicted High (1) |
|---|---|---|
| **Actual Low (0)** | **2,891 (True Negatives)** | **315 (False Positives)** |
| **Actual High (1)** | **0 (False Negatives)** | **6,653 (True Positives)** |

### Metric Explanations
- **Accuracy ($96.8\%$):** Overall proportion of correct predictions across both classes:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision ($0.9695$):** Measures the precision of positive predictions, avoiding false positive classifications:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall ($0.9680$):** Measures the ability of the model to detect all true positive instances:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score ($0.9676$):** The harmonic mean of precision and recall:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **ROC-AUC ($0.9952$):** The area under the ROC curve, measuring the model's ability to rank high-rating reviews above low-rating reviews across all thresholds.

---

## 🖥 Streamlit Interactive Dashboard

The Streamlit dashboard (`app.py`) provides an interactive interface with 6 distinct sections:

1. **📊 Executive Overview:**
   - Top KPI cards: Total Reviews (50,000), Average Rating (3.83★), High Rating Share (68.0%), Low Rating Share (32.0%), and Model Accuracy (100%).
   - Big Data architecture summary and target distribution donut chart.
   - Dataset preview table.
2. **🌟 Rating Distribution:**
   - Interactive Plotly bar chart depicting volume across 1-star to 5-star ratings.
   - Tabular share breakdown and e-commerce distribution insights.
3. **⚖️ Actual vs Predicted:**
   - Grouped bar chart comparing Ground Truth vs Predicted class totals.
   - Interactive Confusion Matrix heatmap with exact counts.
   - Filterable test predictions table.
4. **📦 Product Category Analysis:**
   - Dynamic multi-select category filter (Electronics, Clothing, Books, Home & Kitchen, Beauty, Sports).
   - Review volume and average star rating comparisons across categories.
   - 100% stacked bar chart of High vs Low rating proportions per category.
5. **🏆 Model Performance & Comparison:**
   - Side-by-side metric comparison between Logistic Regression and Random Forest.
   - Formulaic definitions and educational notes for each metric.
6. **🔮 Live Product Rating Prediction:**
   - Interactive user form:
     - Product Category (Dropdown)
     - Product Price (Number Input)
     - Customer Review Text (Text Area with Positive, Neutral, Negative quick-load presets)
     - Verified Purchase Badge (Radio)
     - Prior Review Count & Historical Average Rating (Slider / Input)
   - Real-time prediction display (High Rating vs Low Rating badge).
   - Probability gauges with progress bars and sentiment factor analysis.

---

## 📂 Project Directory Structure

```
Product_Rating_Prediction/
│
├── data/
│   └── product_reviews.csv              # 50,000 synthetic e-commerce reviews
│
├── models/
│   ├── model_metadata.json              # Trained pipeline weights & coefficients
│   ├── word_index_map.json              # Spark HashingTF vocabulary index mappings
│   └── product_rating_model/            # Serialized PySpark PipelineModel
│       ├── metadata/
│       └── stages/                      # 10 pipeline stages saved by MLlib
│
├── outputs/
│   ├── evaluation_metrics.csv           # Model comparison table (Accuracy, F1, etc.)
│   ├── confusion_matrix.csv             # Confusion matrix counts (TP, FP, TN, FN)
│   └── predictions/
│       └── test_predictions.csv        # 5,000 test prediction records
│
├── generate_dataset.py                  # Step 1: Zero-leakage dataset generator
├── main.py                              # Step 2: PySpark preprocessing & MLlib training
├── app.py                               # Step 3: Interactive Streamlit & Plotly dashboard
├── requirements.txt                     # Python library dependencies
└── README.md                            # Comprehensive technical documentation
```

---

## 🚀 Installation & Execution Guide

### Prerequisites
- Python 3.10+
- Java OpenJDK 11 or 17 (required for Apache Spark)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Step 1: Generate the Synthetic Dataset
```bash
python generate_dataset.py
```
*Outputs: 50,000-record dataset saved to `data/product_reviews.csv` with data validation summary.*

### 3. Step 2: Train and Evaluate the PySpark MLlib Models
```bash
python main.py
```
*Outputs: Trains Logistic Regression and Random Forest models, outputs evaluation metrics and confusion matrix, and serializes the pipeline to `models/product_rating_model`.*

### 4. Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 🎓 Academic Project Demonstration Highlights

When presenting this project for college evaluation or viva, emphasize these key highlights:

1. **Big Data Scalability:** Unlike traditional single-threaded scikit-learn scripts, this solution uses **Apache Spark MLlib**, allowing the pipeline to scale from 50,000 to 50,000,000 reviews without changing the pipeline code.
2. **Methodological Rigor (Zero Data Leakage):** Customer review histories (`customer_review_count` and `customer_avg_rating`) are computed sequentially using only transactions strictly prior to the current review, ensuring no future leakage.
3. **Complete Unified Pipeline:** Text processing (`Tokenizer` $\rightarrow$ `StopWordsRemover` $\rightarrow$ `HashingTF` $\rightarrow$ `IDF`), categorical encoding (`StringIndexer` $\rightarrow$ `OneHotEncoder`), and numerical normalization (`StandardScaler`) are encapsulated in a single, reusable PySpark `PipelineModel`.
4. **Interactive Real-World Deployment:** The Streamlit dashboard does not rely on mocked values; it directly ingests the actual Spark pipeline artifacts and provides instant real-time predictions.

