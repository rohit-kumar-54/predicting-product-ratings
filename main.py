"""
Predicting Product Ratings Using Customer Reviews and Product Features
Main PySpark Pipeline & ML Training Script: main.py

Performs:
1. PySpark Data Preprocessing & Validation
2. MLlib Feature Engineering (Categorical, Text TF-IDF, Numerical Scaling)
3. Model Training (Logistic Regression & Random Forest Classifier)
4. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
5. Model & Artifact Serialization for Streamlit Dashboard
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# PySpark Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, regexp_replace, trim, count, lit
from pyspark.sql.types import DoubleType, IntegerType, StringType

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def create_spark_session():
    """Create and configure an optimized PySpark session for Windows."""
    print("=" * 70)
    print("STEP 2.1: Initializing Apache Spark Session")
    print("=" * 70)
    
    spark = SparkSession.builder \
        .appName("ProductRatingPrediction") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark Version: {spark.version}")
    print(f"Spark App Name: {spark.sparkContext.appName}")
    print("Spark Session successfully initialized.\n")
    return spark


def locate_data_file():
    """Find product_reviews.csv path whether running in root or subfolder."""
    possible_paths = [
        os.path.join("data", "product_reviews.csv"),
        os.path.join("Product_Rating_Prediction", "data", "product_reviews.csv"),
        os.path.join("..", "data", "product_reviews.csv")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not find 'product_reviews.csv'. Please run 'python generate_dataset.py' first!"
    )


def preprocess_data(spark, data_path):
    """Load, inspect, clean, and preprocess the dataset using PySpark."""
    print("=" * 70)
    print("STEP 2.2: Loading and Preprocessing Data in PySpark")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    # 1. Load CSV into Spark DataFrame
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    raw_count = df.count()
    print(f"Total Rows Loaded: {raw_count:,}")
    print(f"Total Columns: {len(df.columns)}")

    # 2. Display Schema
    print("\nDataset Schema:")
    df.printSchema()

    # 3. Display Sample Records
    print("\nSample 5 Records:")
    df.show(5, truncate=40)

    # 4. Check for Null / Missing Values
    print("\nChecking for Null / Missing Values:")
    null_counts = {c: df.filter(col(c).isNull()).count() for c in df.columns}
    for col_name, n_null in null_counts.items():
        print(f"  - {col_name:25s}: {n_null} missing")

    # Handle missing values if any (defensive imputation)
    df = df.na.drop(subset=["rating", "review_text", "product_category", "price"])

    # 5. Remove Duplicate Records
    init_count = df.count()
    df = df.dropDuplicates()
    dedup_count = df.count()
    print(f"\nDuplicates removed: {init_count - dedup_count}")

    # 6. Convert Data Types Explicitly
    df = df.withColumn("price", col("price").cast(DoubleType())) \
           .withColumn("customer_review_count", col("customer_review_count").cast(IntegerType())) \
           .withColumn("customer_avg_rating", col("customer_avg_rating").cast(DoubleType())) \
           .withColumn("rating", col("rating").cast(IntegerType()))

    # 7. Clean and Normalize Review Text
    # Lowercase, strip special symbols and punctuation, collapse extra spaces
    print("\nCleaning and normalizing review text...")
    df = df.withColumn(
        "review_clean",
        trim(regexp_replace(lower(col("review_text")), "[^a-zA-Z0-9\\s]", " "))
    )
    # Ensure no empty strings
    df = df.filter(col("review_clean") != "")

    # 8. Check Rating Distribution
    print("\nRating Distribution (1 to 5 Stars):")
    df.groupBy("rating").count().orderBy("rating").show()

    # 9. Create Binary Target Variable
    # Rating 4 or 5 = High Rating (1.0), Rating 1, 2, or 3 = Low Rating (0.0)
    print("Creating binary target label: Rating 4-5 -> 1.0 (High), Rating 1-3 -> 0.0 (Low)...")
    df = df.withColumn(
        "label",
        when(col("rating") >= 4, 1.0).otherwise(0.0).cast(DoubleType())
    )

    print("\nTarget Class Distribution:")
    df.groupBy("label").agg(
        count("*").alias("Count"),
        (count("*") / lit(df.count()) * 100).alias("Percentage (%)")
    ).orderBy("label").show()

    return df


def build_feature_engineering_pipeline():
    """
    Build MLlib Feature Engineering Pipeline:
    - Categorical: StringIndexer + OneHotEncoder for product_category and verified_purchase
    - Text: Tokenizer + StopWordsRemover + HashingTF + IDF
    - Numerical: VectorAssembler + StandardScaler for price, customer_review_count, customer_avg_rating
    - VectorAssembler: Merges all feature vectors into a single 'features' vector
    """
    print("=" * 70)
    print("STEP 2.3: Defining MLlib Feature Engineering Pipeline")
    print("=" * 70)

    # 1. Categorical Feature Stages
    cat_indexer = StringIndexer(
        inputCols=["product_category", "verified_purchase"],
        outputCols=["cat_idx", "verified_idx"],
        handleInvalid="keep"
    )
    cat_encoder = OneHotEncoder(
        inputCols=["cat_idx", "verified_idx"],
        outputCols=["cat_vec", "verified_vec"]
    )

    # 2. Text Feature Stages
    tokenizer = Tokenizer(inputCol="review_clean", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_text_tf", numFeatures=1000)
    idf = IDF(inputCol="raw_text_tf", outputCol="text_idf")

    # 3. Numerical Feature Stages
    num_cols = ["price", "customer_review_count", "customer_avg_rating"]
    num_assembler = VectorAssembler(inputCols=num_cols, outputCol="num_raw")
    num_scaler = StandardScaler(
        inputCol="num_raw",
        outputCol="num_scaled",
        withMean=True,
        withStd=True
    )

    # 4. Final Vector Assembler (combines all feature spaces)
    feature_assembler = VectorAssembler(
        inputCols=["cat_vec", "verified_vec", "text_idf", "num_scaled"],
        outputCol="features"
    )

    pipeline_stages = [
        cat_indexer,
        cat_encoder,
        tokenizer,
        stopwords_remover,
        hashing_tf,
        idf,
        num_assembler,
        num_scaler,
        feature_assembler
    ]
    print("Feature Engineering Pipeline Stages Defined:")
    for i, s in enumerate(pipeline_stages, 1):
        print(f"  {i}. {s.__class__.__name__} -> {s.getOutputCol() if hasattr(s, 'getOutputCol') else s.getOutputCols()}")

    return pipeline_stages


def train_and_evaluate_models(spark, df, feature_stages):
    """
    Train and evaluate Logistic Regression and Random Forest models.
    Split: 80% Train, 20% Test with fixed seed.
    """
    print("\n" + "=" * 70)
    print("STEP 2.4: Splitting Data and Training MLlib Models")
    print("=" * 70)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"Training Dataset Count: {train_count:,} (80%)")
    print(f"Testing Dataset Count:  {test_count:,} (20%)")

    # Models to train
    models_to_train = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=25,
            regParam=0.01,
            elasticNetParam=0.0
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=40,
            maxDepth=7,
            seed=42
        )
    }

    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    results = []
    trained_pipelines = {}
    test_predictions_dict = {}

    for name, clf in models_to_train.items():
        print(f"\n---> Training {name} MLlib Pipeline...")
        full_pipeline = Pipeline(stages=feature_stages + [clf])
        
        # Train strictly on training dataset
        start_time = datetime.now()
        pipeline_model = full_pipeline.fit(train_df)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"     Training completed in {duration:.2f} seconds.")

        # Test predictions
        test_pred = pipeline_model.transform(test_df)
        trained_pipelines[name] = pipeline_model
        test_predictions_dict[name] = test_pred

        # Compute Metrics
        acc = evaluator_acc.evaluate(test_pred)
        prec = evaluator_prec.evaluate(test_pred)
        rec = evaluator_rec.evaluate(test_pred)
        f1 = evaluator_f1.evaluate(test_pred)
        auc = evaluator_auc.evaluate(test_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "Training_Time_Sec": round(duration, 2)
        })

        print(f"\n{name} Evaluation Results:")
        print(f"  - Accuracy:  {acc:.4f} (Proportion of correct predictions overall)")
        print(f"  - Precision: {prec:.4f} (Weighted ability to avoid false positives)")
        print(f"  - Recall:    {rec:.4f} (Weighted ability to detect all actual positives)")
        print(f"  - F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")
        print(f"  - ROC-AUC:   {auc:.4f} (Area under the Receiver Operating Characteristic curve)")

    metrics_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    # Detailed Evaluation for Primary Model (Logistic Regression)
    primary_name = "Logistic Regression"
    primary_model = trained_pipelines[primary_name]
    primary_preds = test_predictions_dict[primary_name]

    print(f"\nDisplaying Sample Actual vs Predicted for {primary_name}:")
    primary_preds.select(
        "product_id",
        "product_category",
        "price",
        "label",
        "prediction",
        "probability"
    ).show(10, truncate=30)

    # Confusion Matrix calculation
    # True Positives (label=1, pred=1), False Positives (label=0, pred=1)
    # True Negatives (label=0, pred=0), False Negatives (label=1, pred=0)
    tp = primary_preds.filter("label = 1.0 AND prediction = 1.0").count()
    fp = primary_preds.filter("label = 0.0 AND prediction = 1.0").count()
    tn = primary_preds.filter("label = 0.0 AND prediction = 0.0").count()
    fn = primary_preds.filter("label = 1.0 AND prediction = 0.0").count()

    print("\nConfusion Matrix (Logistic Regression on Test Set):")
    print(f"  True Negatives (Actual Low, Predicted Low):    {tn:,}")
    print(f"  False Positives (Actual Low, Predicted High):  {fp:,}")
    print(f"  False Negatives (Actual High, Predicted Low):  {fn:,}")
    print(f"  True Positives (Actual High, Predicted High):  {tp:,}")

    return trained_pipelines, metrics_df, primary_preds, (tn, fp, fn, tp)


def save_artifacts(primary_model, metrics_df, test_preds, cm_values):
    """Save trained model, evaluation metrics, and sample predictions for Streamlit."""
    print("\n" + "=" * 70)
    print("STEP 2.5: Saving Model Artifacts & Evaluation Outputs")
    print("=" * 70)

    # 1. Target output directories
    target_dirs = ["outputs", os.path.join("Product_Rating_Prediction", "outputs")]
    model_dirs = [
        os.path.join("models", "product_rating_model"),
        os.path.join("Product_Rating_Prediction", "models", "product_rating_model")
    ]

    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "predictions"), exist_ok=True)

    # 2. Save evaluation metrics CSV
    for d in target_dirs:
        metrics_path = os.path.join(d, "evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved evaluation metrics to: {os.path.abspath(metrics_path)}")

    # 3. Save confusion matrix CSV
    tn, fp, fn, tp = cm_values
    cm_df = pd.DataFrame([
        {"Actual": "Low Rating (0)", "Predicted_Low": tn, "Predicted_High": fp},
        {"Actual": "High Rating (1)", "Predicted_Low": fn, "Predicted_High": tp}
    ])
    for d in target_dirs:
        cm_path = os.path.join(d, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=False)
        print(f"Saved confusion matrix to: {os.path.abspath(cm_path)}")

    # 4. Save test predictions sample (up to 5,000 rows) as CSV for interactive dashboard
    print("\nExtracting test predictions sample for interactive dashboard...")
    sample_df = test_preds.select(
        "product_id",
        "product_category",
        "price",
        "verified_purchase",
        "customer_id",
        "customer_review_count",
        "customer_avg_rating",
        "review_text",
        "rating",
        "label",
        "prediction"
    ).limit(5000).toPandas()

    for d in target_dirs:
        pred_path = os.path.join(d, "predictions", "test_predictions.csv")
        sample_df.to_csv(pred_path, index=False)
        print(f"Saved test predictions sample ({len(sample_df)} records) to: {os.path.abspath(pred_path)}")

    # 5. Save PySpark PipelineModel
    for md in model_dirs:
        os.makedirs(os.path.dirname(md), exist_ok=True)
        try:
            primary_model.write().overwrite().save(md)
            print(f"Saved PySpark PipelineModel to: {os.path.abspath(md)}")
        except Exception as e:
            print(f"Note on saving model to {md}: {e}")

    print("\n" + "=" * 70)
    print("ALL PIPELINE OPERATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main():
    spark = create_spark_session()
    try:
        data_path = locate_data_file()
        df = preprocess_data(spark, data_path)
        feature_stages = build_feature_engineering_pipeline()
        trained_pipelines, metrics_df, primary_preds, cm_values = train_and_evaluate_models(
            spark, df, feature_stages
        )
        save_artifacts(trained_pipelines["Logistic Regression"], metrics_df, primary_preds, cm_values)
    finally:
        spark.stop()
        print("\nSpark session closed cleanly.")


if __name__ == "__main__":
    main()

