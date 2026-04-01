import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

from config import *
from utils import *
from feature_eng import process_features

# Simply setup MLflow here!
setup_mlflow(EXPERIMENT_NAME)

def load_data_from_db(days_back=30):
    """
    Load transaction data from PostgreSQL for the last N days.
    Uses database credentials from config.
    
    Args:
        days_back: Number of days to load (default: 30 for last month)
    
    Returns:
        pandas DataFrame with transaction data
    """
    try:
        # Create connection string from config
        connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Querying data from {start_date.date()} to {end_date.date()}")
        
        # Query transactions from the last month
        # Adjust the query based on your actual table schema
        query = text(f"""
            SELECT * 
            FROM transactions 
            WHERE created_at >= :start_date 
            AND created_at <= :end_date
            ORDER BY created_at DESC
        """)
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                'start_date': start_date,
                'end_date': end_date
            })
        
        print(f"✅ Loaded {len(df)} transactions from database")
        engine.dispose()
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        print("Falling back to CSV file...")
        return pd.read_csv(r"..\data\transactions.csv")

def main():
    print("Loading data from database...")
    
    # Load data from PostgreSQL (last 30 days)
    # Adjust days_back if you want more/less data
    df = load_data_from_db(days_back=30)
    
    # Sample 90% of the data
    df_sample = df.sample(frac=0.9, random_state=42).reset_index(drop=True)
    print(f"Using {len(df_sample)} rows for modeling (90% of total data).")

    print("Engineering features...")
    df_features = process_features(df_sample)
    
    # Target variable
    y = df_features['is_fraud'].astype(int).values
    X = df_features.drop(columns=['is_fraud'])
    
    # Train / Validation Split (80% train, 20% val of the 90% sampled data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Filter training data to ONLY include normal transactions (y == 0)
    normal_idx = (y_train == 0)
    X_train_normal = X_train[normal_idx]
    y_train_normal = y_train[normal_idx]
    
    print(f"Original Training set shape: {X_train.shape}")
    print(f"Filtered Training set (Normal Only) shape: {X_train_normal.shape}")
    print(f"Validation set shape (Mixed): {X_val.shape}")
    
    print("Training model on ONLY normal transactions...")
    
    # Since we train on pure normal data, we expect very little contamination.
    # But for the model parameter, we can use the overall dataset fraud rate 
    # to help it learn the proportion of anomalies we expect in the real world.
    overall_fraud_rate = float(np.mean(y))
    contamination = max(0.001, overall_fraud_rate)
    
    # Start MLflow run
    with mlflow.start_run(run_name="Isolation_Forest_Baseline"):
        
        # Define and log parameters
        params = {
            "n_estimators": 300,
            "max_samples": 256,
            "contamination": contamination,
            "random_state": 42,
            "n_jobs": -1
        }
        mlflow.log_params(params)
        
        model=IsolationForest(**params)
        model.fit(X_train_normal)
        
        print("Evaluating model performance on Validation Set...")
        # Evaluate score
        val_score = -model.decision_function(X_val)
    
        # To improve recall, instead of using a rigid quantile threshold based on contamination,
        # we can use the validation set to find the threshold that gives us a better balance (e.g., maximizing the F1 score for the minority class)
        # or manually relaxing the threshold.
        
        if len(np.unique(y_val)) >= 2:
            precisions, recalls, thresholds = precision_recall_curve(y_val, val_score)
            
            # Calculate F1 scores for each threshold
            # Add a tiny epsilon to avoid division by zero
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            
            # Get the index of the best F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            
            print(f"Optimized Threshold for better Recall/Precision balance: {best_threshold:.4f}")
            mlflow.log_metric("optimized_threshold", float(best_threshold))
            
            y_pred = (val_score >= best_threshold).astype(int)
        else:
            # Fallback if validation set has only 1 class (highly unlikely)
            threshold = np.quantile(val_score, 1 - contamination)
            y_pred = (val_score >= threshold).astype(int)
        
        print("\n=== Validation Results ===")
        if len(np.unique(y_val)) >= 2:
            pr_auc = average_precision_score(y_val, val_score)
            roc_auc = roc_auc_score(y_val, val_score)
            best_f1 = f1_scores[best_idx]
            
            print(f"PR-AUC:  {pr_auc:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            
            # Log metrics to MLflow
            my_result = {
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "f1_score": float(best_f1)
            }
            log_metrics(mlflow, my_result)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Normal', 'Fraud']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        # Save the model locally
        model_filename = 'fraud_isolation_forest.pkl'
        joblib.dump(model, model_filename)
        print(f"\nModel successfully saved locally to {model_filename}")
        
        # Log the model to MLflow (uploads to your local S3 artifact store)
        print("Uploading model to MLflow...")
        log_model(mlflow, model, "FraudDetectionModel_Champion", artifact_path="fraud_isolation_forest_model")


if __name__ == "__main__":
    main()
