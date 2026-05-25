import sys
import os
import joblib
import numpy as np

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from simplified flat modules
from src.preprocessing import load_data, get_data_paths
from src.feature_engineering import build_features
from src.swarm_detection import SwarmDetector, combine_predictions
from src.training import train_model
from src.evaluation import evaluate_model

def main():
    paths = get_data_paths()
    
    print("Loading raw Twitter dataset...")
    train_data = load_data(paths['train'])
    dev_data = load_data(paths['dev'])
    test_data = load_data(paths['test'])
    
    print(f"Ingested {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test user profiles.")
    
    print("\n--- Phase 1: Feature Extraction & Engineering ---")
    print("Building train set feature matrices...")
    df_train = build_features(train_data, max_tweets=10)
    
    print("Building dev set feature matrices...")
    df_dev = build_features(dev_data, max_tweets=10)
    
    print("Building test set feature matrices...")
    df_test = build_features(test_data, max_tweets=10)
    
    # Save processed data partitions
    print("\nSaving processed Parquet partitions...")
    os.makedirs('data/processed', exist_ok=True)
    df_train.to_parquet('data/processed/train.parquet')
    df_dev.to_parquet('data/processed/dev.parquet')
    df_test.to_parquet('data/processed/test.parquet')
    
    # Separate features and labels
    features_to_drop = ['user_id', 'label']
    
    X_train = df_train.drop(columns=features_to_drop)
    y_train = df_train['label']
    
    X_dev = df_dev.drop(columns=features_to_drop)
    y_dev = df_dev['label']
    
    X_test = df_test.drop(columns=features_to_drop)
    y_test = df_test['label']
    
    print("\n--- Phase 2: Supervised LightGBM Training ---")
    model = train_model(X_train, y_train, X_dev, y_dev)
    
    print("\n--- Phase 3: Unsupervised Coordinated Swarm Detection ---")
    y_pred_base = model.predict(X_test)
    y_prob_base = model.predict_proba(X_test)[:, 1]
    
    # Isolate sentence embedding feature columns
    emb_cols = [c for c in X_test.columns if str(c).startswith('emb_')]
    
    # Group accounts and calculate swarm risks
    swarm_detector = SwarmDetector(n_clusters=max(2, len(X_test) // 50))
    df_swarm = swarm_detector.fit_predict(X_test[emb_cols], y_pred_base)
    swarm_scores = df_swarm['swarm_score'].values
    
    print("Fusing Base Model Probabilities and DBSCAN Swarm Risk weights...")
    final_probabilities = combine_predictions(y_prob_base, swarm_scores, alpha=0.7)
    
    print("\n--- Phase 4: Model Performance Metrics & Auditing ---")
    # Evaluate with Late Fusion Fused Probabilities
    evaluate_model(model, X_test, y_test, y_pred_proba=final_probabilities)
    
    # Serialize model checkpoints
    print("\nSaving trained models and Swarm detector checkpoints...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_model.joblib')
    joblib.dump(swarm_detector, 'models/swarm_detector.joblib')
    print("Models saved successfully under 'models/' directory!")

if __name__ == "__main__":
    main()
