"""
AI Swarm Detection on Twitter (X) - Master Pipeline
MSc Data Science Project

Description:
End-to-end pipeline for detecting individual Twitter bots and coordinated bot swarms.
Features behavioral analysis, NLP vectorization, DBSCAN clustering for swarm detection,
LightGBM classification, and SHAP explainability.

Suggested Folder Structure:
project_root/
  ├── data/                 # Raw and processed datasets
  ├── notebooks/            # Exploratory Jupyter notebooks
  ├── src/                  # Helper modules (if separated later)
  ├── outputs/              # Saved models, SHAP plots, metrics
  ├── master_pipeline.py    # THIS FILE
  ├── requirements.txt      # Dependencies
  └── README.md             # Project documentation

Example Execution:
$ python master_pipeline.py --data_path "data/twibot20.csv"
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
def load_data(file_path):
    """
    Loads dataset containing user profiles, tweets, and labels.
    If the file doesn't exist, generates dummy data for demonstration.
    """
    print("[*] Loading data...")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    print("    -> File not found. Generating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 5000
    df = pd.DataFrame({
        'user_id': range(n_samples),
        'screen_name': [f'user_{i}' for i in range(n_samples)],
        'followers_count': np.random.randint(0, 10000, n_samples),
        'friends_count': np.random.randint(0, 5000, n_samples),
        'statuses_count': np.random.randint(0, 50000, n_samples),
        'account_age_days': np.random.randint(1, 3650, n_samples),
        'description': ['Sample bio text'] * n_samples,
        'tweets': ['Sample tweet content here'] * n_samples,
        'label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # 0=Human, 1=Bot
    })
    return df

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def engineer_behavioral_features(df):
    """
    Extracts behavioral and profile-based features from raw metadata.
    """
    print("[*] Engineering behavioral features...")
    
    df['follower_friend_ratio'] = df['followers_count'] / (df['friends_count'] + 1)
    df['tweet_frequency'] = df['statuses_count'] / (df['account_age_days'] + 1)
    df['name_length'] = df['screen_name'].apply(len)
    df['has_description'] = df['description'].apply(lambda x: 1 if pd.notnull(x) and x != '' else 0)
    
    behavioral_cols = ['followers_count', 'friends_count', 'statuses_count', 
                       'account_age_days', 'follower_friend_ratio', 'tweet_frequency', 
                       'name_length', 'has_description']
    
    scaler = StandardScaler()
    df[behavioral_cols] = scaler.fit_transform(df[behavioral_cols])
    return df, behavioral_cols

def extract_nlp_features(df, text_column='tweets', max_features=100):
    """
    Converts textual data into numerical features using TF-IDF.
    """
    print("[*] Extracting NLP features (TF-IDF)...")
    df[text_column] = df[text_column].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    actual_features = tfidf_matrix.shape[1]
    tfidf_cols = [f'tfidf_{i}' for i in range(actual_features)]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
    
    df = pd.concat([df, tfidf_df], axis=1)
    return df, tfidf_cols, vectorizer

# ---------------------------------------------------------
# 3. SWARM DETECTION (UNSUPERVISED CLUSTERING)
# ---------------------------------------------------------
def detect_swarms(df, feature_cols, eps=0.5, min_samples=5):
    """
    Detects coordinated bot swarms using DBSCAN clustering.
    Assigns a 'swarm_score' based on the density of the cluster the user belongs to.
    """
    print("[*] Running Swarm Detection (DBSCAN)...")
    
    cluster_data = df[feature_cols].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    df['cluster_id'] = dbscan.fit_predict(cluster_data)
    
    # Calculate swarm density/size
    cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts()
    
    def assign_swarm_score(cluster_id):
        if cluster_id == -1:
            return 0.0 # Noise / Individual actor
        size = cluster_sizes.get(cluster_id, 0)
        return min(size / 50.0, 1.0) # Normalize score
        
    df['swarm_score'] = df['cluster_id'].apply(assign_swarm_score)
    print(f"    -> Detected {len(cluster_sizes)} potential coordinated swarms.")
    
    return df

# ---------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
def train_and_evaluate(df, feature_cols, target_col='label'):
    """
    Trains a LightGBM classifier integrating all features and evaluates performance.
    """
    print("[*] Splitting dataset...")
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("[*] Evaluating Model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f} (Target: ~0.8100)")
    print(f"ROC-AUC:  {roc:.4f} (Target: ~0.8550)")
    print(f"PR-AUC:   {pr:.4f} (Target: ~0.8280)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_test

# ---------------------------------------------------------
# 5. MODEL EXPLAINABILITY
# ---------------------------------------------------------
def explain_model(model, X_test, output_dir="outputs"):
    """
    Generates SHAP values to explain feature importance globally and locally.
    """
    print("[*] Generating SHAP explainability plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance (Bot vs Human)")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(save_path)
    print(f"    -> Saved SHAP explanation plot to '{save_path}'")

# ---------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Swarm Detection Pipeline")
    parser.add_argument('--data_path', type=str, default='data/twitter_dataset.csv', help='Path to dataset')
    args = parser.parse_args()

    print("\n=== Starting AI Swarm Detection Pipeline ===")
    
    # 1. Load Data
    df = load_data(args.data_path)
    
    # 2. Feature Engineering
    df, behavioral_cols = engineer_behavioral_features(df)
    df, tfidf_cols, vectorizer = extract_nlp_features(df)
    
    # 3. Swarm Detection
    clustering_features = behavioral_cols + tfidf_cols[:20] 
    df = detect_swarms(df, feature_cols=clustering_features)
    
    # 4. Final Model Training
    final_features = behavioral_cols + tfidf_cols + ['swarm_score']
    model, X_test = train_and_evaluate(df, final_features)
    
    # 5. Explainability
    explain_model(model, X_test)
    
    # 6. Save Artifacts
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/lightgbm_bot_model.pkl")
    joblib.dump(vectorizer, "outputs/tfidf_vectorizer.pkl")
    print("[*] Pipeline completed successfully. Models saved to 'outputs/'.\n")

if __name__ == "__main__":
    main()
