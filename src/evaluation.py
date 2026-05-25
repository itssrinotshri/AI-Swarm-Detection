import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import lightgbm as lgb

def evaluate_model(
    model: lgb.LGBMClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    y_pred_proba: np.ndarray = None,
    outputs_dir: str = "outputs"
):
    """
    Evaluates classification performance, prints a detailed validation report,
    and exports confusion matrices and TreeSHAP charts to output subfolders.
    """
    # Create output folders
    plots_dir = os.path.join(outputs_dir, "plots")
    shap_dir = os.path.join(outputs_dir, "shap")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)
    
    print("\n--- Model Evaluation Summary ---")
    if y_pred_proba is None:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 1. Print Standard Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # 2. Recall at 95% Precision
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    thresholds = np.append(thresholds, 1.0)
    
    high_precision_mask = precisions >= 0.95
    if any(high_precision_mask):
        valid_recalls = recalls[high_precision_mask]
        best_recall = np.max(valid_recalls)
        best_idx = np.where((precisions >= 0.95) & (recalls == best_recall))[0][0]
        optimal_threshold = thresholds[best_idx]
        print(f"Recall @ 95% Precision: {best_recall:.4f} (Threshold: {optimal_threshold:.4f})")
    else:
        print("Recall @ 95% Precision: Could not satisfy 95% threshold boundary.")
        
    # 3. Plot and Export Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'Bot'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix plot to: '{cm_path}'")
    
    # 4. TreeSHAP Global Feature Importance
    print("Generating SHAP explanations...")
    try:
        explainer = shap.TreeExplainer(model)
        sample_size = min(1000, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_sample, show=False)
        plt.title("SHAP Feature Importance (Bot vs Human)")
        plt.tight_layout()
        shap_path = os.path.join(shap_dir, "shap_summary.png")
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved TreeSHAP Summary Plot to: '{shap_path}'")
    except Exception as e:
        print(f"Warning: TreeSHAP explanations skipped due to PyTorch/environment error ({e})")
