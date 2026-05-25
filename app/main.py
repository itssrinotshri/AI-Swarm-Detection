import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Ensure both local and parent folders are appended to path for import robustness
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.swarm_detection import SwarmDetector, combine_predictions

# --- Page Config ---
st.set_page_config(
    page_title="TwiBot-20 Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .metric-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00E676;
    }
    .metric-label {
        font-size: 1rem;
        color: #B0BEC5;
    }
    .bot-badge {
        background-color: #FF1744;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(255, 23, 68, 0.5);
    }
    .human-badge {
        background-color: #00B0FF;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 176, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_resource
def load_models():
    model_path = os.path.join(root_dir, "models", "lgbm_model.joblib")
    swarm_path = os.path.join(root_dir, "models", "swarm_detector.joblib")
    if not os.path.exists(model_path) or not os.path.exists(swarm_path):
        st.error("Model files not found. Please run the pipeline first.")
        st.stop()
    return joblib.load(model_path), joblib.load(swarm_path)

@st.cache_data
def load_test_data():
    data_path = os.path.join(root_dir, "data", "processed", "test.parquet")
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please run the pipeline first.")
        st.stop()
    return pd.read_parquet(data_path)

# --- Main App ---
def main():
    st.title("🤖 AI Swarm Detection Dashboard")
    st.markdown("Analyze Twitter accounts in real-time using a fused feature space and Swarm Intelligence.")

    # Load resources
    model, swarm_detector = load_models()
    df_test = load_test_data()

    # Sidebar
    st.sidebar.header("🔍 Account Selection")
    st.sidebar.markdown("Select a user from the test set to analyze.")
    
    user_ids = df_test['user_id'].tolist()
    selected_user = st.sidebar.selectbox("User ID", user_ids)
    
    # Get user data
    user_data = df_test[df_test['user_id'] == selected_user].iloc[0]
    
    # Separate features and label
    features = user_data.drop(['user_id', 'label'])
    true_label = user_data['label']
    
    # Make Base Prediction
    features_df = pd.DataFrame([features])
    base_prob = model.predict_proba(features_df)[0][1]
    
    # Make Swarm Prediction
    emb_cols = [c for c in features.index if str(c).startswith('emb_')]
    # To predict a single user's swarm score using the fitted KMeans, we predict their cluster 
    # and look up the stats. But our SwarmDetector currently only has fit_predict.
    # To quickly assign a score without refitting, we transform and predict the cluster:
    X_scaled = swarm_detector.scaler.transform(features_df[emb_cols])
    cluster_id = swarm_detector.cluster_model.predict(X_scaled)[0]
    
    # Get the swarm score from the pre-computed cluster_stats_ dictionary
    # If the cluster exists in stats, use it. Otherwise default to base probability.
    cluster_stat = swarm_detector.cluster_stats_.get(cluster_id, {})
    swarm_score = cluster_stat.get('swarm_score', base_prob)
    
    # Fused Prediction
    prob = combine_predictions(np.array([base_prob]), np.array([swarm_score]), alpha=0.7)[0]
    pred_class = 1 if prob >= 0.5 else 0

    # Main Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction")
        if pred_class == 1:
            st.markdown('<div class="bot-badge">🚨 MALICIOUS BOT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="human-badge">👤 GENUINE HUMAN</div>', unsafe_allow_html=True)
            
        st.markdown(f"**Final Confidence:** {prob*100:.2f}%")
        st.markdown(f"**True Label:** {'Bot' if true_label == 1 else 'Human'}")
        
        st.markdown("---")
        st.subheader("🐝 Swarm Intelligence")
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Swarm Risk Score (Cluster {cluster_id})</div>
            <div class="metric-value" style="color: {'#FF1744' if swarm_score >= 0.5 else '#00B0FF'}">{swarm_score*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Profile Metadata")
        
        # Display key features nicely
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Followers</div>
            <div class="metric-value">{int(user_data['followers']):,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Following</div>
            <div class="metric-value">{int(user_data['following']):,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Account Age (Days)</div>
            <div class="metric-value">{int(user_data['account_age_days']):,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Tweet Count</div>
            <div class="metric-value">{int(user_data['tweet_count']):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("Graph Topology")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.metric("Sampled Indegree", int(user_data['sampled_indegree']))
        with g_col2:
            st.metric("Sampled Outdegree", int(user_data['sampled_outdegree']))
            
        st.markdown("---")
        st.subheader("🧠 Local Explainability (SHAP)")
        st.markdown("Which features contributed most to this specific prediction?")
        
        # Generate SHAP
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)
            
            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]
                
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            
            # Create a simple bar plot of the top 10 features for this instance
            top_k = 10
            # Get absolute values to find top features
            abs_shap = np.abs(shap_vals)
            # Find indices of top_k absolute values
            top_indices = np.argsort(abs_shap)[-top_k:]
            
            top_features = features.index[top_indices]
            top_values = shap_vals[top_indices]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FF1744' if val > 0 else '#00B0FF' for val in top_values]
            y_pos = np.arange(len(top_features))
            
            ax.barh(y_pos, top_values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('SHAP Value (Impact on Model Output)')
            
            # Set colors for text based on dark theme
            ax.xaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_color('#333333')
                
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")

if __name__ == "__main__":
    main()
