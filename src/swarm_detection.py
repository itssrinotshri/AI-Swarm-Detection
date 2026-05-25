import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SwarmDetector:
    """
    Groups accounts using KMeans clustering and assigns a swarm score to each user
    representing the density and bot proportion within their cluster.
    """
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.cluster_stats_ = {}
        
    def fit_predict(self, features_df: pd.DataFrame, base_predictions: np.ndarray) -> pd.DataFrame:
        """
        Fits clustering on features and calculates swarm risk coefficients.
        """
        df_out = features_df.copy()
        
        # 1. Scale Features
        X_scaled = self.scaler.fit_transform(features_df)
        
        # 2. Fit Clustering
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        df_out['cluster_id'] = cluster_labels
        df_out['base_pred'] = base_predictions
        
        # 3. Calculate Cluster statistics
        total_users = len(df_out)
        cluster_counts = df_out['cluster_id'].value_counts().to_dict()
        swarm_scores = np.zeros(total_users)
        
        for cid in np.unique(cluster_labels):
            mask = df_out['cluster_id'] == cid
            size = cluster_counts[cid]
            
            density = size / total_users
            bot_ratio = df_out.loc[mask, 'base_pred'].mean()
            
            # Swarm Score Formula
            swarm_score = bot_ratio * min(1.0, density * self.n_clusters)
            
            self.cluster_stats_[cid] = {
                'size': size,
                'density': density,
                'bot_ratio': bot_ratio,
                'swarm_score': swarm_score
            }
            swarm_scores[mask] = swarm_score
            
        df_out['swarm_score'] = swarm_scores
        return df_out

def combine_predictions(base_probas: np.ndarray, swarm_scores: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Fuses classifier probabilities with swarm risk factors:
    Final Probabilities = alpha * Base Model + (1 - alpha) * Swarm Score
    """
    swarm_scores_norm = np.clip(swarm_scores, 0.0, 1.0)
    return (alpha * base_probas) + ((1.0 - alpha) * swarm_scores_norm)
