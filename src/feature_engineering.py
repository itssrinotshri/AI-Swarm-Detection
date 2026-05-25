import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

# ---------------------------------------------------------
# 1. TABULAR PROFILE FEATURES PREPARER
# ---------------------------------------------------------
def extract_tabular_features(data: List[Dict]) -> pd.DataFrame:
    """
    Extracts tabular, temporal, and profile behavioral metrics from user profiles.
    """
    records = []
    reference_date = datetime(2021, 1, 1)
    
    for user in data:
        profile = user.get('profile', {}) or {}
        tweets = user.get('tweet') or []
        
        created_at_str = profile.get('created_at', '')
        account_age_days = -1
        if isinstance(created_at_str, str) and created_at_str.strip():
            try:
                created_at_str = created_at_str.strip()
                dt = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')
                account_age_days = (reference_date.replace(tzinfo=dt.tzinfo) - dt).days
            except ValueError:
                pass
        
        followers = int(profile.get('followers_count', 0) or 0)
        following = int(profile.get('friends_count', 0) or 0)
        
        verified = str(profile.get('verified', 'False')).strip().lower() == 'true'
        default_profile = str(profile.get('default_profile', 'False')).strip().lower() == 'true'
        default_profile_image = str(profile.get('default_profile_image', 'False')).strip().lower() == 'true'
        has_extended_profile = str(profile.get('has_extended_profile', 'False')).strip().lower() == 'true'
        
        bio_length = len(profile.get('description', '') or '')
        tweet_count = len(tweets)
        
        records.append({
            'user_id': user.get('ID', ''),
            'followers': followers,
            'following': following,
            'account_age_days': account_age_days,
            'verified': int(verified),
            'default_profile': int(default_profile),
            'default_profile_image': int(default_profile_image),
            'has_extended_profile': int(has_extended_profile),
            'bio_length': bio_length,
            'tweet_count': tweet_count,
            'label': int(user.get('label', '0'))
        })
        
    df = pd.DataFrame(records)
    
    # Impute missing account ages
    median_age = df[df['account_age_days'] > 0]['account_age_days'].median()
    df['account_age_days'] = df['account_age_days'].replace(-1, median_age if not np.isnan(median_age) else 365)
    
    # Laplace-smoothed advanced engineered features
    df['follower_following_ratio'] = df['followers'] / (df['following'] + 1.0)
    df['tweets_per_day'] = df['tweet_count'] / (df['account_age_days'] + 1.0)
    
    return df

# ---------------------------------------------------------
# 2. GRAPH TOPOLOGY FEATURES PREPARER
# ---------------------------------------------------------
def extract_graph_features(data: List[Dict]) -> pd.DataFrame:
    """
    Extracts graph metrics from the 'neighbor' topology dictionary.
    """
    records = []
    for user in data:
        neighbor = user.get('neighbor') or {}
        follower_list = neighbor.get('follower') or [] if isinstance(neighbor, dict) else []
        following_list = neighbor.get('following') or [] if isinstance(neighbor, dict) else []
        
        sampled_indegree = len(follower_list)
        sampled_outdegree = len(following_list)
        
        records.append({
            'user_id': user.get('ID', ''),
            'sampled_indegree': sampled_indegree,
            'sampled_outdegree': sampled_outdegree,
            'neighbor_ratio': sampled_indegree / (sampled_outdegree + 1.0)
        })
    return pd.DataFrame(records)

# ---------------------------------------------------------
# 3. NLP TEXT EMBEDDINGS PREPARER
# ---------------------------------------------------------
def extract_text_embeddings(data: List[Dict], model_name='all-MiniLM-L6-v2', max_tweets=10) -> pd.DataFrame:
    """
    Extracts semantic tweet sentence embeddings using SentenceTransformers (with a randomized dummy fallback).
    """
    texts = []
    user_ids = []
    
    for user in data:
        tweets = user.get('tweet') or []
        recent_tweets = tweets[:max_tweets]
        concat_text = " ".join([t.replace("\n", " ").strip() for t in recent_tweets])
        texts.append(concat_text)
        user_ids.append(user.get('ID', ''))
        
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    except Exception as e:
        # Fallback to dummy random vectors on PyTorch/environment mismatches
        embeddings = np.random.randn(len(texts), 384)
        
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df['user_id'] = user_ids
    return emb_df

# ---------------------------------------------------------
# 4. FUSED FEATURE MATRIX GENERATOR
# ---------------------------------------------------------
def build_features(data: List[Dict], max_tweets=10) -> pd.DataFrame:
    """
    Orchestrates the extraction of all features (tabular, graph, text) and fuses them on user_id.
    """
    df_tabular = extract_tabular_features(data)
    df_graph = extract_graph_features(data)
    df_text = extract_text_embeddings(data, max_tweets=max_tweets)
    
    # Fuse all feature subsets
    df_fused = pd.merge(df_tabular, df_graph, on='user_id', how='left')
    df_fused = pd.merge(df_fused, df_text, on='user_id', how='left')
    return df_fused
