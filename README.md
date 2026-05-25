# 🕸️ AI Swarm Detection on Twitter (X) Using NLP, DBSCAN, LightGBM, and SHAP

An advanced machine learning and natural language processing pipeline designed to detect malicious bot accounts and coordinated bot swarms on social media. Developed as an MSc Data Science thesis project, this system transcends isolated account analysis by fusing individual profile anomalies with unsupervised coordinated network density scoring.

---

## 📌 Problem Statement

Traditional anti-abuse platforms evaluate accounts in isolation. However, modern malicious actors deploy **swarms**—highly coordinated networks of automated accounts that share subtle behavioral similarities, temporal rhythms, and semantic posting patterns. By keeping individual profile characteristics within normal human bounds, swarms easily bypass traditional classification models. 

This project solves this critical gap using a **Hybrid Late-Fusion Stacking Architecture** that merges individual supervised risk probabilities with unsupervised network density shapes.

---

## 🏗️ System Architecture

The pipeline processes user profile metadata, topology relationship lists, and recent tweet histories through three distinct stages:

```text
              [ User profiles & tweet streams ]
                             │
                  [ Feature Engineering ]
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
 [Tabular Profile]    [Graph Topology]     [NLP TF-IDF Vectors]
      │                      │                      │
      └──────────────────────┼──────────────────────┘
                             ▼
                   [ Fused Feature Space ]
                             │
      ┌──────────────────────┴──────────────────────┐
      ▼                                             ▼
[ LightGBM Stage 1 ]                     [ DBSCAN Clustering Stage 2 ]
      │                                             │
      ▼                                             ▼
 [Base Probabilities]                        [Coordinated Swarm Scores]
      │                                             │
      └──────────────────────┬──────────────────────┘
                             ▼
                   [ Late Fusion Combines ]
                             │
                             ▼
                 [ Final Threat Risk Badge ]
                             │
                             ▼
                 [ Interpretability (TreeSHAP) ]
```

1. **Feature Engineering flat module**: Extracts account metadata rates, Laplace-smoothed followers ratios, indegree/outdegree neighborhood ratios, and transforms text into dense numerical features using TF-IDF.
2. **Stage 1 (Supervised classification)**: Trains a **LightGBM** gradient booster on standard and text vectors to predict base probabilities.
3. **Stage 2 (Unsupervised Clustering)**: Employs **DBSCAN** to group users based on spatial distance. Dense coordinate clusters are assigned a normalized `swarm_score`.
4. **Stage 3 (Late Fusion)**: Combines base classifier weights with swarm coordinates:
   $$\text{Final Probabilities} = 0.7 \times P_{\text{Base}} + 0.3 \times S_{\text{Swarm}}$$
5. **Interpretability**: Integrates **TreeSHAP** to rank feature parameters globally and chart individual contributions.

---

## 📂 Reorganized Clean Structure

```text
AI-Swarm-Detection/
│
├── app/
│   └── streamlit_app.py        # Presentation-ready single-page dashboard
│
├── data/
│   ├── raw/                    # train.json, dev.json, test.json
│   └── processed/              # train.parquet, dev.parquet, test.parquet
│
├── models/
│   ├── lgbm_model.joblib       # Pre-trained LightGBM classifier checkpoint
│   └── swarm_detector.joblib   # Pre-trained Swarm Detector checkpoint
│
├── notebooks/
│   └── main_training.ipynb     # Step-by-step runnable pipeline execution notebook
│
├── outputs/                    # Output directory for reports and figures
│   ├── plots/                  # Confusion Matrices and performance curves
│   ├── shap/                   # Global TreeSHAP feature ranking plots
│   └── reports/                # Tabular classification logs
│
├── src/                        # Flat, simplified production source modules
│   ├── preprocessing.py        # Data loaders and raw structures cleaning
│   ├── feature_engineering.py  # Tabular, graph, and NLP feature pipelines
│   ├── swarm_detection.py      # DBSCAN / KMeans SwarmDetector classes
│   ├── training.py             # LightGBM booster loops
│   └── evaluation.py           # Precision, Recall, and SHAP saving routines
│
├── main.py                     # Unified CLI project orchestrator
├── requirements.txt            # Python environment packages dependencies
└── .gitignore                  # Standard repository version control exclusions
```

---

## 📈 Evaluation Results (TwiBot-20)

| Metric | Pipeline Result | Status |
| :--- | :--- | :--- |
| **Accuracy** | **`81.00%`** | **Exact Academic Match** |
| **ROC-AUC** | **`83.96%`** | **Highly Consistent** |
| **PR-AUC** | **`80.81%`** | **Highly Consistent** |

---

## ⚙️ Installation & Usage

### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Modular Pipeline
Re-engineer features, cluster networks, and train/evaluate the LightGBM classifier using the central CLI orchestrator:
```bash
python main.py --run-pipeline
```

### 3. Launching the Streamlit Web Application
Boot up the presentation dashboard:
```bash
python main.py --run-app
```
Navigate to your web browser at **`http://localhost:8501`** to interact with the audit forms, SHAP local graphs, and prediction meters in real-time.
