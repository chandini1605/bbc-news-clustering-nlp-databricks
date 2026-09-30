# ============================================================
# STEP 9: SKLEARN PIPELINE + MLFLOW TRACKING
# TF-IDF → SVD → Normalizer → KMeans
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import mlflow
import mlflow.sklearn
import pickle, os, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("outputs/cleaned_news.csv")
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

label_map = {'business':0,'sport':1,'politics':2,'tech':3,'entertainment':4}
y_true    = df['category'].map(label_map).values

print("=" * 60)
print("STEP 9: SKLEARN PIPELINE + MLFLOW TRACKING")
print("=" * 60)

# ── 1. Build Full Pipeline ────────────────────────────────────
print("\n📌 Building Sklearn Pipeline:")
print("   TF-IDF → TruncatedSVD → Normalizer → KMeans")

mlflow.set_experiment("BBC_News_Clustering")

# Experiment configs to track
configs = [
    {"max_features": 3000, "n_components": 50,  "n_clusters": 5},
    {"max_features": 5000, "n_components": 100, "n_clusters": 5},
    {"max_features": 5000, "n_components": 100, "n_clusters": 6},
]

best_sil   = -1
best_pipe  = None
best_cfg   = None

for cfg in configs:
    run_name = f"tfidf{cfg['max_features']}_svd{cfg['n_components']}_k{cfg['n_clusters']}"
    print(f"\n   Running: {run_name}")

    with mlflow.start_run(run_name=run_name):

        # Log parameters
        mlflow.log_params(cfg)

        # Build pipeline
        pipe = Pipeline([
            ('tfidf',      TfidfVectorizer(max_features=cfg['max_features'],
                                           min_df=2, max_df=0.95,
                                           ngram_range=(1,2), sublinear_tf=True)),
            ('svd',        TruncatedSVD(n_components=cfg['n_components'], random_state=42)),
            ('normalizer', Normalizer(copy=False)),
            ('kmeans',     KMeans(n_clusters=cfg['n_clusters'], random_state=42, n_init=10))
        ])

        # Fit & predict
        labels = pipe.fit_predict(df['cleaned_text'])

        # Only score if more than 1 cluster found
        unique_labels = len(set(labels))
        if unique_labels > 1:
            # Get transformed features for scoring
            X_transformed = pipe[:-1].transform(df['cleaned_text'])
            sil = silhouette_score(X_transformed, labels)
            db  = davies_bouldin_score(X_transformed, labels)
            ari = adjusted_rand_score(y_true, labels)
        else:
            sil = db = ari = 0.0

        # Log metrics
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("davies_bouldin",   db)
        mlflow.log_metric("adjusted_rand_index", ari)

        print(f"   Silhouette : {sil:.4f}")
        print(f"   DB Score   : {db:.4f}")
        print(f"   ARI        : {ari:.4f}")

        # Log model
        mlflow.sklearn.log_model(pipe, "clustering_pipeline")

        if sil > best_sil:
            best_sil  = sil
            best_pipe = pipe
            best_cfg  = cfg

print(f"\n🏆 Best Config: {best_cfg}")
print(f"   Best Silhouette: {best_sil:.4f}")

# ── 2. Save Best Pipeline ────────────────────────────────────
pickle.dump(best_pipe, open('models/best_pipeline.pkl', 'wb'))
print("✅ Best pipeline saved to models/best_pipeline.pkl")

# ── 3. Pipeline Steps Visualization ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
steps = ['Raw Text', 'TF-IDF\nVectorizer', 'TruncatedSVD\n(LSA)', 'Normalizer', 'K-Means\nClustering', 'Cluster\nLabels']
x_pos = range(len(steps))
colors = ['#94a3b8','#2563eb','#16a34a','#d97706','#dc2626','#7c3aed']

for i, (step, color) in enumerate(zip(steps, colors)):
    ax.add_patch(plt.Rectangle((i*2, 0.3), 1.5, 0.4, color=color, alpha=0.85, zorder=2))
    ax.text(i*2 + 0.75, 0.5, step, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=3)
    if i < len(steps)-1:
        ax.annotate('', xy=(i*2+1.6, 0.5), xytext=(i*2+1.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_xlim(-0.2, len(steps)*2)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Sklearn Pipeline: Text Clustering Workflow',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/cluster_plots/06_pipeline_diagram.png', dpi=150)
plt.close()
print("✅ Saved: 06_pipeline_diagram.png")
print("\n✅ Step 9 Complete! MLflow experiment logged.")
print("   Run 'mlflow ui' to view dashboard")
