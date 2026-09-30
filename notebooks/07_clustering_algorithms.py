# ============================================================
# STEP 7: CLUSTERING ALGORITHMS
# K-Means + DBSCAN + Agglomerative
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              adjusted_rand_score, adjusted_mutual_info_score,
                              confusion_matrix)
from scipy.cluster.hierarchy import dendrogram, linkage
import pickle, os, warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/cluster_plots", exist_ok=True)

df     = pd.read_csv("outputs/cleaned_news.csv")
X_pca  = np.load("outputs/X_pca.npy")
X_tsne = np.load("outputs/X_tsne.npy")

# Encode true labels
label_map = {'business':0,'sport':1,'politics':2,'tech':3,'entertainment':4}
y_true = df['category'].map(label_map).values

colors5 = ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed']

print("=" * 60)
print("STEP 7: CLUSTERING ALGORITHMS")
print("K-Means | DBSCAN | Agglomerative")
print("=" * 60)

results = {}

# ── 1. K-Means ────────────────────────────────────────────────
print("\n📌 Algorithm 1: K-Means (K=5)")
km = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(X_pca)

sil = silhouette_score(X_pca, km_labels)
db  = davies_bouldin_score(X_pca, km_labels)
ari = adjusted_rand_score(y_true, km_labels)
ami = adjusted_mutual_info_score(y_true, km_labels)

results['KMeans'] = {'labels': km_labels, 'silhouette': sil, 'db': db, 'ari': ari, 'ami': ami}
print(f"   Silhouette  : {sil:.4f}")
print(f"   DB Score    : {db:.4f}")
print(f"   ARI         : {ari:.4f}")
print(f"   AMI         : {ami:.4f}")
pickle.dump(km, open('models/kmeans.pkl','wb'))

# ── 2. DBSCAN ─────────────────────────────────────────────────
print("\n📌 Algorithm 2: DBSCAN")
dbscan = DBSCAN(eps=2.5, min_samples=5, n_jobs=-1)
db_labels = dbscan.fit_predict(X_pca)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = list(db_labels).count(-1)
print(f"   Clusters Found : {n_clusters}")
print(f"   Noise Points   : {n_noise}")

if n_clusters > 1:
    mask = db_labels != -1
    sil2 = silhouette_score(X_pca[mask], db_labels[mask])
    db2  = davies_bouldin_score(X_pca[mask], db_labels[mask])
    ari2 = adjusted_rand_score(y_true[mask], db_labels[mask])
    ami2 = adjusted_mutual_info_score(y_true[mask], db_labels[mask])
else:
    sil2 = db2 = ari2 = ami2 = 0.0

results['DBSCAN'] = {'labels': db_labels, 'silhouette': sil2, 'db': db2, 'ari': ari2, 'ami': ami2}
print(f"   Silhouette    : {sil2:.4f}")
print(f"   ARI           : {ari2:.4f}")

# ── 3. Agglomerative ──────────────────────────────────────────
print("\n📌 Algorithm 3: Agglomerative Clustering (K=5)")
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
agg_labels = agg.fit_predict(X_pca)

sil3 = silhouette_score(X_pca, agg_labels)
db3  = davies_bouldin_score(X_pca, agg_labels)
ari3 = adjusted_rand_score(y_true, agg_labels)
ami3 = adjusted_mutual_info_score(y_true, agg_labels)

results['Agglomerative'] = {'labels': agg_labels, 'silhouette': sil3, 'db': db3, 'ari': ari3, 'ami': ami3}
print(f"   Silhouette  : {sil3:.4f}")
print(f"   DB Score    : {db3:.4f}")
print(f"   ARI         : {ari3:.4f}")
print(f"   AMI         : {ami3:.4f}")
pickle.dump(agg, open('models/agglomerative.pkl','wb'))

# ── 4. t-SNE Cluster Plots ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
algo_labels = [('KMeans', km_labels), ('DBSCAN', db_labels), ('Agglomerative', agg_labels)]

for ax, (name, labels) in zip(axes, algo_labels):
    unique = sorted(set(labels))
    cmap   = plt.cm.get_cmap('tab10', len(unique))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        lname = f'Noise' if lbl == -1 else f'Cluster {lbl}'
        ax.scatter(X_tsne[mask,0], X_tsne[mask,1],
                   c=[cmap(i)], label=lname, alpha=0.6, s=15)
    ax.set_title(f'{name}\nSilhouette={results[name]["silhouette"]:.3f}  ARI={results[name]["ari"]:.3f}',
                 fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)

plt.suptitle('Clustering Results: K-Means vs DBSCAN vs Agglomerative',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/04_clustering_comparison.png', dpi=150)
plt.close()
print("\n✅ Saved: 04_clustering_comparison.png")

# ── 5. Algorithm Comparison Table ────────────────────────────
print("\n📌 Algorithm Comparison:")
comp = pd.DataFrame({n: {
    'Silhouette ↑': r['silhouette'],
    'DB Score ↓'  : r['db'],
    'ARI ↑'       : r['ari'],
    'AMI ↑'       : r['ami']
} for n, r in results.items()}).T
print(comp.round(4))

# ── 6. Dendrogram (Agglomerative) ────────────────────────────
print("\n📌 Plotting Dendrogram (sample 100 docs)...")
sample_idx = np.random.choice(len(X_pca), 100, replace=False)
Z = linkage(X_pca[sample_idx], method='ward')

plt.figure(figsize=(16, 6))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90,
           color_threshold=0.7*max(Z[:,2]))
plt.title('Hierarchical Clustering Dendrogram (100 sample docs)',
          fontweight='bold', fontsize=13)
plt.xlabel('Document Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/05_dendrogram.png', dpi=150)
plt.close()
print("✅ Saved: 05_dendrogram.png")

# Save labels
np.save("outputs/km_labels.npy",  km_labels)
np.save("outputs/db_labels.npy",  db_labels)
np.save("outputs/agg_labels.npy", agg_labels)
print("\n✅ Step 7 Complete!")
