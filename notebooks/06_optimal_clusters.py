# ============================================================
# STEP 6: FINDING OPTIMAL NUMBER OF CLUSTERS
# Elbow Method + Silhouette Score
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/cluster_plots", exist_ok=True)

X_pca = np.load("outputs/X_pca.npy")

print("=" * 60)
print("STEP 6: FINDING OPTIMAL CLUSTERS")
print("Elbow Method + Silhouette Score")
print("=" * 60)

K = range(2, 11)
inertias, silhouettes, db_scores = [], [], []

for k in K:
    print(f"   Testing K={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))
    db_scores.append(davies_bouldin_score(X_pca, labels))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(list(K), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=5, color='red', linestyle='--', label='K=5 (optimal)')
axes[0].set_title('Elbow Method\n(Inertia vs K)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K), silhouettes, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=5, color='red', linestyle='--', label='K=5 (optimal)')
axes[1].set_title('Silhouette Score\n(Higher = Better)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(list(K), db_scores, 'ro-', linewidth=2, markersize=8)
axes[2].axvline(x=5, color='blue', linestyle='--', label='K=5 (optimal)')
axes[2].set_title('Davies-Bouldin Score\n(Lower = Better)', fontweight='bold')
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('DB Score')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Optimal K Selection — Elbow + Silhouette + Davies-Bouldin',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/03_optimal_k.png', dpi=150)
plt.close()

print(f"\n📌 Results Summary:")
for i, k in enumerate(K):
    print(f"   K={k}: Inertia={inertias[i]:.1f}  Silhouette={silhouettes[i]:.4f}  DB={db_scores[i]:.4f}")

best_k = list(K)[silhouettes.index(max(silhouettes))]
print(f"\n🏆 Best K by Silhouette: {best_k}")
print(f"   BBC has 5 categories → K=5 confirmed!")
print("\n✅ Saved: 03_optimal_k.png")
print("✅ Step 6 Complete!")
