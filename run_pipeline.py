# ============================================================
# MAIN PIPELINE RUNNER
# BBC News Clustering — Run All Steps
# ============================================================

import subprocess, sys, os, time

steps = [
    ("Step 1:  Data Collection",          "notebooks/01_data_collection.py"),
    ("Step 2:  EDA",                       "notebooks/02_eda.py"),
    ("Step 3:  Text Preprocessing (NLP)", "notebooks/03_text_preprocessing.py"),
    ("Step 4:  Feature Extraction TF-IDF","notebooks/04_feature_extraction.py"),
    ("Step 5:  Dimensionality Reduction", "notebooks/05_dimensionality_reduction.py"),
    ("Step 6:  Optimal Clusters",         "notebooks/06_optimal_clusters.py"),
    ("Step 7:  Clustering Algorithms",    "notebooks/07_clustering_algorithms.py"),
    ("Step 8:  Evaluation + Cross Val",   "notebooks/08_evaluation.py"),
    ("Step 9:  Pipeline + MLflow",        "notebooks/09_pipeline_mlflow.py"),
    ("Step 10: Cluster Analysis + Viz",   "notebooks/10_cluster_analysis_viz.py"),
]

print("=" * 60)
print("  BBC NEWS CLUSTERING PIPELINE")
print("  Azure Databricks + NLP + MLflow")
print("=" * 60)

passed, failed = 0, 0
for name, script in steps:
    print(f"\n▶ Running {name}...")
    t = time.time()
    result = subprocess.run([sys.executable, script],
                            capture_output=True, text=True)
    elapsed = time.time() - t
    if result.returncode == 0:
        print(f"  ✅ Done ({elapsed:.1f}s)")
        passed += 1
    else:
        print(f"  ❌ Failed")
        print(result.stderr[-300:])
        failed += 1

print("\n" + "=" * 60)
print(f"✅ Passed: {passed}/{len(steps)}")
if failed: print(f"❌ Failed: {failed}/{len(steps)}")
print("   Outputs  → outputs/")
print("   Models   → models/")
print("   MLflow   → run 'mlflow ui'")
print("=" * 60)
