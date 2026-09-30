# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# BBC News Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs("outputs/eda_plots", exist_ok=True)

df = pd.read_csv("data/bbc_news.csv", encoding='latin-1')
df.rename(columns={'type': 'category', 'news': 'text'}, inplace=True)

print("=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 1. Basic Info ─────────────────────────────────────────────
print(f"\n📌 Shape         : {df.shape}")
print(f"📌 Columns       : {df.columns.tolist()}")
print(f"📌 Null Values   :\n{df.isnull().sum()}")
print(f"📌 Duplicates    : {df.duplicated().sum()}")

# ── 2. Category Distribution ──────────────────────────────────
print("\n📌 Category Distribution:")
print(df['category'].value_counts())
print("\n📌 Category %:")
print((df['category'].value_counts(normalize=True)*100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed']

df['category'].value_counts().plot(kind='bar', ax=axes[0],
    color=colors, edgecolor='black')
axes[0].set_title('Article Count Per Category', fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=30)

df['category'].value_counts().plot(kind='pie', ax=axes[1],
    colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Category Distribution (%)', fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('outputs/eda_plots/01_category_distribution.png', dpi=150)
plt.close()
print("✅ Saved: 01_category_distribution.png")

# ── 3. Text Length Analysis ───────────────────────────────────
df['word_count']  = df['text'].apply(lambda x: len(str(x).split()))
df['char_count']  = df['text'].apply(lambda x: len(str(x)))
df['sent_count']  = df['text'].apply(lambda x: len(str(x).split('.')))

print("\n📌 Text Length Stats:")
print(df[['word_count','char_count','sent_count']].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df['word_count'].hist(ax=axes[0], bins=40, color='#2563eb', edgecolor='black')
axes[0].set_title('Word Count Distribution', fontweight='bold')
axes[0].set_xlabel('Words per Article')

df['char_count'].hist(ax=axes[1], bins=40, color='#16a34a', edgecolor='black')
axes[1].set_title('Character Count Distribution', fontweight='bold')
axes[1].set_xlabel('Characters per Article')

df['sent_count'].hist(ax=axes[2], bins=40, color='#dc2626', edgecolor='black')
axes[2].set_title('Sentence Count Distribution', fontweight='bold')
axes[2].set_xlabel('Sentences per Article')

plt.tight_layout()
plt.savefig('outputs/eda_plots/02_text_length_distribution.png', dpi=150)
plt.close()
print("✅ Saved: 02_text_length_distribution.png")

# ── 4. Word Count By Category ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, cat in enumerate(df['category'].unique()):
    subset = df[df['category'] == cat]['word_count']
    ax.hist(subset, bins=30, alpha=0.6, label=cat, color=colors[i])
ax.set_title('Word Count Distribution by Category', fontweight='bold')
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eda_plots/03_wordcount_by_category.png', dpi=150)
plt.close()
print("✅ Saved: 03_wordcount_by_category.png")

# ── 5. WordCloud Per Category ─────────────────────────────────
os.makedirs("outputs/wordclouds", exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, cat in enumerate(df['category'].unique()):
    text = ' '.join(df[df['category'] == cat]['text'].astype(str))
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap='Blues', max_words=100).generate(text)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(f'WordCloud: {cat.upper()}', fontweight='bold', fontsize=14)
    axes[i].axis('off')

axes[5].axis('off')
plt.suptitle('Word Clouds by News Category', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/wordclouds/all_category_wordclouds.png', dpi=150)
plt.close()
print("✅ Saved: all_category_wordclouds.png")

# ── 6. Avg Word Count Per Category ───────────────────────────
avg_wc = df.groupby('category')['word_count'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
avg_wc.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
ax.set_title('Average Word Count Per Category', fontweight='bold')
ax.set_ylabel('Avg Word Count')
ax.tick_params(rotation=30)
plt.tight_layout()
plt.savefig('outputs/eda_plots/04_avg_wordcount_category.png', dpi=150)
plt.close()
print("✅ Saved: 04_avg_wordcount_category.png")

print("\n✅ EDA Complete!")
df.to_csv("outputs/df_with_stats.csv", index=False)
