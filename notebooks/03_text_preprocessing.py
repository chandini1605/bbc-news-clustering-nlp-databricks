# ============================================================
# STEP 3: TEXT PREPROCESSING (NLP)
# BBC News Clustering
# ============================================================

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv("outputs/df_with_stats.csv")

print("=" * 60)
print("STEP 3: TEXT PREPROCESSING (NLP Pipeline)")
print("=" * 60)

# ── 1. Show Raw Text Sample ───────────────────────────────────
print("\n📌 Raw Text Sample:")
print(df['text'].iloc[0][:300])

# ── 2. Text Cleaning Function ─────────────────────────────────
stop_words   = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()
stemmer      = PorterStemmer()

def clean_text(text):
    # Step 1: Lowercase
    text = str(text).lower()
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Step 3: Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Step 4: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Step 5: Tokenize
    tokens = word_tokenize(text)
    # Step 6: Remove stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Step 7: Lemmatization (better than stemming - keeps real words)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ── 3. Apply Cleaning ─────────────────────────────────────────
print("\n⏳ Cleaning text... (this may take a moment)")
df['cleaned_text'] = df['text'].apply(clean_text)
print("✅ Text cleaning complete!")

# ── 4. Before vs After Comparison ────────────────────────────
print("\n📌 Before Cleaning:")
print(df['text'].iloc[0][:200])
print("\n📌 After Cleaning:")
print(df['cleaned_text'].iloc[0][:200])

# ── 5. Stats After Cleaning ───────────────────────────────────
df['cleaned_word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

print(f"\n📌 Word Count Before Cleaning: {df['word_count'].mean():.0f} avg")
print(f"📌 Word Count After Cleaning : {df['cleaned_word_count'].mean():.0f} avg")
print(f"📌 Words Removed             : {(df['word_count'] - df['cleaned_word_count']).mean():.0f} avg per article")

# ── 6. Vocabulary Size ────────────────────────────────────────
all_words = ' '.join(df['cleaned_text']).split()
vocab = set(all_words)
print(f"\n📌 Total Words (raw)         : {len(all_words):,}")
print(f"📌 Unique Vocabulary Size    : {len(vocab):,}")

# ── 7. Top Words Per Category ─────────────────────────────────
from collections import Counter
print("\n📌 Top 10 Words Per Category After Cleaning:")
for cat in df['category'].unique():
    words = ' '.join(df[df['category'] == cat]['cleaned_text']).split()
    top = Counter(words).most_common(10)
    print(f"\n   {cat.upper()}: {[w[0] for w in top]}")

# ── 8. Visualize Word Count Before vs After ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['word_count'].hist(ax=axes[0], bins=40, color='#dc2626', edgecolor='black', alpha=0.7)
axes[0].set_title('Word Count BEFORE Cleaning', fontweight='bold')
axes[0].set_xlabel('Word Count')

df['cleaned_word_count'].hist(ax=axes[1], bins=40, color='#16a34a', edgecolor='black', alpha=0.7)
axes[1].set_title('Word Count AFTER Cleaning', fontweight='bold')
axes[1].set_xlabel('Word Count')

plt.suptitle('Text Length Before vs After NLP Preprocessing', fontweight='bold')
plt.tight_layout()
os.makedirs("outputs/eda_plots", exist_ok=True)
plt.savefig('outputs/eda_plots/05_before_after_cleaning.png', dpi=150)
plt.close()
print("\n✅ Saved: 05_before_after_cleaning.png")

# Save cleaned data
df.to_csv("outputs/cleaned_news.csv", index=False)
print("✅ Cleaned data saved to outputs/cleaned_news.csv")
print("\n✅ Step 3 Complete!")
