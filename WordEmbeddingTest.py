import pandas as pd
from collections import Counter
import jieba
import numpy as np
from gensim.models import Word2Vec

# Read CSV and map satire_presence
df = pd.read_csv("Test1temp03.csv")
df["satire_presence"] = df["satire_presence"].map({"是": 1, "否": 0})

satirical_texts = df[df["satire_presence"] == 1]["text"].tolist()
non_satirical_texts = df[df["satire_presence"] == 0]["text"].tolist()

# Load Chinese stopwords
stopwords = set(open("Customized_stopwords.txt", encoding="utf-8").read().splitlines())



def tokenize_and_clean(texts):
    words = []
    for text in texts:
        if not isinstance(text, str):
            continue  # Skip non-string values
        tokens = jieba.lcut(text)
        words.extend([word for word in tokens if word not in stopwords and len(word) > 1])
    return words

# Process texts
satirical_words = tokenize_and_clean(satirical_texts)
non_satirical_words = tokenize_and_clean(non_satirical_texts)

# Count word frequencies
satirical_freq = Counter(satirical_words)
non_satirical_freq = Counter(non_satirical_words)

# Show top words
print("Top satire words:", satirical_freq.most_common(20))
print("Top non-satire words:", non_satirical_freq.most_common(20))

# Convert frequencies into DataFrames and merge
satirical_df = pd.DataFrame(satirical_freq.items(), columns=["word", "count_satire"])
non_satirical_df = pd.DataFrame(non_satirical_freq.items(), columns=["word", "count_non_satire"])
word_counts = pd.merge(satirical_df, non_satirical_df, on="word", how="outer").fillna(1)
word_counts["log_odds"] = np.log(word_counts["count_satire"] / word_counts["count_non_satire"])
top_satirical_words = word_counts.sort_values("log_odds", ascending=False).head(20)
print(top_satirical_words)

# Train Word2Vec models
satirical_model = Word2Vec([satirical_words], vector_size=50, window=3, min_count=3, workers=4)
non_satirical_model = Word2Vec([non_satirical_words], vector_size=50, window=3, min_count=3, workers=4)

# Save models
satirical_model.save("satire_word2vec.model")
non_satirical_model.save("non_satire_word2vec.model")

# Show similar words for '降级'
print("Satirical context for '降级':", satirical_model.wv.most_similar("降级"))
print("Non-Satirical context for '降级':", non_satirical_model.wv.most_similar("降级"))



