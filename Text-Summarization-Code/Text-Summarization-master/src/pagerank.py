import time
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

start_time = time.time()
def preprocess_text(text):
    """Tokenizes and preprocesses text by removing stopwords."""
    stop_words = set(nltk.corpus.stopwords.words('english'))
    sentences = nltk.sent_tokenize(text)
    clean_sentences = [
        ' '.join([word for word in nltk.word_tokenize(sentence.lower())
                  if word.isalnum() and word not in stop_words])
        for sentence in sentences
    ]
    return sentences, clean_sentences

def build_similarity_matrix(sentences):
    """Creates a similarity matrix for the sentences."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def rank_sentences(similarity_matrix, sentences, top_n=3):
    """Ranks sentences using the PageRank algorithm."""
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )
    summary = [ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))]
    return summary

# Example Usage
if __name__ == "__main__":
    text = """
    Natural Language Processing (NLP) is a sub-field of artificial intelligence (AI).
    It focuses on the interaction between computers and human language.
    The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages.
    Applications of NLP include chatbots, sentiment analysis, and language translation.
    Recent advancements in NLP are largely driven by deep learning and large language models like GPT.
    """

    # Step 1: Preprocess the text
    original_sentences, clean_sentences = preprocess_text(text)

    # Step 2: Build the similarity matrix
    similarity_matrix = build_similarity_matrix(clean_sentences)

    # Step 3: Rank sentences and generate summary
    summary = rank_sentences(similarity_matrix, original_sentences, top_n=3)

    print("Original Text:")
    print(text)
    print("\nSummary:")
    for i, sentence in enumerate(summary, 1):
        print(f"{i}. {sentence}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")