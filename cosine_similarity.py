from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
        print(sentences)
    return [sentence.strip() for sentence in sentences]


def compute_similarity(sentences_pairs, model):
    similarities = []
    for sent1, sent2 in sentences_pairs:
        # Compute embeddings for each sentence
        embeddings = model.encode([sent1, sent2])

        # Compute cosine similarity
        cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        similarities.append((sent1, sent2, cosine_sim))

    return similarities


# Paths to your files
original_text = 'input.txt'
rephrased_text = 'output.txt'

# Read sentences from files
sentences_original = read_sentences_from_file(original_text)
sentences_rephrased = read_sentences_from_file(rephrased_text)

# Combining sentences from both files into a list of tuples
combined_sentences = list(zip(sentences_original, sentences_rephrased))

# Initializing a pre-trained model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute semantic similarities
semantic_similarities = compute_similarity(combined_sentences, model)
print(semantic_similarities[0][2])