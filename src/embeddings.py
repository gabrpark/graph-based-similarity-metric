from node2vec import Node2Vec
import numpy as np


def graph_to_vec(graph):
    # Initialize Node2Vec model with the graph
    node2vec_model = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

    # Fit Node2Vec model (generate walks and train)
    model = node2vec_model.fit(window=10, min_count=1, batch_words=4)

    # Aggregate embeddings to generate a single sentence vector
    embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])
    sentence_embedding = np.mean(embeddings, axis=0)
    return sentence_embedding