import networkx as nx
# from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sentence_1 = "Climate change is one of the most significant challenges facing humanity."
sentence_2 = "Climate change poses a great obstacle for humanity and is among the most crucial issues we confront."

# AMR data
amr_1 = {
    "significant-02": {
        "degree": "most",
        "domain": {
            "challenge": {
                "mod": {
                    "face-01": {
                        "arg0": "humanity"
                    }
                },
                "consist-of": {
                    "change-01": {
                        "topic": "weather",
                        "mod": "climate"
                    }
                }
            }
        }
    }
}

amr_2 = {
    'pose-01': {
        'ARG1': {
            'change-01': {
                'ARG1': 'climate'
            }
        },
        'ARG2': {
            'obstacle-01': {
                'ARG1': 'humanity',
                'mod': 'great-02'
            }
        },
        'frequency': {
            'confront-01': {
                'ARG0': 'we',
                'ARG1': {
                    'issue-01': {
                        'ARG1': {
                            'and': ['climate', 'change-01']
                        },
                        'mod': 'crucial-02'
                    }
                }
            }
        }
    }
}


# Function to convert AMR to a graph
# def amr_to_graph(amr, graph=None, parent=None):
#     if graph is None:
#         graph = nx.Graph()
#     for key, value in amr.items():
#         if parent:
#             graph.add_edge(parent, key)
#         if isinstance(value, dict):
#             amr_to_graph(value, graph, key)
#         else:
#             graph.add_node(value)
#             graph.add_edge(key, value)
#     return graph


def build_amr_graph(amr_dict, G=None):

    if G is None:
        G = nx.Graph()

    def add_edges(graph, parent, child_dict):
        for key, value in child_dict.items():
            if isinstance(value, dict):
                # Add edge between parent and key, then recurse
                graph.add_edge(parent, key)
                add_edges(graph, key, value)
            elif isinstance(value, list):
                # Add edges to all items in the list
                for item in value:
                    graph.add_edge(parent, item)
            else:
                # Add edge to a single child value
                graph.add_edge(parent, value)

    # Start building the graph from the top-level keys
    for key in amr_dict:
        G.add_node(key)
        add_edges(G, key, amr_dict[key])

    return G

# Convert AMR to graph
graph1 = build_amr_graph(amr_1)
graph2 = build_amr_graph(amr_2)


# Visualize the graph
plt.figure(figsize=(10, 7))
nx.draw(graph1, with_labels=True, font_weight='bold', node_color='skyblue', node_size=2000)
plt.title("AMR Graph")
plt.show()

def apply_node2vec(graph):
    node2vec = Node2Vec(graph, dimensions=20, walk_length=16, num_walks=100, workers=2)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model


# Apply node2vec
model1 = apply_node2vec(graph1)
model2 = apply_node2vec(graph2)

# Get node embeddings
node_embeddings1 = model1.wv.vectors
node_embeddings2 = model2.wv.vectors

# Compute cosine similarity between node embeddings
cosine_sim = cosine_similarity(node_embeddings1, node_embeddings2)
print(cosine_sim)
cosine_sim = np.sum(cosine_sim.flatten())
print(cosine_sim)

