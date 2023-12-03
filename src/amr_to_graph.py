import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import graph_to_vec
from visualizers import visualize_graph

sentence_1 = "Climate change is one of the most significant challenges facing humanity."
sentence_2 = "Climate change poses a great obstacle for humanity and is among the most crucial issues we confront."

# AMR Parsing
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


# Convert AMR to graph
def process_node(graph, key, value, parent=None):
    # Add the current node
    graph.add_node(key)

    # Add an edge from the parent node if it exists
    if parent is not None:
        graph.add_edge(parent, key)

    # If the value is a dictionary, recursively process each item
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            process_node(graph, sub_key, sub_value, key)
    # If the value is a list, recursively process each element
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for sub_key, sub_value in item.items():
                    process_node(graph, sub_key, sub_value, key)
    # If the value is a basic type (e.g., int, str), add it as a node
    else:
        graph.add_node(value)
        graph.add_edge(key, value)


def amr_to_graph(amr):
    graph = nx.DiGraph()
    for key, value in amr.items():
        process_node(graph, key, value)
    return graph


# Convert AMR to graph
graph_1 = amr_to_graph(amr_1)
graph_2 = amr_to_graph(amr_2)

# Apply Node2Vec to a graph get embeddings for each graph
embedding1 = graph_to_vec(graph_1)
embedding2 = graph_to_vec(graph_2)

# Calculate cosine similarity to compare the two graphs' embeddings
similarity = cosine_similarity([embedding1], [embedding2])[0][0]
print(f"Similarity: {similarity}")

# Visualize the graph_1
visualize_graph(graph_1)

# Visualize the graph_2
visualize_graph(graph_2)
