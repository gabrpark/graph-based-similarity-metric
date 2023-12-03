import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(graph):
    plt.figure(figsize=(10, 7))
    nx.draw(graph, with_labels=True, font_weight='bold', node_color='skyblue', node_size=2000)
    plt.title("AMR Graph")
    plt.show()
