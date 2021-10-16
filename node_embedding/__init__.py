"""
Node2Vec keyedvectors object documentation (for the returned embeddings object):
https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors
"""

import itertools

# from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from belief_propagation.bp_around_graph import get_leader_nodes
from main import draw_graph


def make_node2vec_node_embeddings(G):
    G_copy = G.copy()
    node2vec = Node2Vec(G_copy, dimensions=2)
    model = node2vec.fit(window=10)
    embeddings = model.wv
    return embeddings


def make_deep_walk_node_embeddings(G):
    G_copy = G.copy()
    dw = DeepWalk(dimensions=2)
    dw.fit(G_copy)
    embeddings = dw.get_embedding()
    return embeddings


if __name__ == "__main__":
    POPULATION = 30
    NR_OF_CLIQUES = 4
    MIN_CLIQUE_SIZE = 3
    MAX_CLIQUE_SIZE = 6
    ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0.02, 0)
    draw_graph(
        ba_graph, pos_nodes=nx.spring_layout(ba_graph), node_size=200, plot_weight=True
    )
    #get_leader_nodes(ba_graph, 2)
    ba_graph_node_embeddings = make_node2vec_node_embeddings(ba_graph)
    # ba_graph_node_embeddings.similarity("0", "17")
    # ba_graph_node_embeddings.similarity("0", "23")
    # node_embedding_dict = {
    #     node: ba_graph_node_embeddings[node] for node in ba_graph.nodes()
    # }
    clique_seeds = np.random.choice(ba_graph.nodes(), NR_OF_CLIQUES, replace=False)
    for clique_seed in clique_seeds:
        clique_size = np.random.choice(range(MIN_CLIQUE_SIZE, MAX_CLIQUE_SIZE + 1))
        clique_nodes = [
            x[0]
            for x in ba_graph_node_embeddings.most_similar(
                str(clique_seed), topn=clique_size
            )
        ] + [clique_seed]

        ba_graph.add_edges_from(itertools.combinations(clique_nodes, 2))

    draw_graph(
        ba_graph, pos_nodes=nx.spring_layout(ba_graph), node_size=200, plot_weight=True
    )
