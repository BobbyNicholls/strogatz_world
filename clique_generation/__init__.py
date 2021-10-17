from itertools import combinations

import networkx as nx
import numpy as np

from node_embedding import make_node2vec_node_embeddings


def append_vector_to_embeddings(embeddings, new_vector=np.array([1, 1])):
    from gensim.models.keyedvectors import Word2VecKeyedVectors

    keys = embeddings.index_to_key
    node_emb_new = Word2VecKeyedVectors(len(embeddings[keys[0]]) + len(new_vector))
    new_vectors = [
        np.concatenate((embeddings[key], new_vector))
        for key in keys
    ]
    node_emb_new.add_vectors(keys, new_vectors)
    return node_emb_new


def make_embedded_cliques(G, nr_of_cliques=4, min_clique_size=3, max_clique_size=6):
    G_copy = G.copy()

    ba_graph_node_embeddings = make_node2vec_node_embeddings(G_copy)
    clique_seeds = np.random.choice(G_copy.nodes(), nr_of_cliques, replace=False)
    for clique_seed in clique_seeds:
        clique_size = np.random.choice(range(min_clique_size, max_clique_size + 1))
        clique_nodes = [
            int(x[0])
            for x in ba_graph_node_embeddings.most_similar(
                clique_seed, topn=clique_size
            )
        ] + [clique_seed]

        G_copy.add_edges_from(combinations(clique_nodes, 2))

    return G_copy


if __name__ == "__main__":
    ba_graph = nx.extended_barabasi_albert_graph(100, 1, 0.01, 0)

    ba_graph, node_embeddings = make_embedded_cliques(
        ba_graph,
        nr_of_cliques=7,
        min_clique_size=3,
        max_clique_size=6,
    )

    new_embeddings = append_vector_to_embeddings(node_embeddings, new_vector=np.array([4, 3]))

    new_embeddings.most_similar('1')
