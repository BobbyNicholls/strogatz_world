from itertools import combinations

import numpy as np

from node_embedding import make_node2vec_node_embeddings


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
