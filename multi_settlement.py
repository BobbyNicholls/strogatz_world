"""
TODO: create multiple settlements
TODO: Use graph embeddings to compare separate settlements
TODO: create aversions between races
TODO: make probability distributions associated with characterics, links, and beliefs
TODO: add migration? conflict?
This main makes a settlement, initialises leaders and followers, propagaes their characteristics, creates beliefs,
creates cliques, initialises and propagates beliefs.
"""

from networkx.algorithms.operators.all import compose_all
import netwulf as nw

from main import get_settlement

NR_OF_FACTIONS = 3
POPULATION = 200
NR_OF_CLIQUES = 60
MIN_CLIQUE_SIZE = 3
MAX_CLIQUE_SIZE = 5
BELIEF_PROP_ITERATIONS = 1
JOIN_ON_BELIEFS = True
NR_OF_SETTLEMENTS = 3


# this graph results in random associations and therefore isnt as good at accurately modelling a society
# G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
# draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=200, plot_weight=True)


if __name__ == "__main__":
    graphs = []
    for _ in range(NR_OF_SETTLEMENTS):
        graphs.append(
            get_settlement(
                NR_OF_FACTIONS,
                POPULATION,
                NR_OF_CLIQUES,
                MIN_CLIQUE_SIZE,
                MAX_CLIQUE_SIZE,
                BELIEF_PROP_ITERATIONS,
                JOIN_ON_BELIEFS,
            )
        )
    G = compose_all(graphs)
    nw.visualize(G)
