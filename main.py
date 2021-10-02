import matplotlib.pyplot as plt
import networkx as nx
import netwulf as nw
import pandas as pd


def draw_graph(G, pos_nodes, node_names={}, node_size=50, plot_weight=False):
    nx.draw(
        G,
        pos_nodes,
        with_labels=False,
        node_size=node_size,
        edge_color="gray",
        arrowsize=30,
    )

    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)

    nx.draw_networkx_labels(G, pos_attrs, font_family="serif", font_size=20)

    if plot_weight:
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)

        nx.draw_networkx_labels(G, pos_attrs, font_family="serif", font_size=20)

    plt.axis("off")
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])


def get_leader_nodes(G, leader_number=3):
    """
    get top X leaders by degree centrality
    """
    # TODO: find way to determine number of leaders dynamically
    return (
        pd.DataFrame(
            nx.centrality.degree_centrality(G).items(),
            columns=["node", "centrality"],
        )
        .sort_values("centrality", ascending=False)["node"]
        .iloc[:leader_number]
        .values
    )


# this graph results in random associations and therefore isnt as good at accurately modelling a society
G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=200, plot_weight=True)

# this one has a 'preferential attachment schema' which means when we add new nodes they are more likely to be attached
# to already central nodes. This results in "influencer nodes" which become increasingly central as a result of their
# centrality, exhibiting a "power-law distribution" for connectivity between nodes that more accurately represents
# reality in social networks
ba_graph = nx.extended_barabasi_albert_graph(50, 1, 0, 0)
draw_graph(
    ba_graph, pos_nodes=nx.shell_layout(ba_graph), node_size=200, plot_weight=True
)

# nw.visualize(ba_graph)

leaders = get_leader_nodes(ba_graph, leader_number=3)

import yaml

with open("configs/world_features.yaml", 'r') as stream:
    world_features = yaml.safe_load(stream)
