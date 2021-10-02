import matplotlib.pyplot as plt
import networkx as nx
import netwulf as nw
import numpy as np
import pandas as pd
import yaml


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


def get_random_node_features():
    return {
        "race": np.random.choice(world_features["races"]),
        "gender": np.random.choice(world_features["genders"]),
        "faction": np.random.choice(world_features["factions"]),
    }


# this graph results in random associations and therefore isnt as good at accurately modelling a society
# G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
# draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=200, plot_weight=True)

# this one has a 'preferential attachment schema' which means when we add new nodes they are more likely to be attached
# to already central nodes. This results in "influencer nodes" which become increasingly central as a result of their
# centrality, exhibiting a "power-law distribution" for connectivity between nodes that more accurately represents
# reality in social networks
ba_graph = nx.extended_barabasi_albert_graph(9999, 1, 0, 0)
# draw_graph(
#     ba_graph, pos_nodes=nx.shell_layout(ba_graph), node_size=200, plot_weight=True
# )

# nw.visualize(ba_graph)

with open("configs/world_features.yaml", "r") as stream:
    world_features = yaml.safe_load(stream)


leaders = get_leader_nodes(ba_graph, leader_number=5)
leader_node_attributes = {
    node: features
    for node, features in zip(
        leaders, [get_random_node_features() for _ in range(len(leaders))]
    )
}
nx.set_node_attributes(ba_graph, leader_node_attributes)

egos = set(leaders)
for ego in egos:
    ego_attributes = ba_graph.nodes[ego]
    followers = list(
        nx.ego_graph(ba_graph, ego, radius=1, center=True, undirected=True).nodes()
    )
    for follower in followers:
        if len(ba_graph.nodes[follower]) == 0:
            nx.set_node_attributes(ba_graph, {follower: ego_attributes})

while len(egos) < len(ba_graph):
    # TODO: make this node querying more efficient
    new_egos = set(
        [node for node in ba_graph.nodes if len(ba_graph.nodes[node]) != 0]
    ).difference(egos)
    for ego in new_egos:
        ego_attributes = ba_graph.nodes[ego]
        followers = (
            set(
                nx.ego_graph(
                    ba_graph, ego, radius=1, center=True, undirected=True
                ).nodes()
            )
            .difference(new_egos)
            .difference(egos)
        )
        for follower in followers:
            if len(ba_graph.nodes[follower]) == 0:
                nx.set_node_attributes(ba_graph, {follower: ego_attributes})
        egos.add(ego)

# nx.write_gexf(ba_graph, "communities.gexf")

for node in ba_graph.nodes():
    attributes = ba_graph.nodes[node]
    attributes["group"] = attributes["race"]
    nx.set_node_attributes(ba_graph, {node: attributes})

ba_graph = nx.relabel_nodes(
    ba_graph,
    {
        node: f"{ba_graph.nodes[node]['race']}: {ba_graph.nodes[node]['faction']} {node}"
        for node in ba_graph.nodes()
    },
)
nw.visualize(ba_graph)
