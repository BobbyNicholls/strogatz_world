"""
TODO: create multiple settlements
TODO: Use graph embeddings to compare separate settlements
TODO: create aversions between races
TODO: make probability distributions associated with characterics, links, and beliefs
TODO: add migration? conflict?
This main makes a settlement, initialises leaders and followers, propagaes their characteristics, creates beliefs,
creates cliques, initialises and propagates beliefs.
"""

import matplotlib.pyplot as plt
import networkx as nx
import netwulf as nw
import pandas as pd

from belief_propagation.bp_around_graph import(
    initialise_and_propagate_beliefs,
    propagate_beliefs,
    get_belief_string,
)
from clique_generation import make_embedded_cliques
from utils import get_random_node_features, get_belief_dataframe

NR_OF_FACTIONS = 10
POPULATION = 800
NR_OF_CLIQUES = 60
MIN_CLIQUE_SIZE = 3
MAX_CLIQUE_SIZE = 9
BELIEF_PROP_ITERATIONS = 5
JOIN_ON_BELIEFS = True


def draw_graph(G, pos_nodes, node_names={}, node_size=150, plot_weight=False, colour_nodes_on_belief=True):

    try:
        iteration = max(G.nodes[list(G.nodes())[0]]["entity"].beliefs.keys())
        labels = {
            node: G.nodes[node]["race"]
            + " "
            + G.nodes[node]["faction"]
            # + " "
            # + get_belief_string(G.nodes[node]["entity"].beliefs[iteration])
            for node in G.nodes()
        }
    except KeyError:
        labels = {
            node: G.nodes[node]["race"] + " " + G.nodes[node]["faction"]
            for node in G.nodes()
        }
    checked_values = set()
    for key, value in labels.items():
        if value not in checked_values:
            checked_values.add(value)
        else:
            labels[key] = ""
    colour_keys = list(set(labels.values()))
    col_dict = {colour_keys[i]: i for i in range(len(colour_keys))}
    try:
        if colour_nodes_on_belief:
            colours = [tuple(list(G.nodes[node]["entity"].beliefs[iteration][0]) + list(
                G.nodes[node]["entity"].beliefs[iteration][1][:1])) for node in G.nodes()]
        else:
            colours = [
                col_dict[
                    G.nodes[node]["race"]
                    + " "
                    + G.nodes[node]["faction"]
                    + " "
                    + get_belief_string(G.nodes[node]["entity"].beliefs[iteration])
                ]
                for node in G.nodes()
            ]
    except KeyError:
        colours = [
            col_dict[G.nodes[node]["race"] + " " + G.nodes[node]["faction"]]
            for node in G.nodes()
        ]

    nx.draw(
        G,
        pos_nodes,
        node_color=colours,
        with_labels=False,
        node_size=node_size,
        edge_color="gray",
        arrowsize=30,
    )

    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.03)

    nx.draw_networkx_labels(
        G,
        pos_attrs,
        labels=labels,
        font_family="serif",
        font_size=7,
    )
    # nx.draw_networkx()
    if plot_weight:
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)

        nx.draw_networkx_labels(G, pos_attrs, font_family="serif", font_size=7)

    plt.axis("off")
    axis = plt.gca()
    # axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    # axis.set_ylim([1.2 * y for y in axis.get_ylim()])
    # plt.figure(figsize=(10,10))
    plt.savefig(
        f'plots/g{str(pd.to_datetime("now")).replace(":", "-").replace(".", "-")}.png'
    )
    plt.close()


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


def get_leader_node_attributes(leader_nodes):
    return {leader_node: get_random_node_features() for leader_node in leader_nodes}


def propagate_leader_node_attributes(ba_graph, leaders):
    node_attribute_assignment_dict = {}
    egos = set(leaders)
    assigned_nodes = egos.copy()
    while len(egos) > 0:
        next_iteration_egos = set()
        for ego in egos:
            ego_attributes = ba_graph.nodes[ego]
            followers = set(ba_graph.neighbors(ego)).difference(assigned_nodes)
            next_iteration_egos = next_iteration_egos.union(followers)
            node_attribute_assignment_dict.update(
                {follower: ego_attributes for follower in followers}
            )
            assigned_nodes = assigned_nodes.union(followers)
        egos = next_iteration_egos.copy()
    nx.set_node_attributes(ba_graph, node_attribute_assignment_dict)
    return ba_graph


def get_settlement(
    nr_of_factions,
    population,
    nr_of_cliques,
    min_clique_size,
    max_clique_size,
    belief_prop_iterations,
    join_on_beliefs,
):
    # this one has a 'preferential attachment schema' which means when we add new nodes they are more likely to be
    # attached to already central nodes. This results in "influencer nodes" which become increasingly central as a
    # result of their centrality, exhibiting a "power-law distribution" for connectivity between nodes that more
    # accurately represents reality in social networks
    ba_graph = nx.extended_barabasi_albert_graph(population, 1, 0.01, 0)
    # nw.visualize(ba_graph)
    leaders = get_leader_nodes(ba_graph, leader_number=nr_of_factions)
    leader_node_attributes = get_leader_node_attributes(leaders)
    nx.set_node_attributes(ba_graph, leader_node_attributes)
    ba_graph = propagate_leader_node_attributes(ba_graph, leaders)
    pos = nx.kamada_kawai_layout(ba_graph)
    plt.rcParams["figure.figsize"] = (10, 10)
    draw_graph(ba_graph, pos)

    for node in ba_graph.nodes():
        attributes = ba_graph.nodes[node]
        attributes["group"] = attributes["race"]
        nx.set_node_attributes(ba_graph, {node: attributes})

    # nw.visualize(ba_graph)

    ba_graph = initialise_and_propagate_beliefs(
        ba_graph,
        leaders,
        visualise_at_end=False,
        belief_prop_iterations=0,
    )

    draw_graph(ba_graph, pos)

    ba_graph = make_embedded_cliques(
        ba_graph,
        nr_of_cliques=nr_of_cliques,
        min_clique_size=min_clique_size,
        max_clique_size=max_clique_size,
        join_on_beliefs=join_on_beliefs,
    )
    draw_graph(ba_graph, pos)
    pos = nx.kamada_kawai_layout(ba_graph)
    draw_graph(ba_graph, pos)

    # nw.visualize(ba_graph)
    for _ in range(belief_prop_iterations):
        ba_graph = propagate_beliefs(
            ba_graph,
            leaders,
            belief_prop_iterations=1,
        )
        draw_graph(ba_graph, pos)

    # draw_graph(ba_graph, pos)

    iteration = max(ba_graph.nodes[list(ba_graph.nodes())[0]]["entity"].beliefs.keys())
    ba_graph = nx.relabel_nodes(
        ba_graph,
        {
            node: f"{ba_graph.nodes[node]['race']}: {ba_graph.nodes[node]['faction']} {node} "
            f"{[round(y, 2) for x in ba_graph.nodes[node]['entity'].beliefs[iteration] for y in x]}"
            for node in ba_graph.nodes()
        },
    )

    belief_df = get_belief_dataframe(ba_graph)
    # print(belief_df.head(30))

    for node in ba_graph.nodes():
        ba_graph.nodes[node]["entity"] = None

    return ba_graph


# this graph results in random associations and therefore isnt as good at accurately modelling a society
# G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
# draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=200, plot_weight=True)

if __name__ == "__main__":
    ba_graph = get_settlement(
        NR_OF_FACTIONS,
        POPULATION,
        NR_OF_CLIQUES,
        MIN_CLIQUE_SIZE,
        MAX_CLIQUE_SIZE,
        BELIEF_PROP_ITERATIONS,
        JOIN_ON_BELIEFS,
    )
    nw.visualize(ba_graph)
    print("something")
