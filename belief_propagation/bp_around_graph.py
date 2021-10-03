from belief_propagation.bp_entity import Entity
from utils import get_random_node_features

import networkx as nx
import numpy as np
import netwulf as nw
import pandas as pd

LEADER_NUMBER = 3
POPULATION = 100
ITERATION = 1
VISUALISE_AT_END = True


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


def propagate_node_attributes():
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


def iterate_beliefs_from_leaders_outwards(ITERATION):
    egos = set(leaders)
    propagated_nodes = egos.copy()
    while len(egos) > 0:
        next_iteration_egos = set()
        for ego in egos:
            followers = set(ba_graph.neighbors(ego)).difference(propagated_nodes)
            next_iteration_egos = next_iteration_egos.union(followers)
            for follower in followers:
                ba_graph.nodes[follower]["entity"].propagate_belief(
                    influencer=ba_graph.nodes[ego]["entity"], ITERATION=ITERATION
                )
            propagated_nodes = propagated_nodes.union(followers)
        egos = next_iteration_egos.copy()
    for leader in leaders:
        ba_graph.nodes[leader]["entity"].iterate_belief(ITERATION)
    ITERATION += 1
    return ITERATION


# this graph results in random associations and therefore isnt as good at accurately modelling a society
# G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
# draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=200, plot_weight=True)

ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0, 0)

# nw.visualize(ba_graph)

leaders = get_leader_nodes(ba_graph, leader_number=LEADER_NUMBER)
leader_node_attributes = {
    node: features
    for node, features in zip(
        leaders, [get_random_node_features() for _ in range(len(leaders))]
    )
}
nx.set_node_attributes(ba_graph, leader_node_attributes)

start = pd.to_datetime("now")
propagate_node_attributes()
end = pd.to_datetime("now")
print(end - start)

# nw.visualize(ba_graph)

leader_entities = {
    leader: {
        "entity": Entity(
            feature_vector=ba_graph.nodes[leader],
            beliefs=np.array(
                [
                    [np.random.randint(0, 50), np.random.randint(0, 50)],
                    [np.random.randint(0, 50), np.random.randint(0, 50)],
                ]
            ),
        )
    }
    for leader in leaders
}

nx.set_node_attributes(ba_graph, leader_entities)

leader_races = [ba_graph.nodes[leader]['race'] for leader in leaders]
avaialble_beliefs = {
    leader_races[0]: np.array([[70, 20], [5, 4]]),
    leader_races[1]: np.array([[1, 1], [1, 1]]),
    leader_races[2]: np.array([[7, 1], [50, 90]]),
}

followers = set(ba_graph.nodes()).difference(set(leaders))
follower_entities = {
    follower: {
        "entity": Entity(
            feature_vector=ba_graph.nodes[follower],
            beliefs=avaialble_beliefs[ba_graph.nodes[follower]["race"]],
        )
    }
    for follower in followers
}

nx.set_node_attributes(ba_graph, follower_entities)
for _ in range(5):
    ITERATION = iterate_beliefs_from_leaders_outwards(ITERATION)

for node in ba_graph.nodes():
    try:
        print(ba_graph.nodes[node]['race'])
        print("start:")
        print(ba_graph.nodes[node]['entity'].beliefs[0])
        print("end:")
        print(ba_graph.nodes[node]['entity'].beliefs[1])
        print(ba_graph.nodes[node]['entity'].beliefs[2])
        print(ba_graph.nodes[node]['entity'].beliefs[3])
        print(ba_graph.nodes[node]['entity'].beliefs[4])
    except KeyError:
        print("BUGGED")
        continue

if VISUALISE_AT_END:
    for node in ba_graph.nodes():
        attributes = ba_graph.nodes[node]
        attributes["group"] = attributes["race"]
        attributes["entity"] = None
        nx.set_node_attributes(ba_graph, {node: attributes})
    #nw.visualize(ba_graph)

    ba_graph = nx.relabel_nodes(
        ba_graph,
        {
            node: f"{ba_graph.nodes[node]['race']}: {ba_graph.nodes[node]['faction']} {node}"
            for node in ba_graph.nodes()
        },
    )
    nw.visualize(ba_graph)
