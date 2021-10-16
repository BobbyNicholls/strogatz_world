from belief_propagation.bp_entity import Entity
from utils import get_random_node_features

import networkx as nx
import numpy as np
import netwulf as nw
import pandas as pd

from main import draw_graph

LEADER_NUMBER = 7
POPULATION = 50
ITERATION = 1
VISUALISE_AT_END = True
BELIEF_PROP_ITERATIONS = 5


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
    # TODO: if someone is a direct follower of 2 different leaders their assignment should be random
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


def get_belief_string(beliefs):
    belief_string = ""
    for i in range(2):
        for j in range(2):
            belief_string += str(round(beliefs[i, j], 2)) + ", "
    return belief_string[:-2]


if __name__ == "__main__":
    # this graph results in random associations and therefore isnt as good at accurately modelling a society
    # G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
    # ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0, 0)
    ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0.02, 0)
    draw_graph(ba_graph, pos_nodes=nx.spring_layout(ba_graph), node_size=200, plot_weight=True)

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

    print("\nLEADER BELIEF STRUCTURES:")
    [
        print(
            leader_entities[x]["entity"].feature_vector["race"]
            + ": "
            + get_belief_string(leader_entities[x]["entity"].beliefs[0])
        )
        for x in leader_entities.keys()
    ]
    print("\n==============================================\n")
    nx.set_node_attributes(ba_graph, leader_entities)

    leader_races = [ba_graph.nodes[leader]["race"] for leader in leaders]
    avaialble_beliefs = {
        leader_races[0]: np.array([[0.7, 0.2], [0.06, 0.04]]),
        leader_races[1]: np.array([[0.25, 0.25], [0.25, 0.25]]),
        leader_races[2]: np.array([[0.05, 0.007], [0.333, 0.61]]),
        leader_races[3]: np.array([[0.05, 0.007], [0.333, 0.61]]),
        leader_races[4]: np.array([[0.05, 0.007], [0.333, 0.61]]),
        leader_races[5]: np.array([[0.05, 0.007], [0.333, 0.61]]),
        leader_races[6]: np.array([[0.05, 0.007], [0.333, 0.61]]),
    }

    print("RACIAL BELIEF STRUCTURES:")
    [
        print(x + ": " + get_belief_string(avaialble_beliefs[x]))
        for x in avaialble_beliefs.keys()
    ]

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
    for _ in range(BELIEF_PROP_ITERATIONS):
        ITERATION = iterate_beliefs_from_leaders_outwards(ITERATION)

    for node in ba_graph.nodes():
        try:
            print(ba_graph.nodes[node]["race"])
            print("start:")
            print(ba_graph.nodes[node]["entity"].beliefs[0])
            print("end:")
            print(ba_graph.nodes[node]["entity"].beliefs[1])
            print(ba_graph.nodes[node]["entity"].beliefs[2])
            print(ba_graph.nodes[node]["entity"].beliefs[3])
            print(ba_graph.nodes[node]["entity"].beliefs[4])
        except KeyError:
            print("BUGGED")
            continue


    if VISUALISE_AT_END:

        follower_mapping = {
            node: f"{node}: {ba_graph.nodes[node]['race']} {ba_graph.nodes[node]['faction']} "
            f"{get_belief_string(ba_graph.nodes[node]['entity'].beliefs[BELIEF_PROP_ITERATIONS])}"
            for node in set(ba_graph.nodes()).difference(set(leaders))
        }

        leader_mapping = {
            node: f"LEADER{node}: {ba_graph.nodes[node]['race']} {ba_graph.nodes[node]['faction']} "
            f"{get_belief_string(ba_graph.nodes[node]['entity'].beliefs[BELIEF_PROP_ITERATIONS])}"
            for node in leaders
        }

        ba_graph = nx.relabel_nodes(
            ba_graph,
            {**follower_mapping, **leader_mapping},
        )

        for node in ba_graph.nodes():
            attributes = ba_graph.nodes[node]
            attributes["group"] = attributes["race"] + attributes["faction"]
            attributes["entity"] = None
            nx.set_node_attributes(ba_graph, {node: attributes})

        nw.visualize(ba_graph)
