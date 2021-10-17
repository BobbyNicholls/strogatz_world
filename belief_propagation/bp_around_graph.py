from belief_propagation.bp_entity import Entity
from utils import get_random_node_features

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import netwulf as nw
import pandas as pd

LEADER_NUMBER = 7
POPULATION = 50
VISUALISE_AT_END = True
BELIEF_PROP_ITERATIONS = 5


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


def iterate_beliefs_from_leaders_outwards(F, leaders, iteration):
    F_copy = F.copy()
    leaders_copy = leaders.copy()
    egos = set(leaders_copy)
    propagated_nodes = egos.copy()
    while len(egos) > 0:
        next_iteration_egos = set()
        for ego in egos:
            followers = set(F_copy.neighbors(ego)).difference(propagated_nodes)
            next_iteration_egos = next_iteration_egos.union(followers)
            for follower in followers:
                F_copy.nodes[follower]["entity"].propagate_belief(
                    influencer=F_copy.nodes[ego]["entity"], ITERATION=iteration
                )
            propagated_nodes = propagated_nodes.union(followers)
        egos = next_iteration_egos.copy()
    for leader in leaders_copy:
        F_copy.nodes[leader]["entity"].iterate_belief(iteration)
    iteration += 1
    return F_copy, iteration


def get_belief_string(beliefs):
    belief_string = ""
    for i in range(2):
        for j in range(2):
            belief_string += str(round(beliefs[i, j], 2)) + ", "
    return belief_string[:-2]


def initialise_and_propagate_beliefs(
    G, leaders, belief_prop_iterations, visualise_at_end=False
):
    G_copy = G.copy()
    iteration = 1
    print(belief_prop_iterations)
    leader_entities = {
        leader: {
            "entity": Entity(
                feature_vector=G_copy.nodes[leader],
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
    nx.set_node_attributes(G_copy, leader_entities)

    leader_races = set([G_copy.nodes[leader]["race"] for leader in leaders])
    avaialble_beliefs = {
        race: np.array(
            [
                [np.random.randint(0, 50), np.random.randint(0, 50)],
                [np.random.randint(0, 50), np.random.randint(0, 50)],
            ]
        )
        for race in leader_races
    }

    print("RACIAL BELIEF STRUCTURES:")
    [
        print(x + ": " + get_belief_string(avaialble_beliefs[x]))
        for x in avaialble_beliefs.keys()
    ]

    followers = set(G_copy.nodes()).difference(set(leaders))
    follower_entities = {
        follower: {
            "entity": Entity(
                feature_vector=G_copy.nodes[follower],
                beliefs=avaialble_beliefs[G_copy.nodes[follower]["race"]],
            )
        }
        for follower in followers
    }

    nx.set_node_attributes(G_copy, follower_entities)
    for _ in range(belief_prop_iterations):
        G_copy, iteration = iterate_beliefs_from_leaders_outwards(
            G_copy, leaders, iteration
        )

    for node in G_copy.nodes():
        try:
            print(G_copy.nodes[node]["race"])
            print("Belief progression:")
            for key in G_copy.nodes[node]["entity"].beliefs.keys():
                print(G_copy.nodes[node]["entity"].beliefs[key])
        except KeyError:
            print("BUGGED")

    if visualise_at_end:

        follower_mapping = {
            node: f"{node}: {G_copy.nodes[node]['race']} {G_copy.nodes[node]['faction']} "
            f"{get_belief_string(G_copy.nodes[node]['entity'].beliefs[belief_prop_iterations])}"
            for node in set(G_copy.nodes()).difference(set(leaders))
        }

        leader_mapping = {
            node: f"LEADER{node}: {G_copy.nodes[node]['race']} {G_copy.nodes[node]['faction']} "
            f"{get_belief_string(G_copy.nodes[node]['entity'].beliefs[belief_prop_iterations])}"
            for node in leaders
        }

        G_copy = nx.relabel_nodes(
            G_copy,
            {**follower_mapping, **leader_mapping},
        )

        for node in G_copy.nodes():
            attributes = G_copy.nodes[node]
            attributes["group"] = attributes["race"] + attributes["faction"]
            attributes["entity"] = None
            nx.set_node_attributes(G_copy, {node: attributes})

        nw.visualize(G_copy)

    return G_copy


def propagate_beliefs(G, leaders, belief_prop_iterations):
    G_copy = G.copy()
    iteration = max(G_copy.nodes[list(G_copy.nodes())[0]]['entity'].beliefs.keys()) + 1
    for _ in range(belief_prop_iterations):
        G_copy, iteration = iterate_beliefs_from_leaders_outwards(
            G_copy, leaders, iteration
        )
    return G_copy


if __name__ == "__main__":
    ITERATION = 1
    # this graph results in random associations and therefore isnt as good at accurately modelling a society
    # G = nx.watts_strogatz_graph(n=20, k=5, p=0.2)
    # ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0, 0)
    ba_graph = nx.extended_barabasi_albert_graph(POPULATION, 1, 0.02, 0)

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
        ba_graph, ITERATION = iterate_beliefs_from_leaders_outwards(ba_graph, ITERATION)

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
