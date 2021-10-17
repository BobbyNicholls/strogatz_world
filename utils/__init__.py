import numpy as np
import pandas as pd
import yaml

with open("configs/world_features.yaml", "r") as stream:
    world_features = yaml.safe_load(stream)


def get_random_node_features():
    return {
        "race": np.random.choice(world_features["races"]),
        "gender": np.random.choice(world_features["genders"]),
        "faction": np.random.choice(world_features["factions"]),
    }


def get_belief_dataframe(G):
    G_copy = G.copy()
    nodes = list(G_copy.nodes())
    df = pd.DataFrame(
        data=(
            (a[0][0], a[0][1], a[1][0], a[1][1], G_copy.nodes[nodes[0]]["race"])
            for a in G_copy.nodes[nodes[0]]["entity"].beliefs.values()
        ),
        columns=["a", "b", "c", "d", "race"],
    )
    for node in nodes[1:]:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    data=(
                        (
                            a[0][0],
                            a[0][1],
                            a[1][0],
                            a[1][1],
                            G_copy.nodes[node]["race"],
                        )
                        for a in G_copy.nodes[node]["entity"].beliefs.values()
                    ),
                    columns=["a", "b", "c", "d", "race"],
                ),
            ],
            axis=0,
        )

    return df
