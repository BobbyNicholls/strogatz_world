import numpy as np
import yaml

with open("configs/world_features.yaml", "r") as stream:
    world_features = yaml.safe_load(stream)


def get_random_node_features():
    return {
        "race": np.random.choice(world_features["races"]),
        "gender": np.random.choice(world_features["genders"]),
        "faction": np.random.choice(world_features["factions"]),
    }
