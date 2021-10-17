import numpy as np
from utils import get_random_node_features

# left argument "influences" the right argument,
# the first arg determines the row beleif, and the second the column belief
# This means the column of the second argument is "listening" for the values of the row of the first


class Entity:
    def __init__(self, feature_vector, beliefs):
        assert type(beliefs) == np.ndarray, "beliefs must be array"
        self.feature_vector = feature_vector
        self.beliefs = {0: beliefs / beliefs.sum()}

    def propagate_belief(self, influencer, ITERATION):
        """
        my beliefs in iteration t are a product of my t-1 beliefs and the beliefs of my influencer in t-1, representing
        the fact that in t-1 i was influenced by the influencer node and therefore my beliefs are different in t to what
        they were in t-1
        """
        self.beliefs[ITERATION] = np.dot(
            influencer.beliefs[ITERATION - 1], self.beliefs[ITERATION - 1]
        )
        self.beliefs[ITERATION] = (
            self.beliefs[ITERATION] / self.beliefs[ITERATION].sum()
        )

    def iterate_belief(self, ITERATION):
        self.beliefs[ITERATION] = self.beliefs[ITERATION - 1].copy()


if __name__ == "__main__":
    ITERATION = 1
    e1 = Entity(
        feature_vector=get_random_node_features(), beliefs=np.array([[10, 15], [5, 4]])
    )
    e2 = Entity(
        feature_vector=get_random_node_features(), beliefs=np.array([[1, 1], [59, 72]])
    )
    e3 = Entity(
        feature_vector=get_random_node_features(), beliefs=np.array([[2, 1], [50, 55]])
    )
    while ITERATION < 100:
        e1.propagate_belief(influencer=e3, ITERATION=ITERATION)
        e2.propagate_belief(influencer=e3, ITERATION=ITERATION)
        e3.iterate_belief(ITERATION)
        ITERATION += 1

    first_iteration = min(e1.beliefs.keys())
    last_iteration = max(e1.beliefs.keys())
    print(f"First iteration: {e1.beliefs[first_iteration]}")
    print(
        f"First iteration: {e1.beliefs[first_iteration][0].sum() / e1.beliefs[first_iteration][1].sum()}"
    )
    print(f"Last iteration: {e1.beliefs[last_iteration]}")
    print(
        f"Last iteration: {e1.beliefs[last_iteration][0].sum() / e1.beliefs[last_iteration][1].sum()}"
    )
    print(f"First iteration: {e2.beliefs[first_iteration]}")
    print(
        f"First iteration: {e2.beliefs[first_iteration][0].sum() / e2.beliefs[first_iteration][1].sum()}"
    )
    print(f"Last iteration: {e2.beliefs[last_iteration]}")
    print(
        f"Last iteration: {e2.beliefs[last_iteration][0].sum() / e2.beliefs[last_iteration][1].sum()}"
    )
