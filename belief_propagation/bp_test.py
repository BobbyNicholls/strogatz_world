import numpy as np
from utils import get_random_node_features

# left argument "influences" the right argument,
# the first arg determines the row beleif, and the second the column belief
# This means the column of the second argument is "listening" for the values of the row of the first


class Entity:
    def __init__(self, feature_vector, beliefs):
        assert type(beliefs) == np.ndarray, "beliefs must be array"
        self.feature_vector = feature_vector
        self.beliefs = {ITERATION - 1: beliefs/beliefs.sum()}

    def propagate_belief(self, influencer):
        self.beliefs[ITERATION] = np.dot(
            influencer.beliefs[ITERATION - 1], self.beliefs[ITERATION - 1]
        )
        self.beliefs[ITERATION] = self.beliefs[ITERATION]/self.beliefs[ITERATION].sum()

    def iterate_belief(self):
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
        e1.propagate_belief(influencer=e3)
        e2.propagate_belief(influencer=e3)
        e3.iterate_belief()
        ITERATION += 1

    first_iteration = min(e1.beliefs.keys())
    last_iteration = max(e1.beliefs.keys())
    print(f"First iteration: {e1.beliefs[first_iteration][0].sum() / e1.beliefs[first_iteration][1].sum()}")
    print(f"Last iteration: {e1.beliefs[last_iteration][0].sum() / e1.beliefs[last_iteration][1].sum()}")


"""
Ok lets create a scenario, an individual with a fairly strong beleif that something is true, the colums represent their
beleif and the rows how they respond to the belief of others:
"""

p1 = np.array([[1, 15], [5, 50]])

"""
note that the person represented by potential 1 will beleive in the thing more strongly if their influence does too, so
lets take some examples of influences:
(note these influences are not themselves influenced overmuch by their respective influences, columns are the same)
"""

# the influence that confirms the belief strongly. this works, you can see the bottom row goes from being a total of:
# (50+5) / (1+15) = 3.437
# (30+325) / (6+65) = 5.0
# but really it just makes the row in to the exact same ratio as whatever you put in the bottom row of the influencer,
# which doesnt feel great because then we arent really considering the original views of the node. That being said, the
# column ratio is now 10.83:1 on both levels individually and in total (65+325)/(6+30) = 10.833..
p2 = np.array([[1, 1], [5, 5]])
print(str(np.dot(p2, p1)) + "\n========\n")

# the influence that rejects it strongly
p3 = np.array([[50, 55], [2, 1]])
print(str(np.dot(p3, p1)) + "\n========\n")

# the indifferent influence, what we notice here is that the strength of the belief doesnt change, but how we are
# influenced does (we are no longer strongly influenced)
p4 = np.array([[1, 1], [1, 1]])
print(str(np.dot(p4, p1)) + "\n========\n")


"""
now lets look at evidence
"""

# the evidence that confirms it 100%
e1 = np.array([[0, 1]])
print(str(np.dot(e1, p1)) + "\n========\n")

# the evidence that represents strong confirmation
e2 = np.array([8, 2])
print(str(np.dot(e2, p1)) + "\n========\n")

# the strong rejection, this means bottom heavy
e3 = np.array([[1, 1], [10, 10]])
print(str(np.dot(e3, p1)) + "\n========\n")

# the 100% rejection
e4 = np.array([1, 0])
print(str(np.dot(e4, p1)) + "\n========\n")
