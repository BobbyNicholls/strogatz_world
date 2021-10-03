This creates randomly generated social networks, a feature vector is associated with
the individual entities in the network and a belief matrix can be created. The programme
is then designed to propagate beliefs of nodes around the network to demonstrate
the way beliefs are propagated around social networks by influential members of the network.

This is for making fictional worlds.

It uses some Graph Machine Learning concepts to have intelligent evolving networks that use a form of
belief propagation based on the "Loopy belief propagation" algorithm. The networks are
generated using the watts-strogatz generative graph algorithm.

The belief matrices are structured as "potentials", making the graph a Bayesian Network
where the beliefs of a given entity can be updated by the beliefs of a linked
entity through belief propagation and links represent probabilistic dependencies. A "child" entity has beliefs that are conditional
on the "parent" entity, in this case the "parent" can be simply an influential node in
the network, not necessarily the actual parent of the entity.
