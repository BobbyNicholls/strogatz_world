import numpy as np

p1 = np.array([[2, 1],
               [21, 750]])
p2 = np.array([[6, 19],
               [9, 2]])

# left argument "influences" the right argument,
# the first arg determines the row beleif, and the second the column belief
# This means the column of the second argument is "listening" for the values of the row of the first
np.dot(p1, p2)



"""
Ok lets create a scenario, an individual with a fairly strong beleif that something is true, the colums represent their
beleif and the rows how they respond to the belief of others:
"""

p1 = np.array([[1, 15],
              [5, 50]])

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
p2 = np.array([[1, 1],
               [5, 5]])
print(str(np.dot(p2, p1)) + "\n========\n")

# the influence that rejects it strongly
p3 = np.array([[50, 55],
               [2, 1]])
print(str(np.dot(p3, p1)) + "\n========\n")

# the indifferent influence, what we notice here is that the strength of the belief doesnt change, but how we are
# influenced does (we are no longer strongly influenced)
p4 = np.array([[1, 1],
               [1, 1]])
print(str(np.dot(p4, p1)) + "\n========\n")


"""
now lets look at evidence
"""

# the evidence that confirms it 100%
e1 = np.array([[0,1]])
print(str(np.dot(e1, p1)) + "\n========\n")

# the evidence that represents strong confirmation
e2 = np.array([8,2])
print(str(np.dot(e2, p1)) + "\n========\n")

# the strong rejection, this means bottom heavy
e3 = np.array([[1,1],
               [10,10]])
print(str(np.dot(e3, p1)) + "\n========\n")

# the 100% rejection
e4 = np.array([1,0])
print(str(np.dot(e4, p1)) + "\n========\n")