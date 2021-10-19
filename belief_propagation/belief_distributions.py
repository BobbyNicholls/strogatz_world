from numpy.random import normal, uniform
import pandas as pd
from scipy.stats import chi2, skewnorm


def indifference_with_very_little_variation(indifference_scaler=4):
    return abs((normal() / indifference_scaler) + 5)


def extremely_high_aversion():
    return chi2.rvs(0.35)


def quite_high_aversion():
    return skewnorm.rvs(10) * 3 + 1


def extremely_high_affinity():
    return abs(10 - chi2.rvs(0.35))


def quite_high_affinity():
    return abs(skewnorm.rvs(-10) * 3 + 10)


def indifference_with_some_variation():
    return abs(normal() + 5)


def indifference_with_very_little_variation():
    return abs(normal() / 4 + 5)


def true_random():
    return uniform() * 10


belief_dist_function_dict = {
    "true_random": true_random,
    "strong_affinity": extremely_high_affinity,
    "affinity": quite_high_affinity,
    "strong_aversion": extremely_high_aversion,
    "aversion": quite_high_aversion,
    "indifferent": indifference_with_very_little_variation,
    "random_indifferent": indifference_with_some_variation,
}

if __name__ == "__main__":

    # for node in G_copy.nodes:
    #     print(G_copy.nodes[node]["race"], G_copy.nodes[node]["faction"])

    r = chi2.rvs(20, size=1000)
    pd.Series(r).hist(bins=100)
    r = chi2.rvs(2, size=1000)
    pd.Series(r).hist(bins=100)
    r = chi2.rvs(200, size=1000)
    pd.Series(r).hist(bins=100)

    # extremely high aversion belief dist
    df = 0.35
    mean, var, skew, kurt = chi2.stats(df, moments="mvsk")
    r = chi2.rvs(df, size=10000)
    pd.Series(r).hist(bins=80)

    # extremely high affinity belief dist
    r_inv = [abs(10 - x) for x in r]
    pd.Series(r_inv).hist(bins=80)

    # quite high affinity
    a = -10
    mean, var, skew, kurt = skewnorm.stats(a, moments="mvsk")
    r = abs(skewnorm.rvs(a, size=10000) * 3 + 10)
    pd.Series(r).hist(bins=80)

    # quite high aversion
    a = 10
    mean, var, skew, kurt = skewnorm.stats(a, moments="mvsk")
    r = skewnorm.rvs(a, size=10000) * 3 + 1
    # r = abs(skewnorm.rvs(a, size=10000)*3 + 10)
    pd.Series(r).hist(bins=80)

    # indifference with some variation
    r = abs(normal(size=10000) + 5)
    pd.Series(r).hist(bins=80)
    abs(normal() + 5)

    # indifference with very little variation
    r = abs(normal(size=10000) / 4 + 5)
    pd.Series(r).hist(bins=80)
    abs(normal() + 5)

    # true random
    r = uniform(size=10000) * 10
