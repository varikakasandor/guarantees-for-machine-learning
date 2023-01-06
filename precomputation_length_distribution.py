from math import sqrt, log
import random
from matplotlib import pyplot as plt


def simulate(probabilities, delta):
    n = 0
    cnts = [0, 0, 0, 0]
    while min(cnts) - sqrt(n) * sqrt(log(4 / delta) / 2) < 8 * log(32 / delta):
        curr = random.choices(list(range(4)), probabilities)[0]
        n += 1
        cnts[curr] += 1
    return n


def analyse(probabilities, delta, num_simulations=100):
    ns = []
    for i in range(num_simulations):
        ns.append(simulate(probabilities, delta))

    minp = min(probabilities)
    a = minp ** 2
    b = -2 * 8 * log(32 / delta) * minp - log(4 / delta) / 2
    c = 64 * (log(32 / delta) ** 2)
    theoretical_expected_n = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a) # solving quadratic
    # this is an underestimate, because it assumes that the empirical argmin is the same as the argmin probability
    # the more there is a unique minimun probability, the more accurate this estimate is

    average_n = sum(ns) / num_simulations

    plt.axvline(theoretical_expected_n, color="red")
    plt.axvline(average_n, color="green")

    plt.hist(ns)
    plt.show()


if __name__ == "__main__":
    analyse([0.1, 0.1, 0.4, 0.4], 0.05) # the smaller the minimum probability is, the more samples we need
    # analyse([0.01, 0.3, 0.3, 0.39], 0.05)
    # analyse([0.1, 0.3, 0.3, 0.3], 0.05)
