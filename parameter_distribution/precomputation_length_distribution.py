from math import sqrt, log
import random
from matplotlib import pyplot as plt


def simulate(probabilities, delta, correction_factor):
    n = 0
    cnts = [0, 0, 0, 0]
    while correction_factor * (min(cnts) - sqrt(n) * sqrt(log(4 / delta) / 2)) < 8 * log(32 / delta):
        curr = random.choices(list(range(4)), probabilities)[0]
        n += 1
        cnts[curr] += 1
    alpha = 4 * sqrt(log(256 / delta) / min(cnts))
    return n, alpha


def create_plot(expected_value, empirical_average, distribution, quantity, probabilities, delta, correction_factor, show_result):
    plt.clf()
    plt.axvline(expected_value, color="red", label="theoretical mean")
    plt.axvline(empirical_average, color="green", label="empirical mean")
    plt.hist(distribution)
    plt.title(f"distribution of {quantity}: p={probabilities}, delta={delta}, correction={correction_factor}")
    plt.xlabel(f"{quantity}")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(f"./figures/distribution_{quantity}_{probabilities}_{delta}_{correction_factor}.jpg")
    if show_result:
        plt.show()


def analyse(probabilities, delta, correction_factor=0.3, num_simulations=100, show_result=False):
    ns = []
    alphas = []
    for i in range(num_simulations):
        n, alpha = simulate(probabilities, delta, correction_factor)
        ns.append(n)
        alphas.append(alpha)
    minp = min(probabilities)
    a = (correction_factor * minp) ** 2
    b = -2 * correction_factor * 8 * log(32 / delta) * minp - (correction_factor ** 2) * log(4 / delta) / 2
    c = 64 * (log(32 / delta) ** 2)
    expected_n = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)  # solving quadratic
    # this is an underestimate, because it assumes that the empirical argmin is the same as the argmin probability
    # the more there is a unique minimun probability, the more accurate this estimate is
    expected_alpha = 4 * sqrt(log(256 / delta) / (expected_n * min(probabilities)))
    # this is cheating, this is just some approximation

    average_n = sum(ns) / num_simulations
    average_alpha = sum(alphas) / num_simulations

    create_plot(expected_n, average_n, ns, "n", probabilities, delta, correction_factor, show_result)
    create_plot(expected_alpha, average_alpha, alphas, "alpha", probabilities, delta, correction_factor, show_result)


if __name__ == "__main__":
    analyse([0.1, 0.1, 0.4, 0.4], 0.05)
    analyse([0.01, 0.3, 0.3, 0.39], 0.05)  # the smaller the minimum probability is, the more samples we need
    analyse([0.1, 0.3, 0.3, 0.3], 0.05)
