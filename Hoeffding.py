"""
Solve Homework#1 from the course "Learning from data" (edX, Caltech)

Flip 1000 coins 10 times each, and calculate the fraction of 'heads' for each coin.
Repeat this experiment 1000 times, and plot the distribution (histogram) for the following cases:
 - The first coin.
 - A random coin in each experiment instance.
 - The coin with minimum fraction of heads in each experiment.
See which of the coins are under the Hoeffding hyphothesis.
"""

from numpy.random import randint
from functools import reduce
import matplotlib.pyplot as plt


# Wrapper object for Coins
class Coin:
    is_head = [False] * 10  # Each coin will be flipped 10 times
    fraction_heads = 0.0  # Keep track of the fraction of 'heads' obtained after flipping the 10 coins

# Array of coins
coins = [Coin() for i in range(1000)]

# Build three histograms to approximate the distribution of fraction_heads for the following coins:
# c1 is the first coin.
# cmin is the coin with the lowest fraction_heads
# crand is a random coin in each instance of the experiment
histogram_points = 11  # Shall include 0 and 1, in steps of 0.1
fraction_heads_histogram_c1 = [0] * histogram_points
fraction_heads_histogram_cmin = [0] * histogram_points
fraction_heads_histogram_crand = [0] * histogram_points


def fill_histogram(histogram, coin):
    # fraction_heads oscilates between 0 and 1
    histogram_index = int(coin.fraction_heads * (len(histogram) - 1))
    histogram[histogram_index] += 1


for experiment_instance in range(1000):
    if (experiment_instance + 1) % 500 == 0:
        print("Iteration %d" % experiment_instance)
    # Flip all coins 10 times, calculate fraction_heads for each coin
    for coin in coins:
        coin.is_head = randint(0, 2, len(coin.is_head))
        coin.fraction_heads = float(sum(coin.is_head)) / len(coin.is_head)

    # ------ Fill the histograms for First Coin, Random Coin, and min fraction_heads Coin ------
    # First coin
    fill_histogram(fraction_heads_histogram_c1, coins[0])
    # Coin with min fraction_heads
    fill_histogram(fraction_heads_histogram_cmin,
                   reduce(lambda min_coin, current:
                          current if current.fraction_heads < min_coin.fraction_heads
                          else min_coin,
                          coins))
    # Random coin
    fill_histogram(fraction_heads_histogram_crand, coins[randint(0, len(coins))])

# ---------- Plot Histograms ---------
plt.figure()
plt.title('Distribution of fraction_heads in 100000 experiments')
x_axis = [float(i)/histogram_points for i in range(0, histogram_points)]
plt.plot(x_axis, fraction_heads_histogram_c1, label='Distribution of fraction_heads - First coin')
plt.figure()
plt.plot(x_axis, fraction_heads_histogram_cmin, label='Distribution of fraction_heads - Coins with min fraction_heads')
plt.figure()
plt.plot(x_axis, fraction_heads_histogram_crand, label='Distribution of fraction_heads - Random coins')
plt.legend()
plt.show()
