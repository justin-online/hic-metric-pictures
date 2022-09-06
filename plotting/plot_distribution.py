import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_my_hist(temp, my_title):
    # An "interface" to matplotlib.axes.Axes.hist() method
    fig, ax = plt.subplots(figsize=(10, 10))
    n, bins, patches = ax.hist(x=temp, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85
                               )
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(my_title)
    plt.savefig(my_title + ".png")
    plt.close()

    stats.probplot(temp, dist=stats.norm, plot=plt)
    plt.savefig(my_title + "_qq.png")
    plt.close()


def sample(distributions):
    n2 = max(1, len(distributions) // 1000)
    return distributions[::n2]


for key in [10, 100, 1000, 10000]:
    print(key)
    distributions = np.load("distribution_" + str(key) + ".npy").flatten()
    plot_my_hist(distributions, "distribution_" + str(key))
    print(stats.shapiro(sample(distributions)))

    distributions = np.log(1 + distributions)
    plot_my_hist(distributions, "log_distribution_" + str(key))
    print(stats.shapiro(sample(distributions)))

    distributions = np.log(1 + distributions)
    plot_my_hist(distributions, "log_log_distribution_" + str(key))
    print(stats.shapiro(sample(distributions)))

    distributions = np.log(1 + distributions)
    plot_my_hist(distributions, "log_log_log_distribution_" + str(key))
    print(stats.shapiro(sample(distributions)))
