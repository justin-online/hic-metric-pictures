import sys

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility

names = ["expected_1.npy", "expected_10.npy", "expected_11.npy", "expected_12.npy", "expected_13.npy",
         "expected_14.npy", "expected_15.npy", "expected_16.npy", "expected_17.npy", "expected_18.npy",
         "expected_19.npy", "expected_2.npy", "expected_20.npy", "expected_21.npy", "expected_22.npy", "expected_3.npy",
         "expected_4.npy", "expected_5.npy", "expected_6.npy", "expected_7.npy", "expected_8.npy", "expected_9.npy",
         "expected_X.npy"]
resolution = int(sys.argv[1])

plt.rc('font', size=40)  # controls default text sizes
plt.rc('axes', titlesize=100)  # fontsize of the axes title
plt.rc('axes', labelsize=60)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=60)  # fontsize of the tick labels
plt.rc('ytick', labelsize=60)  # fontsize of the tick labels
plt.rc('legend', fontsize=80)  # legend fontsize
plt.rc('figure', titlesize=100)  # fontsize of the figure title


def make_plot(id):
    fig, ax = plt.subplots(figsize=(40, 40))
    for yt in names:
        y = np.load(yt)

        # y2 = y2/y2[0]
        y2 = y[id, :]
        n = np.shape(y)[1]
        x1 = np.asarray(np.linspace(0, n - 1, n)) * resolution
        name = yt.replace("expected_", "").replace(".npy", "")
        plt.loglog(x1, y2, 'k--', label=name, linewidth=6)
    plt.ylabel('Expected Number of Contacts')
    plt.xlabel('Distance (BP)')
    # ax.legend(prop={'size': 100})
    plt.savefig(name + str(id) + '_contact_vs_dist.png')
    plt.close()


make_plot(0)
make_plot(1)
