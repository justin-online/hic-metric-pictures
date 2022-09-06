import sys

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility

y = np.load(sys.argv[1])
n = np.shape(y)[1]
resolution = int(sys.argv[2])

x1 = np.asarray(np.linspace(0, n - 1, n)) * resolution

plt.rc('font', size=40)  # controls default text sizes
plt.rc('axes', titlesize=100)  # fontsize of the axes title
plt.rc('axes', labelsize=60)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=60)  # fontsize of the tick labels
plt.rc('ytick', labelsize=60)  # fontsize of the tick labels
plt.rc('legend', fontsize=80)  # legend fontsize
plt.rc('figure', titlesize=100)  # fontsize of the figure title


def make_plot(y1, y2, name):
    fig, ax = plt.subplots(figsize=(40, 40))
    plt.loglog(x1, y1, 'g-', label='Mean ' + name, linewidth=6)
    plt.loglog(x1, y2, 'y-', label='Mean ' + name + ' Interp', linewidth=6)
    plt.ylabel('Expected Number of Contacts')
    plt.xlabel('Distance (BP)')
    ax.legend(prop={'size': 100})
    plt.savefig(name + '_contact_vs_dist.png')
    plt.close()


make_plot(y[0, :], y[2, :], '')
make_plot(y[1, :], y[3, :], 'log')
make_plot(y[4, :], y[5, :], 'default')
