import sys

import numpy as np
from matplotlib import pyplot as plt

with open(sys.argv[1]) as f:
    v = np.loadtxt(f, delimiter=",", dtype='float', comments="#", skiprows=0, usecols=None)
    v_hist = np.log(1 + np.ravel(v))  # 'flatten' v
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(v_hist, bins=500, facecolor='green')
    plt.show()
    plt.savefig('simple_hist.png')
