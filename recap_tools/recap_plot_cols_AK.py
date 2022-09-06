import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plots = ["OBS_DECAY_A",
         "OE_DECAY_A",
         "OBS_DECAY_k",
         "OE_DECAY_k"]

col1 = int(sys.argv[1])
window = int(sys.argv[2])
scale = 5

main_stem = "summary_" + sys.argv[1]

data = {}
for s in plots:
    input_data = np.squeeze(np.load(s + ".npy")[:, col1])
    # df = pd.DataFrame(input_data)
    # df = df.dropna()
    data[s] = input_data  # df.to_numpy()
    # del df
    del input_data


def plot_map(stem, data1, data2, ax):
    matrix = np.vstack((data1, data2)).T
    print(np.shape(matrix))
    df = pd.DataFrame(matrix)
    df = df.dropna()
    matrix = df.to_numpy()
    del df
    ax.scatter(matrix[:, 0], matrix[:, 1], alpha=0.2)
    # _, _, r_value, p_value, std_err = st.linregress(data)
    # stats_string = "\n(R: " + "{:.2f}".format(r_value) + " , err: " + str(std_err) + ")"
    ax.set_title(stem)  # + " " + stats_string)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2 * scale, scale))
plot_map("OBS", data["OBS_DECAY_A"], data["OBS_DECAY_k"], axs[0])
plot_map("OE", data["OE_DECAY_A"], data["OE_DECAY_k"], axs[1])
fig.tight_layout()
plt.savefig(main_stem + ".png")
plt.close()

plots = ["OBS_DECAY_FULL",
         "OE_DECAY_FULL"]

main_stem += "_decay"

data = {}
for s in plots:
    input_data = np.squeeze(np.load(s + ".npy")[:, col1 * (window + 1):(col1 + 1) * (window + 1)])
    # df = pd.DataFrame(input_data)
    # df = df.dropna()
    data[s] = input_data  # df.to_numpy()
    # del df
    del input_data


def plot_lines(stem, data1, ax):
    df = pd.DataFrame(data1)
    df = df.dropna()
    matrix = df.to_numpy()
    del df
    for r in range(np.shape(matrix)[0]):
        ax.plot(matrix[r, :], label=str(r))
    ax.set_title(stem)  # + " " + stats_string)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2 * scale, scale))
plot_lines("OBS", data["OBS_DECAY_FULL"], axs[0])
plot_lines("OE", data["OE_DECAY_FULL"], axs[1])
fig.tight_layout()
plt.savefig(main_stem + ".png")
plt.close()
