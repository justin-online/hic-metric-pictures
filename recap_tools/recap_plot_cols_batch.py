import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st

plots = [("OBS_VAL", 0, 0),
         ("OE_VAL", 1, 0),
         ("OBS_STD_DEV", 2, 0),
         ("OE_STD_DEV", 3, 0),
         ("OBS_MAX_ENRICHMENT", 0, 1),
         ("OE_MAX_ENRICHMENT", 1, 1),
         ("OBS_MEAN_ENRICHMENT", 2, 1),
         ("OE_MEAN_ENRICHMENT", 3, 1),
         ("OBS_MEDIAN_ENRICHMENT", 0, 2),
         ("OBS_GEO_ENRICHMENT", 1, 2),
         ("OE_MEDIAN_ENRICHMENT", 2, 2),
         ("OE_GEO_ENRICHMENT", 3, 2),
         ("OBS_MIN_ENRICHMENT", 0, 3),
         ("OE_MIN_ENRICHMENT", 1, 3),
         ("OBS_SKEWNESS", 2, 3),
         ("OE_SKEWNESS", 3, 3),
         ("OBS_DECAY_A", 0, 4),
         ("OE_DECAY_A", 1, 4),
         ("OBS_DECAY_k", 2, 4),
         ("OE_DECAY_k", 3, 4),
         ("OBS_KURTOSIS", 0, 5),
         ("OE_KURTOSIS", 1, 5),
         ("PRESENCE", 2, 5),
         ("PRESENCE_INF", 3, 5)]

col1 = int(sys.argv[1])
col2 = int(sys.argv[2])
scale = 5

main_stem = "summary_" + sys.argv[1] + "_" + sys.argv[2]

data = {}
for (s, a, b) in plots:
    input_data = np.load(s + ".npy")[:, [col1, col2]]
    df = pd.DataFrame(input_data)
    df = df.dropna()
    data[s] = df.to_numpy()
    del df
    del input_data


def plot_map(stem, matrix, ax):
    ax.scatter(matrix[:, 0], matrix[:, 1], alpha=0.2)
    _, _, r_value, p_value, std_err = st.linregress(matrix[:, 0], matrix[:, 1])
    stats_string = "\n(R: " + "{:.2f}".format(r_value) + " , err: " + str(std_err) + ")"
    ax.set_title(stem + " " + stats_string)


fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(6 * scale, 4 * scale))
for (s, a, b) in plots:
    plot_map(s, data[s], axs[a, b])
fig.tight_layout()
plt.savefig(main_stem + ".png")
plt.close()

for (s, a, b) in plots:
    input_data = data[s].copy()
    input_data[input_data < 0] = 0
    input_data[input_data > 0] = np.log(input_data[input_data > 0])
    data[s] = input_data
    del input_data

fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(6 * scale, 4 * scale))
for (s, a, b) in plots:
    plot_map(s, data[s], axs[a, b])
fig.tight_layout()
plt.savefig("log_" + main_stem + ".png")
plt.close()
