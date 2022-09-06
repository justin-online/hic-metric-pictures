# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from plottingFxns import *
import sys

sys.path.append("C:/Users/Justin Lee/Downloads")

# LOAD npy files into a 3-level nested dictionary
# tissueResDict[tissue][resolution][t] = corresponding npy matrix
tissueResDict = {}
TISSUES = ["lung", "ovary", "liver", "immune", "colon"]
RESOLUTIONS = ["100", "500", "1000"]
TYPES = ["OBS_DECAY_FULL", "OBS_ROW_SUM", "OBS_AMP_ROW", "OBS_SPREAD_ROW", "OBS_AMP_COL", "OBS_SPREAD_COL"]

for tissue in TISSUES:
    tissueResDict[tissue] = {}
    for resolution in RESOLUTIONS:
        tissueResDict[tissue][resolution] = {}
        for t in TYPES:
            tissueResDict[tissue][resolution][t] = np.load(
                "C:/Users/Justin Lee/Downloads/deepRecap/" + tissue + "_" + resolution + "/" + t + ".npy"
            )

# FIGURE 1: lung decay
fig, axs = plt.subplots(2, 3)
metric_plotter(tissueResDict["lung"]["100"]["OBS_DECAY_FULL"], axs[0, 0], "lung 100 decay")
metric_plotter(tissueResDict["lung"]["500"]["OBS_DECAY_FULL"], axs[0, 1], "lung 500 decay")
metric_plotter(tissueResDict["lung"]["1000"]["OBS_DECAY_FULL"], axs[0, 2], "lung 1000 decay")

aggregateDecaySum = aggregateSumCalculator(RESOLUTIONS, tissueResDict, "lung", "OBS_DECAY_FULL")
metric_plotter(aggregateDecaySum["100"], axs[1, 0], "lung aggregate decay sum 100", oneDarray=True)
metric_plotter(aggregateDecaySum["500"], axs[1, 1], "lung aggregate decay sum 500", oneDarray=True)
metric_plotter(aggregateDecaySum["1000"], axs[1, 2], "lung aggregate decay sum 1000", oneDarray=True)
plt.savefig("aggregatelungdecay.png")
plt.show()

# FIGURE 2: lung row sum
fig, axs = plt.subplots(2, 3)
metric_plotter(tissueResDict["lung"]["100"]["OBS_ROW_SUM"], axs[0, 0], "lung 100 row sum")
metric_plotter(tissueResDict["lung"]["500"]["OBS_ROW_SUM"], axs[0, 1], "lung 500 row sum")
metric_plotter(tissueResDict["lung"]["1000"]["OBS_ROW_SUM"], axs[0, 2], "lung 1000 row sum")

aggregateRowSum = aggregateSumCalculator(RESOLUTIONS, tissueResDict, "lung", "OBS_ROW_SUM")
metric_plotter(aggregateRowSum["100"], axs[1, 0], "lung aggregate row sum 100", oneDarray=True)
metric_plotter(aggregateRowSum["500"], axs[1, 1], "lung aggregate row sum 500", oneDarray=True)
metric_plotter(aggregateRowSum["1000"], axs[1, 2], "lung aggregate row sum 1000", oneDarray=True)
plt.savefig("aggregatelungrowsum.png")
plt.show()

# FIGURE 3: immune decay
fig, axs = plt.subplots(2, 3)
metric_plotter(tissueResDict["immune"]["100"]["OBS_DECAY_FULL"], axs[0, 0], "immune 100 decay")
metric_plotter(tissueResDict["immune"]["500"]["OBS_DECAY_FULL"], axs[0, 1], "immune 500 decay")
metric_plotter(tissueResDict["immune"]["1000"]["OBS_DECAY_FULL"], axs[0, 2], "immune 1000 decay")

aggregateDecaySum = aggregateSumCalculator(RESOLUTIONS, tissueResDict, "immune", "OBS_DECAY_FULL")
metric_plotter(aggregateDecaySum["100"], axs[1, 0], "immune aggregate decay sum 100", oneDarray=True)
metric_plotter(aggregateDecaySum["500"], axs[1, 1], "immune aggregate decay sum 500", oneDarray=True)
metric_plotter(aggregateDecaySum["1000"], axs[1, 2], "immune aggregate decay sum 1000", oneDarray=True)
plt.savefig("immuneaggregatedecay.png")
plt.show()

# FIGURE 4: immune row sum
fig, axs = plt.subplots(2, 3)
metric_plotter(tissueResDict["immune"]["100"]["OBS_ROW_SUM"], axs[0, 0], "immune 100 row sum")
metric_plotter(tissueResDict["immune"]["500"]["OBS_ROW_SUM"], axs[0, 1], "immune 500 row sum")
metric_plotter(tissueResDict["immune"]["1000"]["OBS_ROW_SUM"], axs[0, 2], "immune 1000 row sum")

aggregateRowSum = aggregateSumCalculator(RESOLUTIONS, tissueResDict, "immune", "OBS_ROW_SUM")
metric_plotter(aggregateRowSum["100"], axs[1, 0], "immune aggregate row sum 100", oneDarray=True)
metric_plotter(aggregateRowSum["500"], axs[1, 1], "immune aggregate row sum 500", oneDarray=True)
metric_plotter(aggregateRowSum["1000"], axs[1, 2], "immune aggregate row sum 1000", oneDarray=True)
plt.savefig("immuneaggregaterowsum.png")
plt.show()

# FIGURE 3: [row * col] spread standard deviation vs amplitude std deviation
plt.scatter(tissueResDict["lung"]["100"]["OBS_AMP_ROW"] * tissueResDict["lung"]["100"]["OBS_AMP_COL"], tissueResDict["lung"]["100"]["OBS_SPREAD_ROW"] * tissueResDict["lung"]["100"]["OBS_SPREAD_COL"], alpha=0.5, c="b", label="100")
plt.show()

plt.scatter(tissueResDict["lung"]["500"]["OBS_AMP_ROW"] * tissueResDict["lung"]["500"]["OBS_AMP_COL"], tissueResDict["lung"]["500"]["OBS_SPREAD_ROW"] * tissueResDict["lung"]["500"]["OBS_SPREAD_COL"], alpha=0.5, c="r", label="500")
plt.show()

plt.scatter(tissueResDict["lung"]["1000"]["OBS_AMP_ROW"] * tissueResDict["lung"]["1000"]["OBS_AMP_COL"], tissueResDict["lung"]["1000"]["OBS_SPREAD_ROW"] * tissueResDict["lung"]["1000"]["OBS_SPREAD_COL"], alpha=0.5, c="g", label="1000")
plt.suptitle("[ROW] AMPLITUDE_STD VS SPREAD_STD GRAPH")
plt.legend()
plt.show()


# KMEANS: PLOT "SHARP" AND "DIFFUSE" CLUSTERS

col_matrix1 = tissueResDict["lung"]["100"]["OBS_AMP_ROW"] * tissueResDict["lung"]["100"]["OBS_AMP_COL"]
col_matrix2 = tissueResDict["lung"]["100"]["OBS_SPREAD_ROW"] * tissueResDict["lung"]["100"]["OBS_SPREAD_COL"]

# create dataset of [obs_amp row*col, obs_spread row*col]
new_matrix = np.zeros(shape=(np.size(col_matrix1), 2), dtype=object)
for i in range(np.size(col_matrix1)):
    new_matrix[i] = [col_matrix1[i, 0], col_matrix2[i, 0]]
    # forgot to index 0, that was the bug i fixed

# scale the data before running kmean clustering method
new_matrix = sklearn.preprocessing.scale(new_matrix)

# changing settings for kmeans
kmeans = KMeans(n_clusters=5)
label = kmeans.fit_predict(new_matrix)

u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(new_matrix[label == i, 0], new_matrix[label == i, 1], label = i,alpha=0.2)
plt.legend()
ax = plt.gca()
leg = ax.get_legend()
print(leg.legendHandles)
plt.suptitle("[ROW * COL] Amplitude vs Spread Std Dev")
plt.savefig("banana.png")
plt.show()

# initializes empty row array with width of the 1st element in the obs_row_sum npy file and appends only rows from that
# file that correspond to the correct cluster label

# create a dictionary cluster # -> filtered numpy array

filteredClusterDict = {}
for cluster_number in u_labels:
    filteredClusterDict[cluster_number] = np.empty((0, np.size(tissueResDict["lung"]["100"]["OBS_ROW_SUM"][0])), float)
for i in range(np.size(label)):  # i is the index
    for one_label in u_labels:
        if label[i] == one_label:
            filteredClusterDict[one_label] = np.append(filteredClusterDict[one_label],
                                           np.array([tissueResDict["lung"]["100"]["OBS_ROW_SUM"][i]]), axis=0)

# dictionary label_counts dict label -> count of that label in variable "label"
unique, counts = np.unique(label, return_counts=True)
label_counts_dict = dict(zip(unique, counts))
num_clusters = np.size(u_labels)
figs, axs = plt.subplots(1, num_clusters, sharey=True)
for i in range(num_clusters):
    # normalize by the number of loops being counted in each row sum
    aggregateSumArray = aggregateSumOfArray(filteredClusterDict[i])
    aggregateSumArray = aggregateSumArray / label_counts_dict[i]
    metric_plotter(aggregateSumArray, axs[i], "cluster " + str(i), oneDarray=True)

# makes a common y label
figs.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)

plt.suptitle("Row Sums of Each Cluster")
plt.savefig("rowsumscluster.png")
plt.show()



###### DBSCAN
"""
dbscan = DBSCAN()
label = dbscan.fit_predict(new_matrix)
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(new_matrix[label == i, 0], new_matrix[label == i, 1], label = i)
plt.legend()
plt.show()
"""