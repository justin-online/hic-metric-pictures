# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
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
                "C:/Users/Justin Lee/Downloads/regularRecap/" + tissue + "_" + resolution + "/" + t + ".npy"
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

