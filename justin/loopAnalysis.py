# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
from plottingFxns import *

# LOAD ALL METRICS
diffuse_obs_decay = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_DECAY_FULL.npy")
diffuse_obs_row_sum = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_ROW_SUM.npy")
diffuse_obs_amp_row = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_AMP_ROW.npy")
diffuse_obs_spread_row = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_SPREAD_ROW.npy")
diffuse_obs_amp_col = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_AMP_COL.npy")
diffuse_obs_spread_col = np.load("loopAnalysisFiles/diffuseLoopAnalysis/OBS_SPREAD_COL.npy")
#
sharp_obs_decay = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_DECAY_FULL.npy")
sharp_obs_row_sum = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_ROW_SUM.npy")
sharp_obs_amp_row = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_AMP_ROW.npy")
sharp_obs_spread_row = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_SPREAD_ROW.npy")
sharp_obs_amp_col = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_AMP_COL.npy")
sharp_obs_spread_col = np.load("loopAnalysisFiles/sharpLoopAnalysis/OBS_SPREAD_COL.npy")
#
combined_obs_decay = np.load("loopAnalysisFiles/combinedLoopAnalysis/OBS_DECAY_FULL.npy")
combined_obs_row_sum = np.load("loopAnalysisFiles/combinedLoopAnalysis/OBS_ROW_SUM.npy")
combined_obs_amp_row = np.load("loopAnalysisFiles/combinedLoopAnalysis/OBS_AMP_ROW.npy")
combined_obs_spread_row = np.load("loopAnalysisFiles/combinedLoopAnalysis/OBS_SPREAD_ROW.npy")
#
diffuse_apa = np.load("loopAnalysisFiles/diffuse_APA/apa.npy")
sharp_apa = np.load("loopAnalysisFiles/sharp_APA/apa.npy")

# FUNCTIONS:

# FIGURE 1: DECAY AND ROW SUM OF SHARP AND DIFFUSE
fig, axs = plt.subplots(2, 2)
metric_plotter(diffuse_obs_decay, axs[0, 0], "diffuse_obs_decay")
metric_plotter(sharp_obs_decay, axs[0, 1], "sharp_obs_decay")
metric_plotter(diffuse_obs_row_sum, axs[1, 0], "diffuse_obs_row_sum")
metric_plotter(sharp_obs_row_sum, axs[1, 1], "sharp_obs_row_sum")
plt.suptitle("Decay and Row Sum of Sharp and Diffuse Loops")
print(sharp_obs_row_sum)

plt.savefig("fig1.png")
plt.show()

# FIGURE 2: [ROW] AMPLITUDE_STD VS SPREAD_STD GRAPH
plt.scatter(diffuse_obs_amp_row, diffuse_obs_spread_row, alpha=0.5, c="r", label="diffuse")
plt.scatter(sharp_obs_amp_row, sharp_obs_spread_row, alpha=0.5, c="b", label="sharp")
plt.suptitle("[ROW] Amplitude vs Spread Std Dev")
plt.legend()
plt.savefig("fig2.png") # add this to every
plt.show()

# FIGURE 3: [COL] AMPLITUDE_STD VS SPREAD_STD GRAPH
plt.scatter(diffuse_obs_amp_col, diffuse_obs_spread_col, alpha=0.5, c="r", label="diffuse")
plt.scatter(sharp_obs_amp_col, sharp_obs_spread_col, alpha=0.5, c="b", label="sharp")
plt.suptitle("[COL] Amplitude vs Spread Std Dev")
plt.legend()
plt.savefig("fig3.png")
plt.show()

# FIGURE 4: [ROW * COL] AMPLITUDE_STD VS SPREAD_STD GRAPH (RED = DIFFUSE, BLUE = SHARP)
plt.scatter(diffuse_obs_amp_col * diffuse_obs_amp_row, diffuse_obs_spread_col * diffuse_obs_spread_row, alpha=0.5, c="r", label="diffuse")
plt.scatter(sharp_obs_amp_col * sharp_obs_amp_row, sharp_obs_spread_col * sharp_obs_spread_row, alpha=0.5, c="b", label="sharp")
plt.suptitle("[ROW * COL] Amplitude vs Spread Std Dev")
plt.legend()
plt.savefig("fig4.png")
plt.show()

# FIGURE 5: APA PLOT SHARP VS DIFFUSE
fig, axs = plt.subplots(1, 2)
plot_hic_map(sharp_apa, "sharp", axs[0])
plot_hic_map(diffuse_apa, "diffuse", axs[1])
plt.suptitle("APA Plot of Sharp vs Diffuse")
plt.savefig("fig5.png")
plt.show()

