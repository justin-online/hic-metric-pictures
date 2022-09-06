import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def metric_plotter(matrix, ax, title_name, oneDarray = False):
    if oneDarray:
        for loop_index in range(0,len(matrix),200):
            x = np.arange(len(matrix))  # number of columns
            ax.plot(x, matrix, alpha=0.5)
            ax.set_title(title_name)
    else:
        for loop_index in range(0,len(matrix),200):
            #print(matrix[0])
            x = np.arange(len(matrix[0]))  # number of columns
            ax.plot(x, matrix[loop_index, :], alpha=0.5)
            ax.set_title(title_name)

def get_score(matrix):
    r = np.shape(matrix)[0]
    buffer = r // 4
    color_lim2 = 5 * np.mean(matrix[:buffer, -buffer:])
    score = matrix[r // 2, r // 2] / np.mean(matrix[-buffer:, :buffer])
    return score, color_lim2

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])
def plot_hic_map(dense_matrix, name, ax):
    # helper function for plotting

    # fig, ax = plt.subplots(figsize=(10,10))
    d2 = dense_matrix.copy()
    # d2[d2 < 0.9] = 0
    score, color_lim2 = get_score(d2)
    im = ax.matshow(d2, cmap=REDMAP, vmin=0, vmax=color_lim2)  # cmap=REDMAP
    # plt.colorbar(im)
    # plt.savefig(name+'.pdf')
    ax.set_title(name)
    ax.xaxis.set_ticklabels(np.arange(0, 1), visible=False)
    ax.yaxis.set_ticklabels(np.arange(0, 1), visible=False)
    ax.xaxis.set_tick_params(length=0, labelsize=5, labeltop='off', labelbottom='on')
    ax.yaxis.set_tick_params(length=0, labelsize=5)
    ax.set_title(name + '(score = {:.2f})'.format(score))

def aggregateSumCalculator(list_of_resolutions, numpy_dict, tissue, type):
    # GIVEN a list of resolutions, the dictionary tissueResDict, the tissue, and the type,
    # dictionary[resolution] = [elm1, elm2, ..., elm_n] aggregate array
    aggregateSumDict= {}
    nanInLoop = False
    for resolution in list_of_resolutions:
        aggregateSumDict[resolution] = np.array([0] * len(numpy_dict[tissue][resolution][type][0]), dtype=np.float64) #arbitrary 1st element
        for loop in numpy_dict[tissue][resolution][type]:
            for elm in loop:
                if np.isnan(elm):
                    nanInLoop = True
            if not nanInLoop:
                aggregateSumDict[resolution] += loop
            nanInLoop = False  # at end of every loop, refresh assumption that loop does not contain NaN values
    return aggregateSumDict

def aggregateSumOfArray(array):
    nanInLoop = False
    #print(array)
    aggregateSumArray = np.zeros(np.size(array[0]))
    for loop in array:
        #print("loop:", loop)
        for elm in loop:
            if np.isnan(elm):
                nanInLoop = True
        if not nanInLoop:
            #print(aggregateSumArray, np.size(aggregateSumArray))
            #print(loop, np.size(loop))
            aggregateSumArray = np.add(aggregateSumArray, loop)
        nanInLoop = False  # at end of every loop, refresh assumption that loop does not contain NaN values
    return aggregateSumArray


#fig, axs = plt.subplots(1)
#metric_plotter(np.array([0, 1, 2, 3]), axs, "hello", True)
#plt.show()

