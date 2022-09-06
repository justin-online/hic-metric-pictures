import sys

import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

input_data = np.load(sys.argv[1])
stem = sys.argv[2]
do_log = int(sys.argv[3]) == 1

input_data = input_data[:, [0, 1, 2, 3, 4, 5, 6, 8]]

name_of_files = ["left_ventricle", "right_ventricle", "atria", "colon", "immune",
                 "liver", "lung", "ovary"]

df = pd.DataFrame(input_data, columns=name_of_files)
df = df.dropna()  # todo differently in future
input_data = df.to_numpy()

if do_log:
    input_data[input_data < 0] = 0
    input_data[input_data > 0] = np.log(input_data[input_data > 0])

# down sample
# input_data = input_data[::100, :]

transposed_inputdata = input_data.T

# determines how many principal components or features the original dataset will be reduced to.

title_for_kmeans_clusterfinder = "The Elbow Method showing the optimal k for kmeans"
dendogram_name = "dendogram plot"

num_of_gmm_clusters = 10
num_of_kmeans_clusters = 3

x_label = "embedded1"
y_label = "embedded2"

UMAP_kmeans_title = "plot_UMAP_kmeans"
UMAP_gmm_title = "plot_UMAP_GMM"
UMAP_hier_title = "plot_UMAP_Hierarchical"

# assuming 3 clusters from above input
legend_labels = []
for q in range(num_of_gmm_clusters):
    temp_name = "Cluster " + str(q + 1)
    legend_labels.append(temp_name)

xlabel_dendo = "x axis of dendogram"
ylabel_dendo = "y axis of dendogram"

# 3D inputs

zlabel = "embedded3"


def reduce_dimensionality(original_data, num_of_dims):
    reducer = umap.UMAP(n_components=num_of_dims, n_neighbors=50, random_state=0)
    return reducer.fit_transform(original_data)


def plot_kmeans(original_data, embedding, name, num_clusters, xlabel, ylabel, legend_labels):
    plt.figure(figsize=(10, 10))
    kmeans = KMeans(init="random", n_clusters=num_clusters, random_state=0).fit(original_data)
    label = kmeans.fit_predict(original_data)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(embedding[label == i, 0], embedding[label == i, 1], label=i, alpha=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend(labels=legend_labels)
    plt.savefig(stem + name + ".pdf")
    plt.close()


def do_gmm_clustering(original_data, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(original_data)
    return gmm.fit_predict(original_data)


def plot_GMM(gmm_labels, embedding, name, xlabel, ylabel, legend_labels):
    plt.figure(figsize=(10, 10))

    u_labels = np.unique(gmm_labels)

    for i in u_labels:
        plt.scatter(embedding[gmm_labels == i, 0], embedding[gmm_labels == i, 1], label=i, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend(labels=legend_labels)
    plt.savefig(stem + name + ".pdf")
    plt.close()


def plot_GMM_3D(gmm_labels, embedding, name, xlabel, ylabel, zlabel, legend_labels):
    u_labels = np.unique(gmm_labels)

    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i in u_labels:
        ax.scatter(embedding[gmm_labels == i, 0], embedding[gmm_labels == i, 1], embedding[gmm_labels == i, 2],
                   label=i, alpha=0.2)

    """
    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel(ylabel)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel(zlabel)
    ax.w_zaxis.set_ticklabels([])
    """

    plt.legend(labels=legend_labels)
    plt.savefig(stem + name + ".pdf")
    # plt.show()
    plt.close()


print('umap main')
umap_2d = reduce_dimensionality(input_data, 2)

print('do GMM')
labels_for_gmm = do_gmm_clustering(input_data, num_of_gmm_clusters)

print('gmm main')
plot_GMM(labels_for_gmm, umap_2d, UMAP_gmm_title, x_label, y_label, legend_labels)

"""
print('umap main.T')
umap_2d_transposed = reduce_dimensionality(transposed_inputdata, 2)


print('kmeans main.T')
plot_kmeans(transposed_inputdata, umap_2d_transposed, "Kmeans transposed",
            num_of_kmeans_clusters, x_label, y_label,
            legend_labels)


print('umap main 3d')
umap_3D = reduce_dimensionality(input_data, 3)

print('gmm main 3d')
plot_GMM_3D(labels_for_gmm, umap_3D, UMAP_gmm_title + "3D", x_label, y_label, zlabel,
            legend_labels)
"""
