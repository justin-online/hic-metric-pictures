import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st

input_data = np.load(sys.argv[1])
do_log = int(sys.argv[2]) == 1
col1 = int(sys.argv[3])
col2 = int(sys.argv[4])

stem = sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4]

input_data = input_data[:, [col1, col2]]

df = pd.DataFrame(input_data)
df = df.dropna()
input_data = df.to_numpy()

if do_log:
    input_data[input_data < 0] = 0
    input_data[input_data > 0] = np.log(input_data[input_data > 0])


def plot_map(original_data, xlabel, ylabel):
    plt.figure(figsize=(10, 10))
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _, _, r_value, p_value, std_err = st.linregress(original_data[:, 0], original_data[:, 1])
    stats_string = "( R: " + str(r_value) + " , p: " + str(p_value) + " , err: " + str(std_err) + " )"
    plt.title(stem + " " + stats_string)
    plt.savefig(stem + ".png")
    plt.close()


plot_map(input_data, "Index " + str(col1), "Index " + str(col2))
