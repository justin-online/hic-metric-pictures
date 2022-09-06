import sys

import numpy as np
import pandas as pd
import umap.plot
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


# python3 ~/PycharmProjects/misc/plotting/hier_dendo.py

def convert(X):
    gmm = GaussianMixture(n_components=5).fit(X)
    centers = np.empty(shape=(gmm.n_components, X.shape[1]))
    for i in range(gmm.n_components):
        density = multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]
    return centers


data = np.load(sys.argv[1])
data = data[:, :-5]

# data[data > 0] = np.log(data[data > 0])
data[np.isnan(data)] = 0
data[np.isinf(data)] = 0

data = convert(data)

data = data.T

"""
for k in range(2, len(sys.argv)):
    d2 = np.load(sys.argv[k])
    d2 = d2[:, :-5]
    d2 = d2.T
    data = np.hstack((data, d2))
    del d2
"""
#
print(np.shape(data))
rows = ("cd4_plus_active,entex_prostate,HCT116_MED14_treated,HCT116_SMARCA5_untreated,midfrontal_cortex,w61_aorta,"
        "w71_posterior_vena_cava,w72_left_ventricle,w73_left_atrium,w73_lower_left_lung,w73_ovary,w73_rvent,"
        "w76_left_lung,w80_pancreas,bcell,cd4_plus,cd8_plus_active,cd8_plus,uw038_rvent,uw054_lvent,uw067_lvent,"
        "uw40_lvent,uw40_rvent,uw65_lvent,uw65_rvent,w61_ovary,w71_kidney,w72_ratrium,w73_pancreas,w76_desc_colon,"
        "w78_lr_lung,w78_lvent_inf,w78_lvent_sup,w78_ratrium,w78_ur_lung,w80_desc_colon_mucosa,w80_ovary,uw036_lvent,"
        "uw067_rvent,uw076_lvent,uw076_rvent,w76_rvent,w78_rvent_inf,w78_rvent_sup,UW036_rvent,uw68_rvent,"
        "W61_colon_mucosa,w61_psoas,W61_right_liver,w62_left_lung,w63_right_liver,W73_left_colon,w73_lvent,"
        "w73_ratrium,w76_lvent,w76_ratrium,W78_latrium,monocytes,NK,uw38_lvent,w61_adrenal_gland,w61_cardiac_septum,"
        "w62_pancreas,w72_left_colon,w73_post_vena_cava,w78_ll_lung,w80_psoas,A673,Caco2,GM12878,HCT116,HepG2,HMEC,"
        "HUVEC,IMR90,K562,MCF10A,MCF7,OCILY7,Panc1,PC3,PC9").split(",")

df = pd.DataFrame(np.array(data), index=rows)  # columns=cols,

# Z = hierarchy.linkage(df, method='single', metric=wasserstein_distance) #
# plt.figure()
# dn = hierarchy.dendrogram(Z, labels=df.index)
##plt.ylim([80, 140])
# plt.show()

# if
colors = np.asarray(np.arange(np.shape(data)[0]))
mapper = umap.UMAP().fit(data)
hover_data = pd.DataFrame({'index': colors, 'label': rows})
p = umap.plot.interactive(mapper, labels=colors, hover_data=hover_data, point_size=8)
umap.plot.show(p)
