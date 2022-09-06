import numpy as np
import pandas as pd
import umap.plot
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# python3 ~/PycharmProjects/misc/plotting/hier_dendo.py

actual_file = "/PRESENCE.npy"

do_hierarchy = False

# rows cell type ~50-80
# cols locations/loops 1000s

do_gmm = True

stem = "recap55_random_"
# stem = "recap55_general_"
# stem = "recap5k_general_"

sampling = ["S1_bcell", "S1_cd4_plus", "S1_cd8_plus", "S1_cd8_plus_active", "S1_uw038_rvent", "S1_uw054_lvent",
            "S1_uw067_lvent", "S1_uw40_lvent", "S1_uw40_rvent", "S1_uw65_lvent", "S1_uw65_rvent", "S1_w61_ovary",
            "S1_w71_kidney", "S1_w72_ratrium", "S1_w73_pancreas", "S1_w76_desc_colon", "S1_w78_lr_lung",
            "S1_w78_lvent_inf", "S1_w78_lvent_sup", "S1_w78_ratrium", "S1_w78_ur_lung", "S1_w80_desc_colon_mucosa",
            "S1_w80_ovary", "S2_uw036_lvent", "S2_uw067_rvent", "S2_uw076_lvent", "S2_uw076_rvent", "S2_w76_rvent",
            "S2_w78_rvent_inf", "S2_w78_rvent_sup", "S3_UW036_rvent", "S3_W61_colon_mucosa", "S3_W61_right_liver",
            "S3_W73_left_colon", "S3_W78_latrium", "S3_uw68_rvent", "S3_w61_psoas", "S3_w62_left_lung",
            "S3_w63_right_liver", "S3_w73_lvent", "S3_w73_ratrium", "S3_w76_lvent", "S3_w76_ratrium", "S4_NK",
            "S4_monocytes", "S4_uw38_lvent", "S4_w61_adrenal_gland", "S4_w61_cardiac_septum", "S4_w62_pancreas",
            "S4_w72_left_colon", "S4_w73_post_vena_cava", "S4_w78_ll_lung", "S4_w80_psoas", "S5_A673", "S5_Caco2",
            "S5_Calu3", "S5_GM12878", "S5_HCT116", "S5_HMEC", "S5_HUVEC", "S5_HepG2", "S5_IMR90", "S5_K562",
            "S5_MCF10A", "S5_MCF7", "S5_OCILY7", "S5_PC3", "S5_PC9", "S5_Panc1", "S6_cd4_plus_active",
            "S6_entex_prostate", "S6_midfrontal_cortex", "S6_w61_aorta", "S6_w71_posterior_vena_cava",
            "S6_w72_left_ventricle", "S6_w73_left_atrium", "S6_w73_lower_left_lung", "S6_w73_ovary", "S6_w73_rvent",
            "S6_w76_left_lung", "S6_w80_pancreas", "S7_w61_sciatic_nerve", "S7_w72_rvent"]
sampling = ["S1_bcell", "S1_cd4_plus", "S1_cd8_plus", "S1_cd8_plus_active", "S1_uw038_rvent", "S1_uw054_lvent",
            "S1_w61_ovary", "S1_w71_kidney", "S1_w72_ratrium", "S1_w73_pancreas", "S1_w76_desc_colon", "S1_w78_lr_lung",
            "S1_w78_ratrium", "S1_w78_ur_lung", "S1_w80_desc_colon_mucosa", "S1_w80_ovary", "S3_W61_colon_mucosa",
            "S3_W61_right_liver", "S3_W73_left_colon", "S3_W78_latrium", "S3_w61_psoas", "S3_w62_left_lung",
            "S3_w63_right_liver", "S3_w73_lvent", "S4_NK", "S4_monocytes", "S4_w61_adrenal_gland",
            "S4_w61_cardiac_septum", "S4_w62_pancreas", "S4_w72_left_colon", "S4_w73_post_vena_cava", "S4_w78_ll_lung",
            "S4_w80_psoas", "S5_A673", "S5_Caco2", "S5_Calu3", "S5_GM12878", "S5_HCT116", "S5_HMEC", "S5_HUVEC",
            "S5_HepG2", "S5_IMR90", "S5_K562", "S5_MCF10A", "S5_MCF7", "S5_OCILY7", "S5_PC3", "S5_PC9", "S5_Panc1",
            "S6_cd4_plus_active", "S6_entex_prostate", "S6_midfrontal_cortex", "S6_w61_aorta",
            "S6_w71_posterior_vena_cava", "S6_w73_left_atrium", "S6_w73_ovary", "S6_w80_pancreas",
            "S7_w61_sciatic_nerve"]


def convert(X):
    nn = 3
    gmm = GaussianMixture(n_components=nn).fit(X)
    centers = np.empty(shape=(nn, X.shape[1]))
    for i in range(nn):
        density = multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]
    return centers


def get_file(s):
    return np.squeeze(np.load(stem + s + actual_file))


data = get_file(sampling[0])
for sample in sampling[1:]:
    print(sample, np.shape(data))
    data = np.vstack((data, get_file(sample)))

# data[data > 0] = np.log(data[data > 0])
data[np.isnan(data)] = 0
data[np.isinf(data)] = 0

data[data < 0.02] = 0
data[data > 0] = 1

data = data.T

# cols cell type ~50-80
# rows locations/loops 1000s


if do_hierarchy:
    if do_gmm:
        data = convert(data)
    data = data.T
    # rows cell type ~50-80
    # cols locations/loops 1000s
else:
    data = data[::5]
    # cols cell type ~50-80
    # rows locations/loops 1000s

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

if do_hierarchy:
    df = pd.DataFrame(np.array(data), index=sampling)  # columns=cols,
else:
    df = pd.DataFrame(np.array(data), columns=sampling)  # columns=cols,

# Z = hierarchy.linkage(df, method='single', metric=wasserstein_distance) #
# plt.figure()
# dn = hierarchy.dendrogram(Z, labels=df.index)
##plt.ylim([80, 140])
# plt.show()

# if
colors = np.asarray(np.arange(np.shape(data)[0]))
mapper = umap.UMAP().fit(data)
if do_hierarchy:
    hover_data = pd.DataFrame({'index': colors, 'label': sampling})
else:
    hover_data = pd.DataFrame({'index': colors, 'label': colors})
p = umap.plot.interactive(mapper, labels=colors, hover_data=hover_data, point_size=8)
umap.plot.show(p)
