import sys

import numpy as np
import pandas as pd
import umap.plot


def make_good_colors(names):
    result = []
    for name in names:
        if 'lvent' in name or 'left_vent' in name:
            result.append(0)
        elif 'rvent' in name:
            result.append(0)
        elif 'latr' in name or 'left_atr' in name:
            result.append(2)
        elif 'ratr' in name:
            result.append(2)
        elif 'cardiac' in name:
            result.append(1)
        elif 'psoas' in name:
            result.append(4)
        elif 'vena' in name or 'aorta' in name:
            result.append(6)
        elif 'ovary' in name:
            result.append(7)
        elif 'kidney' in name:
            result.append(9)
        elif 'colon' in name:
            result.append(10)
        elif 'liver' in name:
            result.append(11)
        elif 'pancreas' in name:
            result.append(12)
        elif 'lung' in name:
            result.append(13)
        elif 'prostate' in name:
            result.append(14)
        elif 'adrenal' in name:
            result.append(17)
        elif 'cortex' in name:
            result.append(17)
        elif 'nerve' in name:
            result.append(17)
        elif 'cd4' in name or 'cd8' in name or 'NK' in name or 'bcell' in name or 'monocy' in name:
            result.append(20)
        elif 'S5_' in name:
            result.append(25)
        else:
            print(name)
    return np.asarray(result)


data = np.load(sys.argv[1])

data[np.isnan(data)] = 0
data[np.isinf(data)] = 0

print(np.shape(data))

ranges = np.amax(data, axis=0) - np.amin(data, axis=0)

print(np.shape(ranges))

threshold = np.mean(ranges)
data = data[:, ranges > threshold]

print(np.shape(data))

labels = ["cl_adenocarcinoma", "cl_bloods", "cl_cancer", "cl_cells", "cl_colon", "cl_lung", "cl_mammary",
          "cl_noncancer", "colon", "ecto", "endo", "heart", "immune", "left_atrium", "left_ventricle", "liver", "lung",
          "meso", "misc_tissues1", "misc_tissues2", "ovary", "pancreas", "psoas", "right_atrium", "right_ventricle",
          "vena_cava", "vessels"]
colors = np.asarray(np.arange(np.shape(data)[0]))

if np.shape(data)[0] != len(labels):
    labels = ["S1_bcell", "S1_cd4_plus", "S1_cd8_plus_active", "S1_cd8_plus", "S1_uw038_rvent", "S1_uw054_lvent",
              "S1_uw067_lvent", "S1_uw40_lvent", "S1_uw40_rvent", "S1_uw65_lvent", "S1_uw65_rvent", "S1_w61_ovary",
              "S1_w71_kidney", "S1_w72_ratrium", "S1_w73_pancreas", "S1_w76_desc_colon", "S1_w78_lr_lung",
              "S1_w78_lvent_inf", "S1_w78_lvent_sup", "S1_w78_ratrium", "S1_w78_ur_lung", "S1_w80_desc_colon_mucosa",
              "S1_w80_ovary", "S2_uw036_lvent", "S2_uw067_rvent", "S2_uw076_lvent", "S2_uw076_rvent", "S2_w76_rvent",
              "S2_w78_rvent_inf", "S2_w78_rvent_sup", "S3_UW036_rvent", "S3_uw68_rvent", "S3_W61_colon_mucosa",
              "S3_w61_psoas", "S3_W61_right_liver", "S3_w62_left_lung", "S3_w63_right_liver", "S3_W73_left_colon",
              "S3_w73_lvent", "S3_w73_ratrium", "S3_w76_lvent", "S3_w76_ratrium", "S3_W78_latrium", "S4_monocytes",
              "S4_NK", "S4_uw38_lvent", "S4_w61_adrenal_gland", "S4_w61_cardiac_septum", "S4_w62_pancreas",
              "S4_w72_left_colon", "S4_w73_post_vena_cava", "S4_w78_ll_lung", "S4_w80_psoas", "S5_A673", "S5_Caco2",
              "S5_Calu3", "S5_GM12878", "S5_HCT116", "S5_HepG2", "S5_HMEC", "S5_HUVEC", "S5_IMR90", "S5_K562",
              "S5_MCF10A", "S5_MCF7", "S5_OCILY7", "S5_Panc1", "S5_PC3", "S5_PC9", "S6_cd4_plus_active",
              "S6_entex_prostate", "S6_midfrontal_cortex", "S6_w61_aorta", "S6_w71_posterior_vena_cava",
              "S6_w72_left_ventricle", "S6_w73_left_atrium", "S6_w73_lower_left_lung", "S6_w73_ovary", "S6_w73_rvent",
              "S6_w76_left_lung", "S6_w80_pancreas", "S7_w61_sciatic_nerve", "S7_w72_rvent"]
    colors = make_good_colors(labels)

mapper = umap.UMAP(n_components=2, n_neighbors=5, random_state=0, metric='cosine').fit(data)
hover_data = pd.DataFrame({'index': colors, 'label': labels})
p = umap.plot.interactive(mapper, labels=colors, hover_data=hover_data, point_size=8)
umap.plot.show(p)
