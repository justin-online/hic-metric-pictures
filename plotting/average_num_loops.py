import numpy as np

hiccups = [(74432, 'S1_bcell_hiccups.bedpe'), (39073, 'S1_cd4_plus_hiccups.bedpe'),
           (20323, 'S1_cd8_plus_active_hiccups.bedpe'), (58395, 'S1_cd8_plus_hiccups.bedpe'),
           (8988, 'S1_uw038_rvent_hiccups.bedpe'), (17064, 'S1_uw054_lvent_hiccups.bedpe'),
           (17959, 'S1_uw067_lvent_hiccups.bedpe'), (17401, 'S1_uw40_lvent_hiccups.bedpe'),
           (14124, 'S1_uw40_rvent_hiccups.bedpe'), (15617, 'S1_uw65_lvent_hiccups.bedpe'),
           (10627, 'S1_uw65_rvent_hiccups.bedpe'), (6136, 'S1_w61_ovary_hiccups.bedpe'),
           (4588, 'S1_w71_kidney_hiccups.bedpe'), (14234, 'S1_w72_ratrium_hiccups.bedpe'),
           (1263, 'S1_w73_pancreas_hiccups.bedpe'), (6900, 'S1_w76_desc_colon_hiccups.bedpe'),
           (3206, 'S1_w78_lr_lung_hiccups.bedpe'), (9786, 'S1_w78_lvent_inf_hiccups.bedpe'),
           (9892, 'S1_w78_lvent_sup_hiccups.bedpe'), (7114, 'S1_w78_ratrium_hiccups.bedpe'),
           (4970, 'S1_w78_ur_lung_hiccups.bedpe'), (5780, 'S1_w80_desc_colon_mucosa_hiccups.bedpe'),
           (8446, 'S1_w80_ovary_hiccups.bedpe'), (26152, 'S2_uw036_lvent_hiccups.bedpe'),
           (20979, 'S2_uw067_rvent_hiccups.bedpe'), (24693, 'S2_uw076_lvent_hiccups.bedpe'),
           (20526, 'S2_uw076_rvent_hiccups.bedpe'), (30699, 'S2_w76_rvent_hiccups.bedpe'),
           (40029, 'S2_w78_rvent_inf_hiccups.bedpe'), (16064, 'S2_w78_rvent_sup_hiccups.bedpe'),
           (25473, 'S3_UW036_rvent_hiccups.bedpe'), (16070, 'S3_uw68_rvent_hiccups.bedpe'),
           (18990, 'S3_W61_colon_mucosa_hiccups.bedpe'), (33982, 'S3_w61_psoas_hiccups.bedpe'),
           (5063, 'S3_W61_right_liver_hiccups.bedpe'), (7347, 'S3_w62_left_lung_hiccups.bedpe'),
           (6352, 'S3_w63_right_liver_hiccups.bedpe'), (10261, 'S3_W73_left_colon_hiccups.bedpe'),
           (22831, 'S3_w73_lvent_hiccups.bedpe'), (8789, 'S3_w73_ratrium_hiccups.bedpe'),
           (27110, 'S3_w76_lvent_hiccups.bedpe'), (13288, 'S3_w76_ratrium_hiccups.bedpe'),
           (8908, 'S3_W78_latrium_hiccups.bedpe'), (18443, 'S4_monocytes_hiccups.bedpe'),
           (38898, 'S4_NK_hiccups.bedpe'), (21092, 'S4_uw38_lvent_hiccups.bedpe'),
           (8420, 'S4_w61_adrenal_gland_hiccups.bedpe'), (36548, 'S4_w61_cardiac_septum_hiccups.bedpe'),
           (8868, 'S4_w62_pancreas_hiccups.bedpe'), (2213, 'S4_w72_left_colon_hiccups.bedpe'),
           (11650, 'S4_w73_post_vena_cava_hiccups.bedpe'), (8098, 'S4_w78_ll_lung_hiccups.bedpe'),
           (16713, 'S4_w80_psoas_hiccups.bedpe'), (37823, 'S5_A673_hiccups.bedpe'), (79635, 'S5_Caco2_hiccups.bedpe'),
           (49656, 'S5_Calu3_hiccups.bedpe'), (57480, 'S5_GM12878_hiccups.bedpe'),
           (29919, 'S5_HCT116_BRD4_treated_hiccups.bedpe'), (34447, 'S5_HCT116_BRD4_untreated_hiccups.bedpe'),
           (17790, 'S5_HCT116_CDk7_treated_hiccups.bedpe'), (49968, 'S5_HCT116_CDK7_untreated_hiccups.bedpe'),
           (6790, 'S5_HCT116_CTCF_treated_hiccups.bedpe'), (34386, 'S5_HCT116_CTCF_untreated_hiccups.bedpe'),
           (44221, 'S5_HCT116_hiccups.bedpe'), (36836, 'S5_HCT116_MED14_untreated_hiccups.bedpe'),
           (42810, 'S5_HCT116_RAD21_untreated_hiccups.bedpe'), (35512, 'S5_HCT116_SUPT16H_treated_hiccups.bedpe'),
           (29819, 'S5_HCT116_SUPT16H_untreated_hiccups.bedpe'), (54598, 'S5_HepG2_hiccups.bedpe'),
           (84303, 'S5_HMEC_hiccups.bedpe'), (106241, 'S5_HUVEC_hiccups.bedpe'), (64473, 'S5_IMR90_hiccups.bedpe'),
           (46006, 'S5_K562_hiccups.bedpe'), (63614, 'S5_MCF10A_hiccups.bedpe'), (68072, 'S5_MCF7_hiccups.bedpe'),
           (48284, 'S5_OCILY7_hiccups.bedpe'), (19936, 'S5_Panc1_hiccups.bedpe'), (74534, 'S5_PC3_hiccups.bedpe'),
           (72495, 'S5_PC9_hiccups.bedpe')]

deltas = [(23297, 'S1_bcell_delta.bedpe'), (17014, 'S1_cd4_plus_delta.bedpe'),
          (16461, 'S1_cd8_plus_active_delta.bedpe'), (14979, 'S1_cd8_plus_delta.bedpe'),
          (20669, 'S1_uw038_rvent_delta.bedpe'), (16162, 'S1_uw054_lvent_delta.bedpe'),
          (16936, 'S1_uw067_lvent_delta.bedpe'), (18509, 'S1_uw40_lvent_delta.bedpe'),
          (15728, 'S1_uw40_rvent_delta.bedpe'), (17235, 'S1_uw65_lvent_delta.bedpe'),
          (12847, 'S1_uw65_rvent_delta.bedpe'), (22372, 'S1_w61_ovary_delta.bedpe'),
          (20549, 'S1_w71_kidney_delta.bedpe'), (15287, 'S1_w72_ratrium_delta.bedpe'),
          (24344, 'S1_w73_pancreas_delta.bedpe'), (18752, 'S1_w76_desc_colon_delta.bedpe'),
          (17715, 'S1_w78_lr_lung_delta.bedpe'), (17737, 'S1_w78_lvent_inf_delta.bedpe'),
          (25582, 'S1_w78_lvent_sup_delta.bedpe'), (24115, 'S1_w78_ratrium_delta.bedpe'),
          (21053, 'S1_w78_ur_lung_delta.bedpe'), (20657, 'S1_w80_desc_colon_mucosa_delta.bedpe'),
          (21275, 'S1_w80_ovary_delta.bedpe'), (18575, 'S2_uw036_lvent_delta.bedpe'),
          (18947, 'S2_uw067_rvent_delta.bedpe'), (19333, 'S2_uw076_lvent_delta.bedpe'),
          (18980, 'S2_uw076_rvent_delta.bedpe'), (19191, 'S2_w76_rvent_delta.bedpe'),
          (21660, 'S2_w78_rvent_inf_delta.bedpe'), (16785, 'S2_w78_rvent_sup_delta.bedpe'),
          (17943, 'S3_UW036_rvent_delta.bedpe'), (17349, 'S3_uw68_rvent_delta.bedpe'),
          (20699, 'S3_W61_colon_mucosa_delta.bedpe'), (22308, 'S3_w61_psoas_delta.bedpe'),
          (26035, 'S3_W61_right_liver_delta.bedpe'), (22146, 'S3_w62_left_lung_delta.bedpe'),
          (22803, 'S3_w63_right_liver_delta.bedpe'), (17154, 'S3_W73_left_colon_delta.bedpe'),
          (17673, 'S3_w73_lvent_delta.bedpe'), (15302, 'S3_w73_ratrium_delta.bedpe'),
          (17419, 'S3_w76_lvent_delta.bedpe'), (14975, 'S3_w76_ratrium_delta.bedpe'),
          (20334, 'S3_W78_latrium_delta.bedpe'), (10749, 'S4_monocytes_delta.bedpe'), (18989, 'S4_NK_delta.bedpe'),
          (22139, 'S4_uw38_lvent_delta.bedpe'), (26537, 'S4_w61_adrenal_gland_delta.bedpe'),
          (25171, 'S4_w61_cardiac_septum_delta.bedpe'), (20900, 'S4_w62_pancreas_delta.bedpe'),
          (23669, 'S4_w72_left_colon_delta.bedpe'), (24334, 'S4_w73_post_vena_cava_delta.bedpe'),
          (16512, 'S4_w78_ll_lung_delta.bedpe'), (15530, 'S4_w80_psoas_delta.bedpe'), (31239, 'S5_A673_delta.bedpe'),
          (33642, 'S5_Caco2_delta.bedpe'), (31992, 'S5_Calu3_delta.bedpe'), (41657, 'S5_GM12878_delta.bedpe'),
          (17158, 'S5_HCT116_BRD4_treated_delta.bedpe'), (17372, 'S5_HCT116_BRD4_untreated_delta.bedpe'),
          (12679, 'S5_HCT116_CDk7_treated_delta.bedpe'), (20104, 'S5_HCT116_CDK7_untreated_delta.bedpe'),
          (5945, 'S5_HCT116_CTCF_treated_delta.bedpe'), (27926, 'S5_HCT116_CTCF_untreated_delta.bedpe'),
          (33720, 'S5_HCT116_delta.bedpe'), (25216, 'S5_HCT116_MED14_untreated_delta.bedpe'),
          (25380, 'S5_HCT116_RAD21_untreated_delta.bedpe'), (25119, 'S5_HCT116_SUPT16H_treated_delta.bedpe'),
          (25208, 'S5_HCT116_SUPT16H_untreated_delta.bedpe'), (21333, 'S5_HepG2_delta.bedpe'),
          (34521, 'S5_HMEC_delta.bedpe'), (40130, 'S5_HUVEC_delta.bedpe'), (32432, 'S5_IMR90_delta.bedpe'),
          (30257, 'S5_K562_delta.bedpe'), (27980, 'S5_MCF10A_delta.bedpe'), (36360, 'S5_MCF7_delta.bedpe'),
          (28782, 'S5_OCILY7_delta.bedpe'), (17978, 'S5_Panc1_delta.bedpe'), (28564, 'S5_PC3_delta.bedpe'),
          (26568, 'S5_PC9_delta.bedpe')]

checker = ["S5_A673", "S5_Caco2", "S5_Calu3", "S5_HepG2", "S5_HMEC", "S5_HUVEC", "S5_IMR90", "S5_MCF10A", "S5_MCF7",
           "S5_OCILY7", "S5_Panc1", "S5_PC3", "S5_PC9"]


def get_color(name):
    if 'lvent' in name or 'left_vent' in name:
        return 0
    elif 'rvent' in name:
        return 1
    elif 'latr' in name or 'left_atr' in name:
        return 2
    elif 'ratr' in name:
        return 3
    elif 'cardiac' in name:
        return 4
    elif 'psoas' in name:
        return 5
    elif 'vena' in name:
        return 6
    elif 'aorta' in name:
        return 7
    elif 'ovary' in name:
        return 8
    elif 'kidney' in name:
        return 9
    elif 'colon' in name:
        return 10
    elif 'liver' in name:
        return 11
    elif 'pancreas' in name:
        return 12
    elif 'lung' in name:
        return 13
    elif 'prostate' in name:
        return 14
    elif 'adrenal' in name:
        return 15
    elif 'cortex' in name:
        return 16
    elif 'nerve' in name:
        return 17
    elif 'cd4' in name:
        return 18
    elif 'cd8' in name:
        return 19
    elif 'NK' in name:
        return 20
    elif 'bcell' in name:
        return 21
    elif 'monocy' in name:
        return 22
    elif 'S5_' in name:
        for cc in range(len(checker)):
            if checker[cc] in name:
                return 30 + cc
    return -1


counts = {}

for (h, nm) in hiccups:
    val = get_color(nm)
    if val > -1:
        if val in counts:
            counts[val].append(h)
        else:
            counts[val] = [h]

for (h, nm) in deltas:
    val = get_color(nm)
    if val > -1:
        if val in counts:
            counts[val].append(h)
        else:
            counts[val] = [h]

mins = 0
maxes = 0
means = 0
medians = 0
for c in counts:
    mins += np.min(counts[c])
    maxes += np.max(counts[c])
    means += np.mean(counts[c])
    medians += np.median(counts[c])

print(mins, maxes, means, medians)
