import pandas as pd
from pathlib import Path
import os
import pandas as pd
from pathlib import Path


def performance_tables(name,  roc_auc_df, ap_df, disc_cum_gain_df, arp_df, precision_df, rbp_df, uplift_df, ep_df,
                       n_found_df, n_found_0_1_df, n_found_0_2_df, n_found_0_3_df, n_found_0_4_df, n_found_0_5_df,
                       ep_1_3_df, ep_1_2_df, ep_2_3_df,
                       roc_auc_c_df, ap_c_df, disc_cum_gain_c_df, arp_c_df, precision_c_df, rbp_c_df, uplift_c_df,
                       ep_c_df, n_found_c_df, n_found_0_1_c_df, n_found_0_2_c_df, n_found_0_3_c_df, n_found_0_4_c_df, n_found_0_5_c_df,
                       ep_1_3_c_df,ep_1_2_c_df,ep_2_3_c_df):

    base_path = Path(__file__).parent

    df_means_auc = roc_auc_df.mean(axis=0)
    df_means_ap = ap_df.mean(axis=0)
    df_means_disc_cum_gain = disc_cum_gain_df.mean(axis=0)
    df_means_arp = arp_df.mean(axis=0)
    df_means_precision = precision_df.mean(axis=0)
    df_means_rbp = rbp_df.mean(axis=0)
    df_means_uplift = uplift_df.mean(axis=0)
    df_means_ep = ep_df.mean(axis=0)
    df_means_n_found = n_found_df.mean(axis=0)
    df_means_n_found_0_1 = n_found_0_1_df.mean(axis=0)
    df_means_n_found_0_2 = n_found_0_2_df.mean(axis=0)
    df_means_n_found_0_3 = n_found_0_3_df.mean(axis=0)
    df_means_n_found_0_4 = n_found_0_4_df.mean(axis=0)
    df_means_n_found_0_5 = n_found_0_5_df.mean(axis=0)
    df_means_ep_1_3 = ep_1_3_df.mean(axis=0)
    df_means_ep_1_2 = ep_1_2_df.mean(axis=0)
    df_means_ep_2_3 = ep_2_3_df.mean(axis=0)

    # names = name + '_ROC_AUC_average' + '.csv'
    # df_means_auc.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_ROC_AUC_all' + '.csv'
    # roc_auc_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_AP_average' + '.csv'
    df_means_ap.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_AP_all' + '.csv'
    # ap_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_DCG_average' + '.csv'
    df_means_disc_cum_gain.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_DCG_all' + '.csv'
    # disc_cum_gain_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_ARP_average' + '.csv'
    df_means_arp.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_ARP_all' + '.csv'
    # arp_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_Precision_average' + '.csv'
    df_means_precision.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_Precision_all' + '.csv'
    # precision_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    # names = name + '_RBP_average' + '.csv'
    # df_means_rbp.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_RBP_all' + '.csv'
    # rbp_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_Uplift_average' + '.csv'
    df_means_uplift.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_Uplift_all' + '.csv'
    # uplift_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_average' + '.csv'
    df_means_ep.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_all' + '.csv'
    # ep_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_average' + '.csv'
    df_means_n_found.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_all' + '.csv'
    # n_found_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_1_average' + '.csv'
    df_means_n_found_0_1.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_1_all' + '.csv'
    # n_found_0_1_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_2_average' + '.csv'
    df_means_n_found_0_2.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_2_all' + '.csv'
    # n_found_0_2_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_3_average' + '.csv'
    df_means_n_found_0_3.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_3_all' + '.csv'
    # n_found_0_3_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_4_average' + '.csv'
    df_means_n_found_0_4.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_4_all' + '.csv'
    # n_found_0_4_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_5_average' + '.csv'
    df_means_n_found_0_5.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_5_all' + '.csv'
    # n_found_0_5_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_1_3_average' + '.csv'
    df_means_ep_1_3.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_1_3_all' + '.csv'
    # ep_1_3_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_1_2_average' + '.csv'
    df_means_ep_1_2.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_1_2_all' + '.csv'
    # ep_1_2_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_2_3_average' + '.csv'
    df_means_ep_2_3.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_2_3_all' + '.csv'
    # ep_2_3_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())


    df_means_auc_c = roc_auc_c_df.mean(axis=0)
    df_means_ap_c = ap_c_df.mean(axis=0)
    df_means_disc_cum_gain_c = disc_cum_gain_c_df.mean(axis=0)
    df_means_arp_c = arp_c_df.mean(axis=0)
    df_means_precision_c = precision_c_df.mean(axis=0)
    df_means_rbp_c = rbp_c_df.mean(axis=0)
    df_means_uplift_c = uplift_c_df.mean(axis=0)
    df_means_ep_c = ep_c_df.mean(axis=0)
    df_means_n_found_c = n_found_c_df.mean(axis=0)
    df_means_n_found_0_1_c = n_found_0_1_c_df.mean(axis=0)
    df_means_n_found_0_2_c = n_found_0_2_c_df.mean(axis=0)
    df_means_n_found_0_3_c = n_found_0_3_c_df.mean(axis=0)
    df_means_n_found_0_4_c = n_found_0_4_c_df.mean(axis=0)
    df_means_n_found_0_5_c = n_found_0_5_c_df.mean(axis=0)
    df_means_ep_c_1_3 = ep_1_3_c_df.mean(axis=0)
    df_means_ep_c_1_2 = ep_1_2_c_df.mean(axis=0)
    df_means_ep_c_2_3 = ep_2_3_c_df.mean(axis=0)

    # names = name + '_ROC_AUC_c_average' + '.csv'
    # df_means_auc_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_ROC_AUC_c_all' + '.csv'
    # roc_auc_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_AP_c_average' + '.csv'
    df_means_ap_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_AP_c_all' + '.csv'
    # ap_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_DCG_c_average' + '.csv'
    df_means_disc_cum_gain_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_DCG_c_all' + '.csv'
    # disc_cum_gain_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_ARP_c_average' + '.csv'
    df_means_arp_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_ARP_c_all' + '.csv'
    # arp_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_Precision_c_average' + '.csv'
    df_means_precision_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_Precision_c_all' + '.csv'
    # precision_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    # names = name + '_RBP_c_average' + '.csv'
    # df_means_rbp_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_RBP_c_all' + '.csv'
    # rbp_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_Uplift_c_average' + '.csv'
    df_means_uplift_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_Uplift_c_all' + '.csv'
    # uplift_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_c_average' + '.csv'
    df_means_ep_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_c_all' + '.csv'
    # ep_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_c_average' + '.csv'
    df_means_n_found_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_c_all' + '.csv'
    # n_found_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_1_c_average' + '.csv'
    df_means_n_found_0_1_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_1_c_all' + '.csv'
    # n_found_0_1_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_2_c_average' + '.csv'
    df_means_n_found_0_2_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_2_c_all' + '.csv'
    # n_found_0_2_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_3_c_average' + '.csv'
    df_means_n_found_0_3_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_3_c_all' + '.csv'
    # n_found_0_3_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_4_c_average' + '.csv'
    df_means_n_found_0_4_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_4_c_all' + '.csv'
    # n_found_0_4_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_N_found_0_5_c_average' + '.csv'
    df_means_n_found_0_5_c.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_N_found_0_5_c_all' + '.csv'
    # n_found_0_5_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_1_3_c_average' + '.csv'
    df_means_ep_c_1_3.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_1_3_c_all' + '.csv'
    # ep_1_3_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_1_2_c_average' + '.csv'
    df_means_ep_c_1_2.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_1_2_c_all' + '.csv'
    # ep_1_2_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

    names = name + '_EP_2_3_c_average' + '.csv'
    df_means_ep_c_2_3.to_csv((base_path / "../../tables/tables performance" / names).resolve())
    # names = name + '_EP_2_3_c_all' + '.csv'
    # ep_2_3_c_df.to_csv((base_path / "../../tables/tables performance" / names).resolve())

