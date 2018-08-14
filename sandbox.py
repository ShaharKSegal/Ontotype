import matplotlib.pyplot as plt

import data_model as dm
import ppi_data as ppi
import ontotype_random_forest as rf

skip_shuffled_models = True

vidal_ppi_data_clean = ppi.VidalPPIData(True)
imex_ppi_data_clean = ppi.IMExPPIData(True)
unified_ppi_data_clean = ppi.UnifiedPPIData()
unified_ppi_data_clean.add_ppi(vidal_ppi_data_clean)
unified_ppi_data_clean.add_ppi(imex_ppi_data_clean)


k = 5
model_kwargs = {'ignore_zeros': False, 'max_go_threshold': 500}


model = dm.AssymetricGODataModel(unified_ppi_data_clean, similarity_feature=True, **model_kwargs)
model_symmetric = dm.SymetricGODataModel(unified_ppi_data_clean, similarity_feature=True, **model_kwargs)
model_no_sim = dm.AssymetricGODataModel(unified_ppi_data_clean, similarity_feature=False, **model_kwargs)
model_shuffled = dm.AssymetricGODataModel(unified_ppi_data_clean, shuffle_genes=True, similarity_feature=True, **model_kwargs)
model_symmetric_shuffled = dm.SymetricGODataModel(unified_ppi_data_clean, shuffle_genes=True, similarity_feature=True, **model_kwargs)
model_no_sim_shuffled = dm.AssymetricGODataModel(unified_ppi_data_clean, shuffle_genes=True, similarity_feature=False, **model_kwargs)



def test():
    rf_dict = dict()
    n_figure = 1

    def plot_over_list(model, model_s, key, values, pop_key=True):
        nonlocal n_figure
        n_figure += 1
        plt.figure(n_figure)
        for i, val in enumerate(values):
            rf_dict[key] = val
            plt.subplot(len(values), 2, 2 * i + 1)
            plt.title(f'{key}: {val}')
            rf.rf_cl_regular(model, k, **rf_dict)
            plt.subplot(len(values), 2, 2 * i + 2)
            plt.title(f'{key}: {val} SHUFFLED')
            rf.rf_cl_regular(model_s, k, **rf_dict)
        if pop_key:
            rf_dict.pop(key)

    plt.figure(n_figure)
    plt.subplot(3, 2, 1)
    plt.title('Base')
    rf.rf_cl_regular(model, k)
    if not skip_shuffled_models:
        plt.figure(n_figure)
        plt.subplot(3, 2, 2)
        plt.title('Base SHUFFLED')
        rf.rf_cl_regular(model_shuffled, k)

    plt.figure(n_figure)
    plt.subplot(3, 2, 3)
    plt.title('Base NO SIM')
    rf.rf_cl_regular(model_no_sim, k)

    if not skip_shuffled_models:
        plt.figure(n_figure)
        plt.subplot(3, 2, 4)
        plt.title('Base SHUFFLED NO SIM')
        rf.rf_cl_regular(model_no_sim_shuffled, k)

    plt.figure(n_figure)
    plt.subplot(3, 2, 5)
    plt.title('SYMMETRIC')
    rf.rf_cl_regular(model_symmetric, k)

    if not skip_shuffled_models:
        plt.figure(n_figure)
        plt.subplot(3, 2, 6)
        plt.title('SYMMETRIC SHUFFLED')
        rf.rf_cl_regular(model_symmetric_shuffled, k)

    # Adding arguments to random forest
    # plot_over_list(data_df, data_df_shuffled, 'min_impurity_decrease', [0.005, 0.01, 0.03])
    # plot_over_list(data_df, data_df_shuffled, 'max_depth', [7, 5, 3])
    # plot_over_list(data_df, data_df_shuffled, 'min_samples_split', [0.03, 0.05, 0.1])
    plt.show()


test()
