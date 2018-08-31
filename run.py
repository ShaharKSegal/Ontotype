import argparse

import matplotlib.pyplot as plt

import model as dm
import ppi_data as ppi
import ontotype_random_forest as rf


def main():
    parser = argparse.ArgumentParser(description='Train and present ROC curve of a model with the data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', dest='data', type=str, choices=['vidal', 'imex'], required=True, nargs='+',
                        help='Specified list of data sources to use for the model. Available options are: vidal, imex')
    parser.add_argument('-m', '--model', dest='model', type=str, choices=['asym_go', 'sym_go'], required=True,
                        help='model to use for learning.'
                             '\n Available options: '
                             '\n\t asym_go - asymmetric go terms model ')
    parser.add_argument('-t', '--test', dest='full_test', action='store_true',
                        help='Test all models against a shuffled versions and plot it')
    parser.add_argument('-s', '--shuffle', '--shuffle_genes', dest='shuffle_genes', action='store_true',
                        help='Shuffle the gene2go dictionary to produce a random model.')
    parser.add_argument('-sf', '--similarity_feature', dest='similarity_feature', action='store_true',
                        help='Add gene similarity feature to the model.')
    parser.add_argument('-k', '--k_fold', dest='k_fold', type=int, default=5,
                        help='Defined k for the k-fold cross validation')
    parser.add_argument('-iz', '--ignore_zeros', dest='ignore_zeros', action='store_true',
                        help='Ignore zero rows produced by the model (both in training and testing).')
    parser.add_argument('-mt', '--max_go_threshold', dest='max_go_threshold', type=int, default=1000,
                        help='Max threshold for each go term which are associated with too many genes.')
    args = parser.parse_args()

    ppi_data = ppi.UnifiedPPIData()
    for d in args.data:
        if d == 'vidal':
            ppi_data.add_ppi(ppi.VidalPPIData())
        elif d == 'imex':
            ppi_data.add_ppi(ppi.IMExPPIData())
        else:
            raise NotImplementedError('data type is not supported')

    model_kwargs = {'shuffle_genes': args.shuffle_genes,
                    'similarity_feature': args.similarity_feature,
                    'ignore_zeros': args.ignore_zeros,
                    'max_go_threshold': args.max_go_threshold
                    }
    k = args.k_fold

    def plot_model(model, k, subplot, title):
        plt.subplot(*subplot)
        plt.title(title)
        rf.rf_cl_plot(model, k)

    if args.full_test:
        model_kwargs.pop('similarity_feature')
        model_kwargs.pop('shuffle_genes')
        model = dm.AsymmetricGOModel(ppi_data, shuffle_genes=False, similarity_feature=True, **model_kwargs)
        model_symmetric = dm.SymmetricGOModel(ppi_data, shuffle_genes=False, similarity_feature=True, **model_kwargs)
        model_no_sim = dm.AsymmetricGOModel(ppi_data, shuffle_genes=False, similarity_feature=False, **model_kwargs)
        model_shuffled = dm.AsymmetricGOModel(ppi_data, shuffle_genes=True, similarity_feature=True,
                                              **model_kwargs)
        model_symmetric_shuffled = dm.SymmetricGOModel(ppi_data, shuffle_genes=True, similarity_feature=True,
                                                       **model_kwargs)
        model_no_sim_shuffled = dm.AsymmetricGOModel(ppi_data, shuffle_genes=True, similarity_feature=False,
                                                     **model_kwargs)

        plot_model(model, k, [3, 2, 1], 'Base')
        plot_model(model_shuffled, k, [3, 2, 2], 'Base SHUFFLED')
        plot_model(model_no_sim, k, [3, 2, 3], 'Base NO SIM')
        plot_model(model_no_sim_shuffled, k, [3, 2, 4], 'Base SHUFFLED NO SIM')
        plot_model(model_symmetric, k, [3, 2, 5], 'SYMMETRIC')
        plot_model(model_symmetric_shuffled, k, [3, 2, 6], 'SYMMETRIC SHUFFLED')
    else:
        model_cls = None
        model_str = args.model
        if model_str == 'asym_go':
            model_cls = dm.AsymmetricGOModel
        elif model_str == 'sym_go':
            model_cls = dm.SymmetricGOModel
        model = model_cls(ppi_data, **model_kwargs)
        plot_model(model, k, [1, 1, 1], f"Model {model_str}")

    plt.show()


if __name__ == "__main__":
    main()
