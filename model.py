import copy
import pickle
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

import data_utils
from consts import *
from general_utils import lazy_init
from ppi_data import AbstractPPIData


class AbstractModel(metaclass=ABCMeta):
    """
    Abstract class for data models.
    Each new class must implement get_classification_data and get_regression_data which compute the features and values
    of the model using the ppi data in the constructor
    """
    LABEL_COL_NAME = 'label'

    @abstractmethod
    def __init__(self, ppi: AbstractPPIData):
        self._ppi_data = ppi

    @abstractmethod
    def get_regression_data(self):
        pass

    @abstractmethod
    def get_classification_data(self):
        pass


class AbstractGOModel(AbstractModel, metaclass=ABCMeta):
    """
    Abstract class for GO terms data models.
    Implements the basic features and data processing needed for each GO terms model
    Each new class must implement get_classification_data and get_regression_data which compute the features and values
    of the model using the ppi data in the constructor.
    NOTE: Currently only multiplication between gene and interactor vector is supported.
    """

    SIMILARITY_FEATURE_NAME = 'Gene_Interactor_Similarity'

    def __init__(self, ppi: AbstractPPIData, shuffle_genes=False, similarity_feature=False, ignore_zeros=True,
                 max_go_threshold=1000):
        """
        Constructor for go models
        :param ppi: ppi data for the model
        :param shuffle_genes: should perform a shuffling of the gene2go dic, each gene will be associated with the wrong
                              go terms
        :param similarity_feature: Add the similarity feature between gene and interactor
        :param ignore_zeros: ignore rows that have zeros in all of it's features
        :param max_go_threshold: define a max threshold for each go term which are associated with too many genes
        """
        super().__init__(ppi)

        self.shuffle_genes = shuffle_genes
        self.similarity_feature = similarity_feature
        self.ignore_zeros = ignore_zeros
        self.max_go_threshold = max_go_threshold

    @property
    def gene2go(self):
        return lazy_init(self, '_gene2go', self._get_gene2go)

    @property
    def symbol2go(self):
        return lazy_init(self, '_symbol2go', self._get_symbol2go)

    @property
    def symbol2go_extended(self):
        return lazy_init(self, '_symbol2go_extended', self._get_symbol2go_extended)

    @property
    def symbol2go_df(self):
        return lazy_init(self, '_symbol2go_df', self._get_symbol2go_df)

    def get_regression_data(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame with features and values (as column name defined by LABEL_COL_NAME const).
        Aggregated ppi data by gene and interactor to calculate mean value for this interaction type.
        Features are go terms and the value of each feature is the multiplication of the gene vector with the
        interactor vector. Each vector is a binary vector of associated go terms (after custom filteration described in
        other methods).
        TODO: Add support for other vector options, e.g. addition.
        :return: the data of the model to be used for learning
        """
        rows = dict()
        empty_vec = pd.Series(0, index=self.symbol2go_df.columns)
        gene_interactor_sim = self._ppi_data.compute_gene_interactor_similarity(self.symbol2go)
        ppi_group = self._ppi_data.data.groupby(self._ppi_data.PPI_DATA_KEYS, as_index=False)
        ppi_group_mean = ppi_group[self._ppi_data.PPI_DATA_VALUES].mean()
        for symbol, interactor, interaction_mean in ppi_group_mean.itertuples(index=False, name=None):
            gene_vec = self.symbol2go_df.loc[symbol] if symbol in self.symbol2go_df.index else empty_vec
            inter_vec = self.symbol2go_df.loc[interactor] if interactor in self.symbol2go_df.index else empty_vec
            go_row = gene_vec * inter_vec
            if self.ignore_zeros and go_row.sum() == 0:
                continue
            if self.similarity_feature:
                go_row[self.SIMILARITY_FEATURE_NAME] = gene_interactor_sim[(symbol, interactor)]
            go_row[self.LABEL_COL_NAME] = interaction_mean
            rows[(symbol, interactor)] = go_row
        data = pd.DataFrame.from_dict(rows, orient='index')
        data = data.astype({col: np.int16 for col in data.columns if
                            col not in (self.LABEL_COL_NAME, self.SIMILARITY_FEATURE_NAME)})
        return data

    def get_classification_data(self):
        """
        Binarize the regression data.
        See get_regression_data for detailed implementation
        :return: the data of the model to be used for learning
        """
        def calc_threshold(y2h_score):
            return 1 if y2h_score > 0.5 else 0

        data = self.get_regression_data()
        data[self.LABEL_COL_NAME] = data[self.LABEL_COL_NAME].map(calc_threshold)
        data = data.astype({self.LABEL_COL_NAME: np.int8})
        return data

    def _get_gene2go(self):
        gene2go = data_utils.get_gene2go()
        if self.shuffle_genes:
            genes = list(gene2go.keys())
            genes_shuffled = genes.copy()
            random.shuffle(genes_shuffled)
            gene2go_shuffled = {gene_shuffled: (gene, gene2go[gene])
                                for gene, gene_shuffled in zip(genes, genes_shuffled)}
            gene2go = {gene_shuffled: go_terms for gene_shuffled, (gene, go_terms) in gene2go_shuffled.items()}
        return gene2go

    def _get_symbol2go(self):
        """
        Converts the gene2go dict to symbol to go terms dict
        :param extend_terms: look through all go DAG and add all ancestors of symbol.
        :return: symbol2go dict (str -> Set[str])
        """
        gene2symbol = data_utils.get_gene2symbol()
        symbol2go = {}
        for gene, go_terms in self.gene2go.items():
            symbol2go[gene2symbol[gene]] = go_terms
        return symbol2go

    def _get_symbol2go_extended(self):
        """
        Create an extended symbol2go dict, which adds all ancestors of the go terms in the set (using the go_obo file).
        :param symbol2go: the original symbol2go dict
        :return: extended symbol2go dict with the parent via the go obo file.
        """
        symbol2go_extended = copy.deepcopy(self.symbol2go)
        go_dag = data_utils.get_go_dag()
        for symbol in self.symbol2go.keys():
            for go_term in self.symbol2go[symbol]:
                symbol2go_extended[symbol] = symbol2go_extended.get(symbol, set()) | go_dag[go_term].get_all_parents()
        return symbol2go_extended

    @abstractmethod
    def _get_symbol2go_df(self):
        pass

    def _symbol2go_extended_to_df(self):
        cache_file = go_model_symbol2go_extended_mat_path
        if not self.shuffle_genes and os.path.exists(cache_file):
            with open(cache_file, 'rb') as cachehandle:
                return pickle.load(cachehandle)
        res = data_utils.convert_dict_to_indicator_df(self.symbol2go_extended)
        # write to cache file
        if not self.shuffle_genes:
            with open(cache_file, 'wb') as cachehandle:
                pickle.dump(res, cachehandle)
        return res

    def _symbol2go_df_max_threshold_filter(self, symbol2go_df) -> pd.DataFrame:
        symbol2go_sum = symbol2go_df.sum(axis=0)
        max_terms = symbol2go_sum[symbol2go_sum <= self.max_go_threshold].index
        symbol2go_df = symbol2go_df.filter(items=max_terms, axis='columns')
        return symbol2go_df


class AsymmetricGOModel(AbstractGOModel):
    """
    Asymmetric go terms model. (the asymmetry is on genes to interactors)
    Filters go terms which doesn't appear in any (gene, interactor) tuple.
    """
    def _get_symbol2go_df(self):
        symbol2go_df_full = self._symbol2go_extended_to_df()
        symbol2go_df_full = self._symbol2go_df_max_threshold_filter(symbol2go_df_full)
        symbol2go_filtered = self._filter_symbol2go(self.symbol2go_extended)
        symbol2go_df_filtered = data_utils.convert_dict_to_indicator_df(symbol2go_filtered)
        columns = symbol2go_df_filtered.columns.intersection(symbol2go_df_full.columns)
        symbol2go_df = symbol2go_df_filtered.filter(items=columns, axis='columns')
        return symbol2go_df

    def _filter_symbol2go(self, symbol2go):
        """
        Filters an existing symbol2go dict to only contain symbols in gene2interactors
        and also only go terms that appears both in a gene and one of it's interactors
        :return: symbol2go filtered by gene2interactors as explained above
        """
        new_symbol2go = dict()
        gene2interactors = self._ppi_data.gene2interactors
        for gene in gene2interactors.keys():
            # skip genes not in the original dict
            if gene not in symbol2go:
                continue
            for interactor in gene2interactors[gene]:
                # skip interactors not in the original dict
                if interactor not in symbol2go:
                    continue
                # add GO terms that are both in the gene terms and the interactor
                gene_intractor_intersect = symbol2go[gene] & symbol2go[interactor]
                new_symbol2go[gene] = new_symbol2go.get(gene, set()) | gene_intractor_intersect
                new_symbol2go[interactor] = new_symbol2go.get(interactor, set()) | gene_intractor_intersect
        return new_symbol2go


class SymmetricGOModel(AbstractGOModel):
    """
    Symmetric go terms model.
    Filters go terms which doesn't appear in any two genes (could be 2 genes, 2 interactors or (gene, interactor).
    """
    def _get_symbol2go_df(self):
        symbol2go_df_full = self._symbol2go_extended_to_df()
        symbol2go_df_full = self._symbol2go_df_max_threshold_filter(symbol2go_df_full)

        min_threshold = 2  # at least 2 genes/interactors in ppi have the go term
        gene_list = np.unique(self._ppi_data.data[self._ppi_data.PPI_DATA_KEYS].values.flatten())
        symbol2go_df_filtered = symbol2go_df_full.filter(items=gene_list, axis='index')
        symbol2go_filtered_sum = symbol2go_df_filtered.sum(axis=0)

        min_terms = symbol2go_filtered_sum[symbol2go_filtered_sum >= min_threshold].index
        symbol2go_df_final = symbol2go_df_filtered.filter(items=min_terms, axis='columns')
        return symbol2go_df_final

'''
class EmbeddingModel(AbstractModel):
    def __init__(self, ppi: AbstractPPIData):
        super().__init__(ppi)
        raise NotImplementedError('Not implemented yet!')
'''