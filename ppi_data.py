import re

import pandas as pd

from abc import ABC, ABCMeta, abstractmethod
from typing import List, Dict, Set

import data_utils
from consts import *
from general_utils import lazy_init


class AbstractPPIData(metaclass=ABCMeta):
    PPI_DATA_KEYS = ["symbol", "interactor"]
    PPI_DATA_VALUES = ["interaction"]
    PPI_DATA_COL_LST = [*PPI_DATA_KEYS, *PPI_DATA_VALUES]

    @abstractmethod
    def __init__(self):
        self._data, self._gene2interactor = None, None

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with columns as in the PPI_DATA_COL_LST constant
        :return: pd.DataFrame
        """
        return lazy_init(self, '_data', self._get_data)

    @property
    def gene2interactors(self) -> Dict[str, Set[str]]:
        """
        Returns a dictionary mapping genes to interactors in the ppi data
        :return: Dict[str, Set[str]]
        """
        return lazy_init(self, '_gene2interactor', self._get_gene2interactors)

    def compute_gene_interactor_similarity(self, symbol2go):
        gene_interactor_sim = dict()
        for gene, interactors in self.gene2interactors.items():
            for interactor in interactors:
                similarity = data_utils.get_genes_semantic_similarity(gene, interactor, symbol2go)
                gene_interactor_sim[(gene, interactor)] = similarity
        return gene_interactor_sim

    def refresh_data(self):
        """
        Use in case some meta parameters were changed
        :return: None
        """
        self._data, self._gene2interactor = None, None

    def _get_data(self):
        data = self._get_data_impl()
        return self._standardize_column_names(data)

    @abstractmethod
    def _get_data_impl(self):
        pass

    def _get_gene2interactors(self):
        gene2interactor = dict()
        for symbol, interactor, interaction in self.data.itertuples(index=False, name=None):
            gene2interactor[symbol] = gene2interactor.get(symbol, set()) | {interactor}
        return gene2interactor

    @classmethod
    @abstractmethod
    def _get_column_mapping(cls):
        pass

    @classmethod
    def _standardize_column_names(cls, data: pd.DataFrame):
        """
        Alters column names of the data to match the convension of all PPIData objects
        :param data: data with column names not necessarily matching the standard
        :return: standard data
        """
        return data.rename(cls._get_column_mapping(), axis='columns')[cls.PPI_DATA_COL_LST].reset_index(drop=True)


class VidalPPIData(AbstractPPIData):
    _vidal_interaction_col = 'Y2H_score'
    _vidal_symbol_col = 'Symbol'
    _vidal_interactor_col = 'Interactor_symbol'

    def __init__(self, ignore_pseudo_wt_mutations=False):
        raw_data = pd.read_excel(vidal_data_path)
        self._raw_data = raw_data
        self.ignore_pseudo_wt_mutations = ignore_pseudo_wt_mutations

    def _get_data_impl(self):
        data = self._raw_data[self._raw_data['Category'] == "Disease mutation"]  # filter non-disease data
        symbols = data_utils.get_symbols()
        data = data[data[self._vidal_symbol_col].isin(symbols)]  # filter unidentified symbols
        data = data[data[self._vidal_interactor_col].isin(symbols)]  # filter unidentified interactors
        if self.ignore_pseudo_wt_mutations:
            grp = data.groupby('Allele_ID')
            data = grp.filter(lambda g: g[self._vidal_interaction_col].mean() != 1)
        return data

    @classmethod
    def _get_column_mapping(cls):
        return dict(zip([cls._vidal_symbol_col, cls._vidal_interactor_col, cls._vidal_interaction_col],
                        cls.PPI_DATA_COL_LST))


class IMExPPIData(AbstractPPIData):
    _imex_symbol_col = 'Affected protein symbol'
    _imex_interactor_col = 'Interaction participants'
    _imex_interaction_col = 'Feature type'
    _imex_human_str = "9606 - Homo sapiens"

    def __init__(self, ignore_pseudo_wt_mutations=True, ignore_pubmed_ids: List = None):
        raw_data = pd.read_csv(imex_data_path, sep='\t', header=0, encoding='utf8', engine='python')
        self._raw_data = raw_data
        self.ignore_pseudo_wt_mutations = ignore_pseudo_wt_mutations
        self.ignore_pubmed_ids = ignore_pubmed_ids

    def _get_data_impl(self):
        zero_label = "mutation with no effect(MI:2226)"
        mutation_types = ["mutation disrupting(MI:0573)", "mutation disrupting rate(MI:1129)",
                          "mutation disrupting strength(MI:1128)", zero_label]
        symbols = data_utils.get_symbols()

        # filter non-human data
        data = self._raw_data[self._raw_data['Affected protein organism'] == self._imex_human_str]
        # filter by mutation types (interactions) and symbols
        data = data[data[self._imex_symbol_col].isin(symbols) & data[self._imex_interaction_col].isin(mutation_types)]
        # split rows by interactors and filter interactors
        data = self._extract_and_filter_imex_interactors(data)
        # interactors are in uniprotkb ids, need to convert to gene symbols
        interactors_uniprot_ids = data[self._imex_interactor_col].unique()
        uniprot2symbol = data_utils.get_uniprot2symbol(interactors_uniprot_ids)
        data = data[data[self._imex_interactor_col].isin(list(uniprot2symbol.keys()))]
        data[self._imex_interactor_col] = data[self._imex_interactor_col].replace(uniprot2symbol)
        # binarize interaction
        data[self._imex_interaction_col] = data[self._imex_interaction_col].replace(mutation_types,
                                                                                    [int(v != zero_label) for v in
                                                                                     mutation_types])
        # filter duplicated rows - symbol + 'Feature short label' is the mutation identifier
        mutation_identifier_cols = [self._imex_symbol_col, 'Feature short label']
        identifier_cols = [*mutation_identifier_cols, self._imex_interactor_col, self._imex_interaction_col]
        data = data.drop_duplicates(subset=identifier_cols)
        # remove mutations with no affect on all of it's interactions
        if self.ignore_pseudo_wt_mutations:
            grp = data.groupby(mutation_identifier_cols)
            data = grp.filter(lambda g: g[self._imex_interaction_col].mean() != 1)
        if isinstance(self.ignore_pubmed_ids, List):
            data = data[~data['PubMedID'].isin(self.ignore_pubmed_ids)]
        return data

    @classmethod
    def _get_column_mapping(cls):
        return dict(zip([cls._imex_symbol_col, cls._imex_interactor_col, cls._imex_interaction_col],
                        cls.PPI_DATA_COL_LST))

    @staticmethod
    def _extract_and_filter_imex_interactors(raw_data: pd.DataFrame):
        """
        IMEx data concatenate all participants of the interaction under a single column (including the mutated protein).
        We are only interested in interaction between the mutated protein and a different WT protein.
        Thus, we filter interactions without exactly 2 participants and one extract the one which is not the same as the
        mutated protein.
        :param raw_data: IMEx data with the concatenate participants (as a pandas DataFrame)
        :return: A new pandas DataFrame with the interactor extracted and filtered
        """
        rows = []
        imex_uniprot_prefix = "uniprotkb:"
        imex_protein_ontotype_id = "MI:0326"
        for _, row in raw_data.iterrows():
            split_row = row[IMExPPIData._imex_interactor_col].split(';')
            # extract uniprot id of the mutated protein
            mutated_protein_search = re.search(f'{imex_uniprot_prefix}(.*)', row['Affected protein AC'])
            if mutated_protein_search is None:
                continue
            mutated_protein = mutated_protein_search.group(1)
            # skip non-simple interactions (keep only interactions with exactly 2 proteins)
            if len(split_row) != 2:
                continue
            for s in split_row:
                # filter out non-protein or non-human protein interactors
                if imex_protein_ontotype_id not in s or IMExPPIData._imex_human_str not in s:
                    continue
                interactor_search = re.search(f'{imex_uniprot_prefix}(.*?)\(', s)  # extract uniprot id from string
                if interactor_search is None:
                    continue
                s = interactor_search.group(1)
                # skip the mutated protein in the interaction
                if s == mutated_protein:
                    continue
                new_row = row.to_dict()
                new_row[IMExPPIData._imex_interactor_col] = s
                rows.append(new_row)
        return pd.DataFrame(rows)


class UnifiedPPIData(AbstractPPIData):
    #class PPIDataValue(NamedTuple):
    #    source: AbstractPPIData
    #    data: pd.DataFrame

    def __init__(self):
        super().__init__()
        self._update_ppi = True
        self._ppi_dict = {}

    def add_ppi(self, ppi: AbstractPPIData):
        if type(ppi) in self._ppi_dict:
            raise KeyError("ppi type already exists in dict")
        self._ppi_dict[type(ppi)] = ppi
        if VidalPPIData in self._ppi_dict and IMExPPIData in self._ppi_dict:
            imex_ppi = self._ppi_dict[IMExPPIData]
            vidal_pubmed_id = 25910212
            if isinstance(imex_ppi.ignore_pubmed_ids, List):
                ignore_pubmed_ids = imex_ppi.ignore_pubmed_ids.copy()
                ignore_pubmed_ids.append(vidal_pubmed_id)
            else:
                ignore_pubmed_ids = [vidal_pubmed_id]
            new_imex_ppi = IMExPPIData(imex_ppi.ignore_pseudo_wt_mutations, ignore_pubmed_ids)
            self._ppi_dict[IMExPPIData] = new_imex_ppi
        self.refresh_data()

    def _get_data_impl(self):
        ppi_data_list = []
        for ppi in self._ppi_dict.values():
            ppi_data_list.append(ppi.data)
        return pd.concat(ppi_data_list, ignore_index=True)

    @classmethod
    def _get_column_mapping(cls):
        return dict(zip(cls.PPI_DATA_COL_LST, cls.PPI_DATA_COL_LST))
