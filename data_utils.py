import itertools
from typing import Dict, Hashable, Set

import mygene
import numpy as np
import pandas as pd

from goatools import obo_parser
from goatools.associations import read_ncbi_gene2go, read_gaf
from goatools.semantic import TermCounts, resnik_sim

from consts import *
from cacher import cache_func

mg = mygene.MyGeneInfo()


@cache_func(gene2symbol_path, False, not save_files, verbose_caching)
def _init_gene2symbol_dict():
    """
    Returns a dictionary of entrez gene ids to symbols using mygene.info API.
    cachable
    :return: dictionary with key: entrez gene id, value: gene symbol
    """
    genes = _gene2go.keys()
    query_result_list = []
    for genes_chunk in np.array_split(genes, max(genes.shape[0] // 1000, 1)):
        query_res = mg.querymany(genes_chunk, scopes='entrezgene', fields='entrezgene,symbol',
                                 species='human', entrezonly=True, as_dataframe=True,
                                 df_index=False, verbose=False)
        if 'notfound' in query_res.columns:
            query_res = query_res[query_res.notfound != True]
        query_result_list.append(query_res)
    df_res = pd.concat(query_result_list)
    res = dict(zip(df_res.entrezgene, df_res.symbol))
    return res


_go_dag = obo_parser.GODag(go_obo_path)
_gaf = read_gaf(gaf_path)
_termcounts = TermCounts(_go_dag, _gaf)
_gene2go = read_ncbi_gene2go(gene2go_path)
_gene2symbol = _init_gene2symbol_dict()
_symbol2gene = {symbol: gene for gene, symbol in _gene2symbol.items()}


def get_genes():
    return list(_gene2go.keys())


def get_symbols():
    return list(_gene2symbol.values())


def get_gene2go():
    return _gene2go


def get_gene2symbol(reverse=False):
    return _symbol2gene if reverse else _gene2symbol


def get_symbol2gene():
    return _symbol2gene


def get_go_dag():
    """
    Gets the GO DAG by a given path
    :return: GODag object see goatools.obo_parser.GODag for details
    """
    return _go_dag


def get_gaf():
    return _gaf


def get_uniprot2symbol(uniprot_ids: np.ndarray, drop_duplicates=True):
    """
    Returns a numpy array of  symbols of the uniprot ids given. Using mygene.info API.
    cachable
    :return: dictionary with key: entrez gene id, value: gene symbol
    """
    query_result_list = []
    for chunk in np.array_split(uniprot_ids, max(uniprot_ids.shape[0] // 1000, 1)):
        query_res = mg.querymany(chunk, scopes='uniprot', fields='symbol',
                                 species='human', entrezonly=True, as_dataframe=True,
                                 df_index=True, verbose=False)
        if 'notfound' in query_res.columns:
            query_res = query_res[query_res.notfound != True]
        query_result_list.append(query_res)
    df_res = pd.concat(query_result_list)
    df_res = df_res[df_res['symbol'].isin(get_symbols())]
    if drop_duplicates:
        df_res = df_res[~df_res.index.duplicated()]
    res = dict(zip(df_res.index, df_res['symbol']))
    return res


def convert_dict_to_indicator_df(d: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with symbols as rows and go terms as columns.
    Value cell is 1 if the term appears in the symbol's symbol2go dict, otherwise 0.
    :return: pandas DataFrame as explained above
    """
    series_dict = dict()
    df_dtype = np.int8
    for key, value in d.items():
        series_dict[key] = pd.Series(1, index=value, dtype=df_dtype)
    df = pd.DataFrame.from_dict(series_dict, orient='index', dtype=df_dtype)
    df.fillna(0, inplace=True, downcast='infer')
    df = df.astype(df_dtype)
    return df


def get_genes_semantic_similarity(gene1, gene2, symbol2go) -> float:
    """
    Calculate similarity between 2 genes using the symbol2go mapping of genes to go terms.
    Similarity is implemented as the average of all resnik go terms similarity of any go term associated with gene 1
    to any go term associated with gene 2
    :param gene1:
    :param gene2:
    :param symbol2go:
    :return:
    """
    terms = itertools.product(symbol2go[gene1], symbol2go[gene2])
    sum_sim = 0.0
    for term1, term2 in terms:
        sim = resnik_sim(term1, term2, _go_dag, _termcounts)
        sum_sim += sim if sim is not None else 0.0
    return sum_sim / (len(symbol2go[gene1]) * len(symbol2go[gene2]))
