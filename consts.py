import os

overwrite_files = False
save_files = True
verbose_caching = True

project_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_dir, 'data')

def _get_data_path(filename):
    return os.path.join(data_folder, filename)

go_obo_path = _get_data_path('go-basic.obo')
gaf_path = _get_data_path('goa_human.gaf')
gene2go_path = _get_data_path('gene2go')
vidal_data_path = _get_data_path('1-s2.0-S0092867415004304-mmc3.xlsx')
imex_data_path = _get_data_path('mutations.tsv')

vidal_y2h_col = 'Y2H_score'
vidal_symbol_col = 'Symbol'
vidal_interactor_col = 'Interactor_symbol'

gene_interactor_similarity_col = 'Gene_Interactor_Similarity'

gene2symbol_path = _get_data_path('gene2symbol_dict.pickle')
symbol2go_path = _get_data_path('symbol2go_dict.pickle')
symbol2go_extended_path = _get_data_path('symbol2go_extended_dict.pickle')
symbol2go_mat_path = _get_data_path('symbol2go_mat.pickle')
go_model_symbol2go_extended_mat_path = _get_data_path('go_model_symbol2go_extended_mat.pickle')
learning_data_classifier_path = _get_data_path('learning_data_classifier_mat.pickle')
learning_data_regression_path = _get_data_path('learning_data_regression_mat.pickle')
vidal_gene_interactor_sim_path = _get_data_path('vidal_gene_interactor_sim_dict.pickle')
