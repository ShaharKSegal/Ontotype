import os

overwrite_files = False
save_files = True
verbose_caching = True

data_folder = os.path.join(os.getcwd(), 'data')
go_obo_path = os.path.join(data_folder, 'go-basic.obo')
gaf_path = os.path.join(data_folder, 'goa_human.gaf')
gene2go_path = os.path.join(data_folder, 'gene2go')
vidal_data_path = os.path.join(os.path.join(os.getcwd(), 'vidal_data'), '1-s2.0-S0092867415004304-mmc3.xlsx')
imex_data_path = os.path.join(os.path.join(os.getcwd(), 'IMEx_data'), 'mutations.tsv')

vidal_y2h_col = 'Y2H_score'
vidal_symbol_col = 'Symbol'
vidal_interactor_col = 'Interactor_symbol'

gene_interactor_similarity_col = 'Gene_Interactor_Similarity'

gene2symbol_path = os.path.join(data_folder, 'gene2symbol_dict.pickle')
symbol2go_path = os.path.join(data_folder, 'symbol2go_dict.pickle')
symbol2go_extended_path = os.path.join(data_folder, 'symbol2go_extended_dict.pickle')
symbol2go_mat_path = os.path.join(data_folder, 'symbol2go_mat.pickle')
go_model_symbol2go_extended_mat_path = os.path.join(data_folder, 'go_model_symbol2go_extended_mat.pickle')
learning_data_classifier_path = os.path.join(data_folder, 'learning_data_classifier_mat.pickle')
learning_data_regression_path = os.path.join(data_folder, 'learning_data_regression_mat.pickle')
vidal_gene_interactor_sim_path = os.path.join(data_folder, 'vidal_gene_interactor_sim_dict.pickle')
