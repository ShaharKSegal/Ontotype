import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import model as dm


def kfold_split_by_multiindex_level(df, k_iters, index_level=0):
    """
    NOTE: The function returns the values of the level. Cannot be used with numpy, only pandas
    :param df: dataframe to work on with multiindex
    :param k_iters: int for the k fold
    :return: (train_level_values, test_level_values) for the pandas filter
    """
    lvl_values = df.index.levels[index_level].values
    kfold = KFold(n_splits=k_iters, shuffle=True)
    for train_lvl_index, test_lvl_index in kfold.split(lvl_values):
        train_lvl, test_lvl = lvl_values[train_lvl_index], lvl_values[test_lvl_index]
        yield train_lvl, test_lvl


def get_xy_from_df(df, y_col_name):
    def get_target_values_from_df(df, target_col_name):
        return df[target_col_name].values

    def get_features_from_df(df, target_col_name):
        return df.loc[:, df.columns != target_col_name].values

    return get_features_from_df(df, y_col_name), get_target_values_from_df(df, y_col_name)


def get_train_test_split_by_level_values(df, train_level_values, test_level_values, target_col_name):
    df_train, df_test = df.loc[train_level_values, :], df.loc[test_level_values, :]
    return get_xy_from_df(df_train, target_col_name), get_xy_from_df(df_test, target_col_name)


def rf_cl_train_and_predict(X_train, y_train, X_test, return_prob):
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    return forest.predict_proba(X_test) if return_prob else forest.predict(X_test)


def rf_cl_plot(model: dm.AbstractModel, k_iters: int) -> None:
    """
    Train and test the model with a random forest classifier using k-fold and plot ROC curve accordingly
    :param model: The data model
    :param k_iters: k parameter for the k-fold
    :return: None
    """
    df = model.get_classification_data()
    print_prefix = "Interactions RandomForest Classifier"
    print(f"{print_prefix}: "
          f"Making a {k_iters}-fold training on {df.shape[0]} interations with {df.shape[1] - 1} features")
    kfold = KFold(n_splits=k_iters, shuffle=True)
    X, y = get_xy_from_df(df, model.LABEL_COL_NAME)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        print(f"{print_prefix}: Starting iteration {i+1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = rf_cl_train_and_predict(X_train, y_train, X_test, True)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc="lower right")
