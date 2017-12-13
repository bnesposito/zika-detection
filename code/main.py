import numpy as np
import pandas as pd
import time

import config
import process
import models

def main():
    LOGGER_LEVEL = 10
    RAW_DATA_PATH = './data/raw/'
    RAW_CSV_NAME = 'raw_data.csv'

    t0 = time.time()
    logger = config.config_logger(__name__, LOGGER_LEVEL)
    pd.set_option('display.float_format', lambda x: '{0:.2f}'.format(x))
    logger.info('Beginning execution: zika dataset')
    logger.info('Logger configured - level {0}'.format(LOGGER_LEVEL))

    logger.info('Opening CSV: {0}{1}'.format(RAW_DATA_PATH, RAW_CSV_NAME))
    raw_data = pd.read_csv(RAW_DATA_PATH + RAW_CSV_NAME)
   
    logger.info('Raw dataset description:') 
    process.basic_descriptives(raw_data)
    raw_data = process.preprocess(raw_data) 
    #print(raw_data.describe().transpose().to_string())
    #print(raw_data.head().to_string())
    #print(raw_data.info().to_string())

    y_dengue = raw_data['dengue_pcr']
    y_zika = raw_data['zika_pcr']
    y_chik = raw_data['chik_pcr']
    diseases = [y_dengue, y_zika, y_chik]
    # Check process code for further explanation of select_disease function.
    # code: 1. Dengue, 2. Zika, 3. Chik, 4. Any
    # only_one: if True, input np.nan to patients with another disease.
    y = process.select_disease(diseases, code=1, only_one=False)
    logger.info('Target var frequency: \n{0}'.format(y.value_counts()))
    logger.info('Total obs: {0}'.format(y.value_counts().sum()))

    remove_list = ['id', 'centro_pob', 'name', 'dep', 'prov', 'dist',
                   'serotipo1', 'serotipo2', 'serotipo3', 'serotipo4',
                   'dengue_pcr', 'zika_pcr', 'chik_pcr']

    X = process.remove_vars(raw_data, remove_list)
    X = process.keep_non_nan(X, y)
    y = y.dropna()

    logger.info('Features dataset:')
    process.basic_descriptives(X)

    logger.info('Estimating models')
    grid_xgboost = models.xgboost_grid(X, y, n_cv=5)
    print(grid_xgboost.best_params_)
    print(grid_xgboost.best_score_)
    hi



    lam_list = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 10, 100]
    neig_list = [1, 2, 3, 4, 5, 10, 15, 20, 30]
    cv_logit = list()
    cv_logit_lasso = list()
    cv_knn = list()
    n_cv = 4

    for lam in lam_list:
        temp_logit = models.logistic_cv(X, y, n_cv=n_cv, lam=lam)
        cv_logit.append(temp_logit)
        temp_logit_lasso = models.logistic_lasso_cv(X, y, n_cv=n_cv, lam=lam)
        cv_logit_lasso.append(temp_logit_lasso)
    for neig in neig_list:
        temp_knn = models.knn_cv(X, y, n_cv=n_cv, n_neighbors=neig)
        cv_knn.append(temp_knn)
    cv_rf = models.random_forrest_cv(X, y, n_cv=n_cv)
    cv_bernoulliNB = models.bernoulli_naive_bayes_cv(X, y, n_cv=n_cv)
    cv_NB = models.naive_bayes_cv(X, y, n_cv=n_cv)
    cv_NN = models.neural_network_cv(X, y, n_cv=n_cv)

    for i in range(len(lam_list)):
        models.report_model_output(cv_logit[i], 'Logit_{0}'.format(lam_list[i]))
        models.report_model_output(cv_logit_lasso[i], 'Logit_lasso_{0}'.format(lam_list[i]))
    for i in range(len(neig_list)):
        models.report_model_output(cv_knn[i], 'KNN_{0}'.format(neig_list[i]))

    models.report_model_output(cv_rf, 'Random Forrest')
    models.report_model_output(cv_bernoulliNB, 'BernoulliNB')
    models.report_model_output(cv_NB, 'Naive Bayes')
    models.report_model_output(cv_NN, 'Neural Network')

    #print(cv_logit)

    config.time_taken_display(t0)
    print(' ')


if __name__ == '__main__':
    main()

