import numpy as np
import pandas as pd
import time

from sklearn.ensemble import VotingClassifier

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

    logger.info('Features dataset')
    process.basic_descriptives(X)

    logger.info('Split train test')
    X_train, X_test, y_train, y_test = models.split_data(X, y, proportion=0.4)

    logger.info('Estimating models')
    logger.info('GBM')
    grid_gbm = models.gbm_grid(X_train, y_train, n_cv=5)
    logger.info(grid_gbm.best_params_)
    logger.info('Train score: {0}'.format(grid_gbm.best_score_))
    logger.info('Test score: {0}'.format(grid_gbm.score(X_test, y_test)))

    logger.info('Logit')
    grid_logit = models.logit_grid(X_train, y_train, n_cv=5)
    logger.info(grid_logit.best_params_)
    logger.info('Train score: {0}'.format(grid_logit.best_score_))
    logger.info('Test score: {0}'.format(grid_logit.score(X_test, y_test)))

    logger.info('AdaBoost')
    grid_adaboost = models.adaboost_grid(X_train, y_train, n_cv=5)
    logger.info(grid_adaboost.best_params_)
    logger.info('Train score: {0}'.format(grid_adaboost.best_score_))
    logger.info('Test score: {0}'.format(grid_adaboost.score(X_test, y_test)))

    logger.info('Soft Voting')
    eclf = VotingClassifier(estimators=[('gbm', grid_gbm), ('logit', grid_logit),
                                        ('ada', grid_adaboost)], voting='soft')
    eclf.fit(X_train, y_train)
    y_pred = eclf.predict_proba(X_test)
    print(y_pred[:5,:])
    logger.info('Train score: {0}'.format(eclf.score(X_train, y_train)))
    logger.info('Test score: {0}'.format(eclf.score(X_test, y_test)))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()

