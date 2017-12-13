import numpy as np
import config
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, accuracy_score

logger = config.config_logger(__name__,10)
np.random.seed(42)


def report_model_output(model_output, label):
    logger.info('{2} -- Mean: {0:.3g} -- std: {1:.3g}'.format(np.mean(model_output), np.std(model_output), label))
    return


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score


def logistic_cv(x, y, n_cv=10, lam=1):
    logit = LogisticRegression(C=lam)
    cv_score = cross_val_score(logit, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def random_forrest_cv(x, y, n_cv=10):
    RF = RandomForestClassifier()
    cv_score = cross_val_score(RF, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def logistic_lasso_cv(x, y, n_cv=10, lam=1):
    logit_lasso = LogisticRegression(C=lam, penalty='l1', solver='liblinear')
    cv_score = cross_val_score(logit_lasso, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def bernoulli_naive_bayes_cv(x, y, n_cv=10):
    bernoulli_bayes = BernoulliNB()
    cv_score = cross_val_score(bernoulli_bayes, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def naive_bayes_cv(x, y, n_cv=10):
    naive_bayes = GaussianNB()
    cv_score = cross_val_score(naive_bayes, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def neural_network_cv(x, y, n_cv=10):
    nn = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(100,80,10), random_state=1)
    cv_score = cross_val_score(nn, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def knn_cv(x, y, n_cv=10, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors)
    cv_score = cross_val_score(knn, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def logit_grid(x, y, n_cv=10):
    space_C = np.logspace(-3, 2, num=20)
    print(space_C)


def xgboost_grid(x, y, n_cv=10):
    space_est = range(1, 50)
    space_depth = range(1, 5)
    space_loss = ['deviance', 'exponential']
    space_criterion = ['friedman_mse', 'mse']
    param_grid = {'n_estimators': space_est, 'max_depth': space_depth,
                  'criterion': space_criterion, 'loss': space_loss}
    xgBoost = GradientBoostingClassifier()
    grid_xgboost = GridSearchCV(xgBoost, param_grid, cv=n_cv, n_jobs=4)
    grid_xgboost.fit(x, y)
    return grid_xgboost
