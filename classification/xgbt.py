#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/8/30 18:38
@author: Silence
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

train = pd.read_csv(r'C:\Users\Silance\PycharmProjects\faceDig\classification\train_modified.csv')
test = pd.read_csv(r'C:\Users\Silance\PycharmProjects\faceDig\classification\test_modified.csv')

# print train.shape, test.shape

target='Disbursed'
IDcol = 'ID'

train['Disbursed'].value_counts()

test_results = pd.read_csv(r'C:\Users\Silance\PycharmProjects\faceDig\classification\test_results.csv')


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='auc',early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)

    print "sensitivity : %.4g" % metrics.recall_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "specificity : %.4g" % metrics.recall_score(dtrain['Disbursed'].values, dtrain_predictions)

    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    #     Predict on testing data:
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    dtest['pred'] = alg.predict(dtest[predictors])[:, 1]
    results = test_results.merge(dtest[['ID', 'predprob']], on='ID')

    tn, fp, fn, tp = metrics.confusion_matrix(results['Disbursed'], results['pred'])
    print (tn, fp, fn, tp)

    print 'AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob'])

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


if __name__ == '__main__':

    predictors = [x for x in train.columns if x not in [target, IDcol]]
    print predictors
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    modelfit(xgb1, train, test, predictors)

    # 'max_depth', 'min_child_weight'
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,min_child_weight=1, gamma=0,
                                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                                                    scale_pos_weight=1, seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    gsearch1.fit(train[predictors], train[target])

    print gsearch1.cv_results_
    print gsearch1.best_params_
    print gsearch1.best_score_


    param_test2 = {
        'max_depth': [4, 5, 6],
        'min_child_weight': [4, 5, 6]
    }
    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=2, gamma=0,
                                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                                                    scale_pos_weight=1, seed=27),
                            param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    gsearch2.fit(train[predictors], train[target])

    print gsearch2.cv_results_
    print gsearch2.best_params_
    print gsearch2.best_score_

    param_test2b = {
        'min_child_weight': [6, 8, 10, 12]
    }
    gsearch2b = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                     min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                     objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                     seed=27),
                             param_grid=param_test2b, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch2b.fit(train[predictors], train[target])

    print gsearch2b.cv_results_
    print gsearch2b.best_params_
    print gsearch2b.best_score_

    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch3.fit(train[predictors], train[target])

    print gsearch3.cv_results_
    print gsearch3.best_params_
    print gsearch3.best_score_

    predictors = [x for x in train.columns if x not in [target, IDcol]]
    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb2, train, test, predictors)

    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(train[predictors], train[target])

    print gsearch4.cv_results_
    print gsearch4.best_params_
    print gsearch4.best_score_

    param_test5 = {
        'subsample': [i / 100.0 for i in range(75, 90, 5)],
        'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)]
    }
    gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch5.fit(train[predictors], train[target])

    print gsearch5.cv_results_
    print gsearch5.best_params_
    print gsearch5.best_score_

    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                    min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch6.fit(train[predictors], train[target])

    print gsearch6.cv_results_
    print gsearch6.best_params_
    print gsearch6.best_score_

    param_test7 = {
        # 'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
        'reg_alpha': [i for i in range(6)]
    }
    gsearch7 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=4,
                                                    min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test7, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch7.fit(train[predictors], train[target])

    print gsearch7.cv_results_
    print gsearch7.best_params_
    print gsearch7.best_score_

    xgb3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=1,
        gamma=0,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=3,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb3, train, test, predictors)