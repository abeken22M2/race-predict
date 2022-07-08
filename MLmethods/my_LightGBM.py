import csv
import datetime
import glob
import os
import pickle
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as optuna_lgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_splitimport


def my_lightgbm(X_train, y_train, num_class, objective, metric):
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    X_test = X_train.copy()  ######### 修正必要
    y_test = y_train.copy()  ######### 修正必要
    eval_data = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    params = {
        "objective": objective,
        "metric": metric,
        "num_class": num_class,
        "class_weight": "balanced",
        #'verbosity': -1,
        #'feature_pre_filter': False,
        #'lambda_l1': 0.0,
        #'lambda_l2': 0.0,
        #'num_leaves': 136,
        #'feature_fraction': 0.948,
        #'bagging_fraction': 0.6223835318704922,
        #'bagging_freq': 6,
        #'min_child_samples': 20
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(
        params, lgb_train, valid_sets=eval_data, verbose_eval=False, num_boost_round=100, early_stopping_rounds=4
    )

    print("Params: ", model.params)

    return model


def optuna_lightgbm(X_train, y_train, num_class, objective, metric):
    X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(X_train, y_train, test_size=0.1)
    lgb_train = lgb.Dataset(np.array(X_train_opt), label=np.array(y_train_opt))
    reg_eval = lgb.Dataset(X_valid, y_valid)

    params = {
        "objective": objective,
        "num_class": num_class,
        "metric": metric,
        "class_weight": "balanced",
        "verbosity": -1,
    }
    opt = optuna_lgb.train(
        params,
        lgb_train,
        valid_sets=reg_eval,
        verbose_eval=0,
        num_boost_round=5  # ,
        # early_stopping_rounds = 100
    )

    # チューニングしたパラメータで，全データを使用し，モデル作成
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    print(f"調整したパラメータ\t{opt.params}")
    model = lgb.train(opt.params, lgb_train)

    return model
