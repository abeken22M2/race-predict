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
import pandas as pd


def my_lightgbm(X_train, y_train, X_valid, y_valid, num_class, objective, metric):
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    eval_data = lgb.Dataset(np.array(X_valid), label=np.array(y_valid), reference=lgb_train)

    # LightGBM のハイパーパラメータ
    params = {
        "objective": objective,
        "metric": metric,
        "num_class": num_class,
        "is_unbalance": True,
        "learning_rate": 0.005,
        "verbosity": 0,
        "feature_pre_filter": False,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "num_leaves": 222,
        "feature_fraction": 0.4,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "min_child_samples": 20,
        "num_iterations": 1000,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "early_stopping_round": 3,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(params, lgb_train, valid_sets=eval_data, verbose_eval=True)

    print("Params: ", model.params)

    return model


def optuna_lightgbm(X_train, y_train, X_valid, y_valid, num_class, objective, metric):
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    reg_eval = lgb.Dataset(np.array(X_valid), np.array(y_valid))

    params = {
        "objective": objective,
        "num_class": num_class,
        "metric": metric,
        "is_unbalance": True,
        "early_stopping_round": 3,
        "learning_rate": 0.005,
        "verbosity": -1,
        "force_col_wise": True,
    }
    opt = optuna_lgb.train(params, lgb_train, valid_sets=reg_eval, verbose_eval=False)

    # チューニングしたパラメータで，モデル作成
    print(f"調整したパラメータ\t{opt.params}")
    model = lgb.train(opt.params, lgb_train, valid_sets=reg_eval)

    return model
