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
from keras.utils import np_utils
from scipy.misc import derivative


def my_lightgbm(X_train, y_train, X_valid, y_valid, num_class):
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    eval_data = lgb.Dataset(np.array(X_valid), np.array(y_valid), reference=lgb_train)
    
    focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 1., num_class)
    eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 1., num_class)

    # LightGBM のハイパーパラメータ
    params = {
        "num_class": num_class,
        "learning_rate": 0.005,
        "verbosity": 0,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "metric" : "None",
        'feature_pre_filter': False, 
        'lambda_l1': 0.0, 
        'lambda_l2': 0.0, 
        'num_leaves': 132, 
        'feature_fraction': 0.5, 
        'bagging_fraction': 0.5045608972476101, 
        'bagging_freq': 4, 
        'min_child_samples': 20, 
        'num_iterations': 10000,
        'early_stopping_round': 80,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(params, lgb_train, valid_sets=eval_data, fobj=focal_loss, feval=eval_error)

    print("Params: ", model.params)

    return model


def optuna_lightgbm(X_train, y_train, X_valid, y_valid, num_class):
    lgb_train = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    eval_data = lgb.Dataset(np.array(X_valid), np.array(y_valid), reference=lgb_train)
    
    focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 1., num_class)
    eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 1., num_class)

    params = {
        "num_class": num_class,
        "early_stopping_round": 80,
        "learning_rate": 0.005,
        "verbosity": -1,
        "force_col_wise": True,
        "metric": "multi_logloss",
        'num_iterations': 10000,
    }
    opt = optuna_lgb.train(params, lgb_train, valid_sets=eval_data, fobj=focal_loss, feval=eval_error)

    # チューニングしたパラメータで，モデル作成
    print(f"調整したパラメータ\t{opt.params}")
    
    model = lgb.train(opt.params, lgb_train, valid_sets=eval_data, fobj=focal_loss, feval=eval_error)

    return model



def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    a,g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad.flatten('F'), hess.flatten('F')

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma, num_class):
    a,g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    return 'focal_loss', np.mean(loss), False
