import os
import sys
import traceback
from io import StringIO
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging

logger = logging.getLogger(__name__)  # ファイルの名前を渡す

my_token = os.environ["LINE_TOKEN"]

import tensorflow as tf
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

from MLmethods import Transformer


def send_line_notification(message):
    line_token = my_token
    endpoint = "https://notify-api.line.me/api/notify"
    message = "\n{}".format(message)
    payload = {"message": message}
    headers = {"Authorization": "Bearer {}".format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def train_test_time_split_no_obstacle(dataflame, train_ratio=0.8):
    """
    時系列を加味してデータをsplit
    """
    X = dataflame[dataflame["is_obstacle"] == 0].sort_values("date")
    train_size = int(len(X) * train_ratio)
    logger.info("split trian and test :{} (train_ratio:{})".format(X["date"][train_size], train_ratio))

    return X[0:train_size].copy().reset_index(drop=True), X[train_size : len(X)].copy().reset_index(drop=True)


def label_split_and_drop(X_df, target_name):
    """
    target_nameをYに分割して、Xから余分なカラムを削除し、numpyの形式にする
    """
    Y = X_df[target_name].values
    X = X_df.drop(["is_tansyo", "is_hukusyo", "date", "race_id"], axis=1).values
    # logger.info("train columns: {}".format(X_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).columns))
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, Y


def prepare_data_is_tansyo():
    target_name = "is_tansyo"
    final_df = pd.read_csv("csv/final_data.csv", sep=",")

    train_ratio = 0.8
    X = final_df[final_df["is_obstacle"] == 0].sort_values("date")
    train_size = int(len(X) * train_ratio)
    train_df = X[0:train_size].copy().reset_index(drop=True)
    test_df = X[train_size : len(X)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(["is_tansyo", "is_hukusyo", "date", "race_id"], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(["is_tansyo", "is_hukusyo", "date", "race_id"], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test


def prepare_data_is_hukusyo():
    target_name = "is_hukusyo"
    final_df = pd.read_csv("csv/final_data.csv", sep=",")

    train_ratio = 0.8
    X = final_df[final_df["is_obstacle"] == 0].sort_values("date")
    train_size = int(len(X) * train_ratio)
    train_df = X[0:train_size].copy().reset_index(drop=True)
    test_df = X[train_size : len(X)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(["is_tansyo", "is_hukusyo", "date", "race_id"], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(["is_tansyo", "is_hukusyo", "date", "race_id"], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test


def by_transformer(target_name):
    if target_name == "is_tansyo":
        X_train, y_train, X_test, y_test = prepare_data_is_tansyo()
    elif target_name == "is_hukusyo":
        X_train, y_train, X_test, y_test = prepare_data_is_hukusyo()
    save_model_path = "model/{}_{}_transformer_model.h5".format(OWN_FILE_NAME, target_name)
    savepltname = "model/{}_{}_transformer_model.jpg".format(OWN_FILE_NAME, target_name)
    Transformer.create_transformer_model(
        X_train,
        y_train,
        X_valid=X_test,
        y_valid=y_test,
        num_class=2,
        save_model_path=save_model_path,
        savepltname=savepltname,
        embed_dim=128,
        hopping_num=1,
        num_heads=3,
        ff_dim=128,
        dropout_rate_for_embedding=0.1,
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        epochs=100,
        batch_size=128,
        class_weight_equally=False,
    )

    model = tf.keras.models.load_model(
        save_model_path,
        compile=False,
        custom_objects={
            "TransformerBlock": Transformer.TransformerBlock,
            "TokenAndPositionEmbedding": Transformer.TokenAndPositionEmbedding,
        },
    )

    predict_proba_results = model.predict(X_test)

    # 混同行列
    predict_results = np.where(predict_proba_results > 0.5, 1, 0)  # 確率に応じて0,1に変換
    logger.info("{} confusion_matrix:\n{}\n".format(target_name, confusion_matrix(y_test, predict_results)))

    # 結果の保存のためにシリーズにする
    predict_proba_results = predict_proba_results.flatten()
    return pd.Series(data=predict_proba_results, name="predict_{}".format(target_name), dtype="float")


if __name__ == "__main__":
    try:
        formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s"  # フォーマットを定義
        logging.basicConfig(
            filename="logfile/" + OWN_FILE_NAME + ".logger.log", level=logging.INFO, format=formatter_func
        )

        is_tansyo_se = by_transformer("is_tansyo")
        is_hukusyo_se = by_transformer("is_hukusyo")

        # 結果の保存
        final_df = pd.read_csv("csv/final_data.csv", sep=",")
        _, test_df = train_test_time_split_no_obstacle(final_df)
        predicted_test_df = pd.concat([test_df, is_tansyo_se, is_hukusyo_se], axis=1)
        predicted_test_df.to_csv("predict/{}_predicted_test.csv".format(OWN_FILE_NAME), index=False)

        send_line_notification(OWN_FILE_NAME + " end!")
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t, v, tb):
            str = "\n" + str
            logger.error(str)
            send_line_notification(str)
