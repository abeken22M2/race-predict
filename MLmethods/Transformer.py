import csv
import glob
import os
import pickle
import sys
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, hopping_num, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.hopping_num = hopping_num
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
            ffn_layer = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), layers.Dropout(dropout_rate)]
            )
            # 以下，hopping_num 回だけ繰り返される層
            self.attention_block_list.append(
                [attention_layer, layers.LayerNormalization(), ffn_layer, layers.LayerNormalization()]
            )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "hopping_num": self.hopping_num,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, inputs, training):
        query = inputs
        for i, layers in enumerate(self.attention_block_list):
            attention_layer, output_normalization_attn, ffn_layer, output_normalization_ffn = tuple(layers)
            with tf.name_scope(f"hopping_{i}"):
                attn_output = attention_layer(query, query, training=training)
                out1 = output_normalization_attn(query + attn_output)
                ffn_output = ffn_layer(out1, training=training)
                query = output_normalization_ffn(out1 + ffn_output)
        return query


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "maxlen": self.maxlen,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


"""
Create transformer model
embed_dim  # Embedding size for each token
num_heads  # Number of attention heads
ff_dim     # Hidden layer size in feed forward network inside transformer

本関数による作成させるモデルの読み込み方法は以下の通り
model = tf.keras.models.load_model(model_path, compile=False, custom_objects={
                                       "TransformerBlock": Transformer.TransformerBlock, "TokenAndPositionEmbedding": Transformer.TokenAndPositionEmbedding})
"""


def create_transformer_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_class,
    save_model_path,
    savepltname,
    embed_dim=128,
    hopping_num=2,
    num_heads=6,
    ff_dim=128,
    dropout_rate_for_embedding=0.1,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    epochs=100,
    batch_size=256,
    class_weight_equally=True,
):
    p = np.random.permutation(X_train.shape[0])  # ランダムなインデックス順の取得
    X_train, y_train = X_train[p], y_train[p]  # その順で全行を抽出する（＝シャッフル）

    if class_weight_equally == True:
        class_weight = decide_class_weight(y_train, num_class, Threshold_max_weight=9)
    else:
        class_weight = {}
        for i in range(0, num_class):
            class_weight[i] = 1

    inputs = layers.Input(shape=(X_train.shape[1],))
    embedding_layer = TokenAndPositionEmbedding(X_train.shape[0], X_train.shape[1], embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, hopping_num, num_heads, ff_dim, dropout_rate_for_embedding)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)  # x = layers.Flatten()(x)でもよいか
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)  # 調整可能か
    x = layers.Dropout(0.1)(x)
    output_activation = "softmax" if num_class > 2 else "sigmoid"
    outputs = layers.Dense(num_class, activation=output_activation)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    callbacks = [EarlyStopping(patience=3), ModelCheckpoint(save_model_path, save_best_only=True)]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    model.summary()

    plt_NN_train_transition(history, savepltname)


"""    
classの重みを学習データ内の件数で決める関数
Threshold_max_weight  # これ以上大きいweightにはならない
"""


def decide_class_weight(y_train, num_class, Threshold_max_weight=9):
    num_of_each_class = [0] * num_class
    for i, data in enumerate(y_train):
        num_of_each_class[data] += 1
    max_num_of_class = max(num_of_each_class)  # 最も件数の多いクラスのweightを1とし，他クラスのweightを算出する．
    class_weight = {}
    for i in range(0, len(num_of_each_class)):
        class_weight[i] = max(max_num_of_class / num_of_each_class[i], Threshold_max_weight)

    return class_weight


# 学習過程をプロットする関数
def plt_NN_train_transition(history, savepltname, nnpltdir="./nnresults"):

    plt.figure()
    plt.plot(history.epoch, history.history["accuracy"], label="accuracy")
    plt.plot(history.epoch, history.history["loss"], label="loss")
    plt.plot(history.epoch, history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.epoch, history.history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.legend()

    pltname = "/result_" + os.path.splitext(os.path.basename(savepltname))[0] + ".jpg"
    os.makedirs(nnpltdir, exist_ok=True)
    plt.savefig(nnpltdir + pltname)
    plt.close()


def test_model(X_test, y_test, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(X_test)
    preds = np.argmax(preds, 1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("Loss of test : " + str(test_loss))
    print("Accuracy     : " + str(test_acc))
    print("Precision    : " + str(precision_score(y_test, preds, average=None)))
    print("Precision_ave: " + str(precision_score(y_test, preds, average="macro")))
    print("Recall       : " + str(recall_score(y_test, preds, average=None)))
    print("Recall_ave   : " + str(recall_score(y_test, preds, average="macro")))

    # 予測結果の混合行列を表示
    matrix = confusion_matrix(y_test, preds)
    print(str(matrix))


def pred(X_pred, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(X_pred)
    # preds = np.argmax(preds, 1)

    return preds


def save_preds(preds):
    result_path = os.getcwd() + "/result/result.csv"
    os.makedirs(result_path, exist_ok=True)
    with open(result_path, mode="w") as File:
        for item in preds:
            File.write(str(item) + "\n")
    print("result保存完了")


def save_preds_with_index(preds, index):
    result_path = os.getcwd() + "/result/result.csv"
    os.makedirs(result_path, exist_ok=True)
    with open(result_path, mode="w") as File:
        for i, item in enumerate(index):
            File.write(str(item) + "," + str(preds[i]) + "\n")
    print("result保存完了")
