import csv
import os
import re
import sys
import time
import traceback
from os import path

import pandas as pd
from nbformat import write
from sqlalchemy import column
from tqdm import tqdm

OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging

logger = logging.getLogger(__name__)  # ファイルの名前を渡す


def get_pedigree_data(horse_id):
    url = "https://db.netkeiba.com/horse/ped/" + str(horse_id) + "/"
    ped = pd.read_html(url)[0]
    time.sleep(0.95)

    return ped


def get_one_shaped_pedigree(horse_id):
    generations = {}
    ped = get_pedigree_data(horse_id)

    for i in reversed(range(5)):
        generations[i] = ped[i]
        ped.drop([i], axis=1, inplace=True)
        ped = ped.drop_duplicates()

    ped = pd.concat([generations[i] for i in range(5)])
    series_horse_id = pd.Series(horse_id)
    ped = pd.concat([series_horse_id, ped])

    return ped


def scrape_and_save_pedigree_updete():
    save_file_path = "csv/pedigree.csv"
    File_exist = os.path.isfile(save_file_path)

    horse_df = pd.read_csv("csv/cleaned_horse_data.csv", sep=",", index_col=0)
    horse_id_df = horse_df["horse_id"]
    horse_id_df = horse_id_df.drop_duplicates()

    if File_exist:
        already_load_data = pd.read_csv(save_file_path, sep=",")
        already_load_horse_id = already_load_data["horse_id"]

        horse_id_df = horse_id_df.to_list()
        for i in already_load_horse_id:
            horse_id_df.remove(i)
        horse_id_df = pd.Series(horse_id_df)

    else:
        with open(save_file_path, "w", encoding="UTF-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            header = ["horse_id"]
            for i in range(62):
                header.append("peds_{}".format(i))
            writer.writerow(header)

    # print(horse_id_df)

    with open(save_file_path, "a", encoding="UTF-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        for index, horse_id in tqdm(horse_id_df.iteritems()):
            ped = get_one_shaped_pedigree(horse_id)
            writer.writerow(ped)

    # 最後にファイルを確認
    got_data_df = pd.read_csv(save_file_path, sep=",")
    null_sum = got_data_df["horse_id"].isnull().sum()
    print("null_sum: {}".format(null_sum))


if __name__ == "__main__":
    try:
        scrape_and_save_pedigree_updete()
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t, v, tb):
            str = "\n" + str
            logger.error(str)
