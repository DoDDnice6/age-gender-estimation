#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta
from shutil import copy2
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    db = args.db
    min_score = args.min_score
    validation = args.validation_split

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
        mat_path, db)

    out_genders = []
    out_ages = []
    out_fullpath = []
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        out_fullpath.append(full_path[i][0])

    print(type(out_genders))
    data_num = len(out_ages)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    out_fullpath = np.array(out_fullpath)[indexes]
    out_genders = np.array(out_genders)[indexes]
    out_ages = np.array(out_ages)[indexes]

    train_num = int(data_num*(1-validation))
    genders_train = out_genders[:train_num]
    genders_test = out_genders[train_num:]
    age_train = out_ages[:train_num]
    age_test = out_ages[train_num:]
    fullpath_train = out_fullpath[:train_num]
    fullpath_test = out_fullpath[train_num:]
    
    out_name = [name.split("/")[1] for name in out_fullpath]
    name_train=out_name[:train_num]
    name_test=out_name[train_num:]



    train_dir = Path(__file__).resolve().parent.joinpath(root_path,"train")
    train_dir.mkdir(parents=True, exist_ok=True)

    train_mat = {"full_path": np.array(name_train), "age": np.array(age_train),
                 "gender": np.array(genders_train), "db": db, "min_score": min_score}
    print(train_dir)
    
    for i in tqdm(range(len(fullpath_train))):
        copy2(root_path+fullpath_train[i], train_dir)
    scipy.io.savemat(str(train_dir), train_mat)

    test_dir = Path(__file__).resolve().parent.joinpath(root_path,"test")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_mat = {"full_path": np.array(name_test), "age": np.array(age_test),
                "gender": np.array(genders_test), "db": db, "min_score": min_score}
    
    scipy.io.savemat(str(test_dir), test_mat)
    for i in tqdm(range(len(fullpath_test))):
        copy2(root_path+fullpath_test[i], test_dir)


if __name__ == '__main__':
    main()
