import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor
from utils import get_meta
from tqdm import tqdm
from scipy.io import loadmat


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2,
                        grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class ImageGenerator(Sequence):
    def __init__(self, db_dir, type, batch_size=32, image_size=224, minscore=1.0):
        self.minscore = minscore
        self.image_path_and_age_gender = []

        self._load_db(db_dir, type)

        self.image_num = len(self.image_path_and_age_gender)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()
        self.log = open("log.txt", "w")

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y_g = np.zeros((batch_size, 1), dtype=np.int32)
        y_a = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            image_path, age, gender = self.image_path_and_age_gender[sample_id]
            try:
                image = cv2.imread(str(image_path))
                x[i] = self.transform_image(
                    cv2.resize(image, (image_size, image_size)))
            except Exception as e:
                print(str(image_path))
                self.log.write(str(image_path)+"\n")
                # print(str(e))
            age += math.floor(np.random.randn() * 2 + 0.5)
            y_a[i] = np.clip(age, 0, 100)
            y_g[i] = np.clip(age, 0, 1)

        return x, [to_categorical(y_g, 2), to_categorical(y_a, 101)]

    def len(self):
        return self.image_num

    def img_path(self):
        return self.image_path_and_age_gender[0]

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

    # def _load_appa(self, appa_dir):
    #     appa_root = Path(appa_dir)
    #     train_image_dir = appa_root.joinpath("train")
    #     gt_train_path = appa_root.joinpath("gt_avg_train.csv")
    #     df = pd.read_csv(str(gt_train_path))

    #     for i, row in df.iterrows():
    #         age = min(100, int(row.apparent_age_avg))
    #         # age = int(row.real_age)
    #         image_path = train_image_dir.joinpath(row.file_name + "_face.jpg")

    #         if image_path.is_file():
    #             self.image_path_and_age_gender.append([str(image_path), age])

    # def _load_utk(self, utk_dir):
    #     image_dir = Path(utk_dir)

    #     for image_path in image_dir.glob("*.jpg"):
    #         # [age]_[gender]_[race]_[date&time].jpg
    #         image_name = image_path.name
    #         age = min(100, int(image_name.split("_")[0]))

    #         if image_path.is_file():
    #             self.image_path_and_age_gender.append([str(image_path), age])

    def _load_db(self, db_dir, type):
        root_path = db_dir
        mat_path = str(root_path) + "/{}.mat".format(type)
        d = loadmat(mat_path)
        ages = d["age"][0]
        genders = d["gender"][0]
        fullpaths = d["full_path"]
        for i in tqdm(range(len(ages))):
            self.image_path_and_age_gender.append(
                [str(root_path + "/"+str(fullpaths[i])).strip(), ages[i], genders[i]])


