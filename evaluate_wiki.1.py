import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from utils import load_data
from sklearn.metrics import log_loss

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'


def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate age estimation model "
                                                 "using the APPA-REAL validation data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    depth = args.depth
    k = args.width

    img_size = 64
    batch_size = 32
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights("checkpoints168/weights.48-3.62.hdf5")
    dataset_root = Path(__file__).parent.joinpath("data", "imdb_db.mat.mat")

    images, genders, ages, _, image_size, _ = load_data(dataset_root)
    pre_ages = []
    faces = np.empty((batch_size, img_size, img_size, 3))

    print("number of images: ", len(images))
    for i, image in tqdm(enumerate(images)):
        faces[i % batch_size] = image

        if (i + 1) % batch_size == 0 or i == len(images) - 1:
            results = model.predict(faces)
            ages_out = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages_out).flatten()
            pre_ages += list(predicted_ages)
            # len(ages) can be larger than len(image_names) due to the last batch, but it's ok.
    wiki_abs_error = 0.0
    for i in range(len(ages)):
        wiki_abs_error += abs(pre_ages[i]-ages[i])
        # print(str(image_paths[i]))

    print("MAE : {}".format(wiki_abs_error / len(ages)))

    # name2age = {image_names[i]: ages[i] for i in range(len(image_names))}
    # df = pd.read_csv(str(gt_valid_path))

    # real_abs_error = 0.0

    # for i, row in df.iterrows():
    #     appa_abs_error += abs(name2age[row.file_name] - row.apparent_age_avg)
    #     real_abs_error += abs(name2age[row.file_name] - row.real_age)

    # print("MAE Apparent: {}".format(appa_abs_error / len(image_names)))
    # print("MAE Real: {}".format(real_abs_error / len(image_names)))


if __name__ == '__main__':
    main()
