import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate age estimation model "
                                                 "using the APPA-REAL validation data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
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
    weight_file = args.weight_file

    img_size = 64
    batch_size = 32
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights("checkpoints168/weights.48-3.62.hdf5")
    dataset_root = Path(__file__).parent.joinpath("data", "utk_with_margin")

    for region in range(5):
        # for age in range(10):
            # validation = '[{}-{}]_*_{}_*.jpg'.format(age*10, age*10+9, region)
        validation = '*_*_{}_*.jpg'.format(region)
        image_paths = list(dataset_root.glob(validation))
        faces = np.empty((batch_size, img_size, img_size, 3))
        ages = []
        image_ages = []

        for i, image_path in tqdm(enumerate(image_paths)):
            faces[i % batch_size] = cv2.resize(cv2.imread(
                str(image_path), 1), (img_size, img_size))
            image_ages.append(int(image_path.name.split("_")[0]))

            if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
                results = model.predict(faces)
                ages_out = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages_out).flatten()
                ages += list(predicted_ages)
                # len(ages) can be larger than len(image_names) due to the last batch, but it's ok.
        utk_abs_error = 0.0
        for i in range(len(image_ages)):
            utk_abs_error += abs(image_ages[i]-ages[i])
        print("Region (0-White,1-Black,2-Asian,3-Indian,4-Other): ", region)
        print("Number of ages:", len(ages))
        print("Number of images: ", len(image_ages))
        print("MAE : {}".format(utk_abs_error / len(image_ages)))

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
