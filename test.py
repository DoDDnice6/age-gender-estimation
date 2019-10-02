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
    dataset_root = Path(__file__).parent.joinpath("data", "wiki_db.mat")

    images, genders, ages, _, image_size, _ = load_data(dataset_root)
    print(images[0])

if __name__ == '__main__':
    main()
