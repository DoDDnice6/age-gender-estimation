#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import logging
import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from model import get_model

from utils import load_data
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
from generator_imdb_copy import ImageGenerator
logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="model name: 'ResNet50' or 'InceptionResNetV2'")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    model_name = args.model_name
    if model_name == "ResNet50":
        image_size = 224
    if model_name == "InceptionResNetV2":
        image_size = 229
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.debug("Loading data...")

    train_gen = ImageGenerator(
        "/home/dodat/Documents/python-projects/age-gender-estimation/data/wiki_crop/train", "train", image_size=image_size, batch_size=batch_size)
    val_gen = ImageGenerator("/home/dodat/Documents/python-projects/age-gender-estimation/data/wiki_crop/test",
                             "test", image_size=image_size, batch_size=batch_size)
    # print(train_gen.len())
    # print(val_gen.len())
    # print(train_gen.img_path())
    model = get_model(model_name)
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    logging.debug("Running training...")

    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    logging.debug("Saving history...")
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath(
        "history_.h5".format(model_name)), "history")


if __name__ == '__main__':
    main()
