""" Simple data loader :) """

import numpy as np

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

import os

def load(path, split, load_labels=True):
    if load_labels:
        labels = np.load(f'{path}/{split}.labels.npy')
    else:
        labels = None
    data = np.load(f'{path}/{split}.feats.npy')
    return data, labels

def random_transform(image: ndarray):
    random_scale = random.uniform(0.8, 1.2)
    random_rotate = random.uniform(-25, 25)
    random_translation = random.uniform(10, 10)
    transform = sk.transform.SimilarityTransform(image, scale=random_scale, rotation=random_rotate, translation=random_translation)
    transformed = sk.transform.warp(image, transform, clip=False, preserve_range=True)
    return image + transform

def random_noise(image: ndarray):
    return sk.util.random_noise(image)

if __name__ == '__main__':
    data_path = "release-data"
    train_data, train_labels = load(data_path, split="train")
    num_files_desired = 10000

    images = [os.path.join(data_path, f) for f in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, f))]

    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        image_path = random.choice(images)
        image_to_transform = sk.io.imread(image_path)
        transformed_image = random_noise(random_transform(image_to_transform))






