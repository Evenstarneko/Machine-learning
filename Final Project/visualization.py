"""
@file visualization.py
@author Chang Yan (cyan13@jhu.edu)
"""
import argparse
from rcnn import Rcnn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import torch
from ensemble import EnsembleWrapper
import cv2 as cv
from PreprocessImage import PreprocessImage

parser = argparse.ArgumentParser("Visualization")
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--svpath", type=str, required=True)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("image.jpg")

def main(args):
    image = cv.imread("./Test/6.jpg")
    imageT = read_image("./Test/6.jpg")
    image = PreprocessImage.preprocess(image)
    shape = image.shape
    print(shape)
    imageT = F.resize(imageT, [shape[1], shape[2]])
    # TODO: create test_image as (1,C,H,W) numpy array
    test_image = []
    test_image.append(image[0:3])
    
    model = Rcnn(args.svpath, "Rcnn_full_feature_fold_0.pt", 50, 1)
    model.load()
    boxes, scores = model.predict(test_image)
    print(boxes)
    if (boxes[0] is not None):
        result = draw_bounding_boxes(imageT, torch.tensor([boxes[0]], dtype=torch.float), colors=["blue"], width=5)
        show(result)
        
    # imageCropped = cv.imread('./Test/cropped3.jpg')
    # imageCropped = PreprocessImage.preprocess2(imageCropped).reshape((1, 5, 224, 224))
    # model = EnsembleWrapper(args.svpath, "Ensemble_age_fold_0.pt", 12, 50, 1)
    # model.load()
    # age = model.predict(imageCropped)
    # print(age)
    # model = EnsembleWrapper(args.svpath, "Ensemble_full_sex_fold_4.pt", 2, 50, 1)
    # model.load()
    # sex = model.predict(imageCropped)
    # print(sex)
    # model = EnsembleWrapper(args.svpath, "Ensemble_full_race_fold_4.pt", 5, 50, 1)
    # model.load()
    # race = model.predict(imageCropped)
    # print(race)



if __name__ == "__main__":
    main(parser.parse_args())