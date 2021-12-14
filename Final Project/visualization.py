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

def main(args):
    image = cv.imread(args.path)
    imageT = read_image(args.path)
    image = PreprocessImage.preprocess(image)
    shape = image.shape
    imageT = F.resize(imageT, [shape[1], shape[2]])
    # TODO: create test_image as (1,C,H,W) numpy array
    test_image = []
    test_image.append(image[0:3])
    
    model = Rcnn(args.svpath, "Rcnn_feature_fold_3.pt", 50, 1)
    model.load()
    boxes, scores = model.predict(test_image)
    if (boxes[0] is not None):
        result = draw_bounding_boxes(imageT, torch.tensor([boxes[0]], dtype=torch.float), colors=["blue"], width=5)
        show(result)
        imageCropped = image[:, int(boxes[0][0]):int(boxes[0][2]), int(boxes[0][1]):int(boxes[0][3])]
        imageCropped = np.moveaxis(imageCropped, 0, -1)
        imageCropped = cv.resize(imageCropped, (224, 224))
        imageCropped = np.moveaxis(imageCropped, -1, 0)
        

        # model = EnsembleWrapper(args.svpath, "Ensemble_age_fold_1.pt", 12, 50, 1)
        # model.load()
        # age = model.predict([imageCropped])
        # model = EnsembleWrapper(args.svpath, "Ensemble_sex_fold_1.pt", 12, 50, 1)
        # model.load()
        # sex = model.predict([imageCropped])
        # model = EnsembleWrapper(args.svpath, "Ensemble_race_fold_1.pt", 12, 50, 1)
        # model.load()
        # race = model.predict([imageCropped])



if __name__ == "__main__":
    main(parser.parse_args())