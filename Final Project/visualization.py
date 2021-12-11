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
    image = read_image(args.path)
    # TODO: create test_image as (1,C,H,W) numpy array
    test_image = []
    
    model = Rcnn(args.svpath, "Rcnn_fold_1")
    model.load()
    boxes, scores = model.predict(test_image)
    result = draw_bounding_boxes(image, torch.tensor(boxes, dtype=torch.float), colors=["blue"], width=5)
    show(result)
    
    # TODO: crop image to 200*200 using bounding box