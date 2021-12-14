"""
@file train_rcnn.py
@author Chang Yan (cyan13@jhu.edu)
"""
import argparse
import os
from rcnn import Rcnn
import numpy as np

parser = argparse.ArgumentParser("RCNN Training Script")
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--svpath", type=str, required=True)

def main(args):
    boxes = []
    images = [[], [], [], [], []]
    n_s = 100
    batch = 50
    
    file1 = open("log_full_feature_rcnn.txt","a")
    
    for i in range(5):
        path = os.path.join(args.path, str(i))
        file = os.path.join(path, "cropbox.npz")
        npzfile = np.load(file)
        box = npzfile['a']
        sample = np.zeros(box.shape[0], dtype = bool)
        sample[0:n_s] = 1
        np.random.shuffle(sample)
        box = box[sample]
        boxes.append(box)
        lenth = npzfile['a'].shape[0]
        for j in range(lenth):
            if sample[j] == 1:
                file = os.path.join(path, str(j) + ".npz")
                npzfile = np.load(file)
                images[i].append(npzfile['a'].astype(float)[0:3,:,:] / 255)

    for i in range(5):
        print("*** Train: Fold "+ str(i) + " ***")
        file1.write("*** Train: Fold "+ str(i) + " ***\n")
        train_images = [] 
        train_boxes = np.empty((0, 4))
        for j in range(5):
            if i != j:
                train_images.extend(images[j])
                train_boxes = np.append(train_boxes, boxes[j], axis = 0)
  
        test_images = images[i]
        test_boxes = boxes[i]
  
        model = Rcnn(args.svpath, "Rcnn_full_feature_fold_" + str(i) + ".pt", 51, batch)
        model.train_val(train_images, train_boxes, test_images, test_boxes)
        
    file1.close()

if __name__ == "__main__":
    main(parser.parse_args())