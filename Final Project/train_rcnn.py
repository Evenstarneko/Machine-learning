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
    for i in range(5):
        print("*** Train: Fold "+ str(i) + " ***")
        for fold in os.listdir(path):
            file_path = os.path.join(path, fold) 
            for files in os.listdir(file_path):
        
        model = Rcnn(args.svpath, "Rcnn_fold_" + str(i))
        model.train(trian_images, train_boxes)
        model.save()
        print("*** Predict: Fold "+ str(i) + " ***")
        boxes, scores = model.predict(test_images)
        
        scores = np.sum(np.array(scores))
        avg_scores = scores / test_boxes.shape[0]
        print("*** Fold "+ str(i) + " score : " + str(avg_scores) + " ***")
        mse = np.square(np.array(boxes) - test_boxes)
        mse = np.sum(mse)
        avg_mse = mse / test_boxes.shape[0] / 4
        print("*** Fold "+ str(i) + " MSE : " + str(avg_mse) + " ***")
    

if __name__ == "__main__":
    main()