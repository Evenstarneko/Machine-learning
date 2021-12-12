"""
@file train_rcnn.py
@author Chang Yan (cyan13@jhu.edu)
"""
import argparse
import os
from ensemble import EnsembleWrapper
import numpy as np

parser = argparse.ArgumentParser("EnsembleTraining Script")
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--svpath", type=str, required=True)

def main(args):
    boxes = []
    images = [[], [], [], [], []]
    
    file1 = open("log_ensemble.txt","a")
    file1.close()
    
    for i in range(5):
        path = os.path.join(args.path, str(i))
        file = os.path.join(path, "cropbox.npz")
        npzfile = np.load(file)
        boxes.append(npzfile['a'])
        for j in range(npzfile['a'].shape[0]):
            file = os.path.join(path, str(i) + ".npz")
            npzfile = np.load(file)
            images[i].append(npzfile['a'])
            
    for i in range(5):
        print("*** Train: Fold "+ str(i) + " ***")
        file1.write("*** Train: Fold "+ str(i) + " ***\n")
        trian_images = []
        train_boxes = np.empty((0, 4))
        for j in range(5):
            if i != j:
                trian_images.extend(images[j])
                train_boxes = np.append(train_boxes, boxes[j])
  
        test_images = images[i]
        test_boxes = boxes[i]
  
        model = Rcnn(args.svpath, "Rcnn_fold_" + str(i))
        model.train_val(trian_images, train_boxes, test_images, test_boxes)
        print("*** Predict: Fold "+ str(i) + " ***")
        file1.write("*** Predict: Fold "+ str(i) + " ***\n")
        boxes, scores = model.predict(test_images)
        
        scores = np.sum(np.array(scores))
        avg_scores = scores / test_boxes.shape[0]
        print("*** Fold "+ str(i) + " score : " + str(avg_scores) + " ***")
        file1.write("*** Fold "+ str(i) + " score : " + str(avg_scores) + " ***\n")
        mse = np.square(np.array(boxes) - test_boxes)
        mse = np.sum(mse)
        avg_mse = mse / test_boxes.shape[0] / 4
        print("*** Fold "+ str(i) + " MSE : " + str(avg_mse) + " ***")
        file1.write("*** Fold "+ str(i) + " MSE : " + str(avg_mse) + " ***\n")
        
    file1.close()

if __name__ == "__main__":
    main()