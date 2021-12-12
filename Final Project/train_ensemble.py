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
    labels = []
    images = []
    
    file1 = open("log_ensemble.txt","a")
    
    for i in range(5):
        path = os.path.join(args.path, str(i))
        file = os.path.join(path, "label.npz")
        npzfile = np.load(file)
        labels.append(npzfile['a'])
        file = os.path.join(path, "imageCropped.npz")
        npzfile = np.load(file)
        images.append(npzfile['a'])
    
    class_num = [12, 2, 5]
    type_name = ["age", "sex", "race"]        
    for k in range(3):
        for i in range(5):
            print("*** Train: "+ type_name[k] +" Fold "+ str(i) + " ***")
            file1.write("*** Train: "+ type_name[k] +" Fold "+ str(i) + " ***\n")
            train_images = np.empty((0, 224, 224, 5))
            train_labels = np.empty(0)
            for j in range(5):
                if i != j:
                    train_images = np.append(train_images, images[j], axis = 0)
                    train_labels = np.append(train_labels, labels[j][:,k], axis = 0)
  
            test_images = images[i]
            test_labels = labels[i][:,k]
  
            model = EnsembleWrapper(args.svpath, "Ensemble_"+ type_name[k] +"_fold_" + str(i) + ".pt", class_num[k], 100)
            model.train_val(train_images, train_labels, test_images, test_labels)
        
    file1.close()

if __name__ == "__main__":
    main(parser.parse_args())