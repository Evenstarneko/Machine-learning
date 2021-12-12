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
    n_s = 180
    file1 = open("log_ensemble.txt","a")
    
    for i in range(5):
        path = os.path.join(args.path, str(i))
        file = os.path.join(path, "label.npz")
        npzfile = np.load(file)
        label = npzfile['a']
        sample = np.zeros(label.shape[0], dtype = bool)
        sample[0:n_s] = 1
        np.random.shuffle(sample)
        label = label[sample]
        labels.append(label)
        file = os.path.join(path, "imageCropped.npz")
        npzfile = np.load(file)
        image = npzfile['a']
        image = image[sample]
        images.append(image)
    
    class_num = [12, 2, 5]
    type_name = ["age", "sex", "race"]        
    for k in range(3):
        for i in range(5):
            print("*** Train: "+ type_name[k] +" Fold "+ str(i) + " ***")
            file1.write("*** Train: "+ type_name[k] +" Fold "+ str(i) + " ***\n")
            train_images = np.empty((0, 5, 224, 224))
            train_labels = np.empty(0)
            for j in range(5):
                if i != j:
                    train_images = np.append(train_images, images[j], axis = 0)
                    train_labels = np.append(train_labels, labels[j][:,k], axis = 0)
  
            test_images = images[i]
            test_labels = labels[i][:,k]
  
            model = EnsembleWrapper(args.svpath, "Ensemble_"+ type_name[k] +"_fold_" + str(i) + ".pt", class_num[k], 50, 5)
            model.train_val(train_images, train_labels, test_images, test_labels)
        
    file1.close()

if __name__ == "__main__":
    main(parser.parse_args())