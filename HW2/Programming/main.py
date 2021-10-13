""" Main file. This is the starting point for your code execution. 

You shouldn't need to change much of this code, but it's fine to as long as we
can still run your code with the arguments specified!
"""


import os
import json
import pickle
import argparse as ap

import numpy as np

import models
from data import load_data


def w1(x):
    return x


def w2(x):
    return x * x


def get_args():
    p = ap.ArgumentParser()

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                    help="Operating mode: train or test.")
    p.add_argument("--train-data", type=str, help="Training data file")
    p.add_argument("--test-data", type=str, help="Test data file")
    p.add_argument("--model-file", type=str, required=True,
                    help="Where to store and load the model parameters")
    p.add_argument("--predictions-file", type=str, 
                    help="Where to dump predictions")
    p.add_argument("--algorithm", type=str,
                   choices=['logistic', 'l2'],
                    help="The type of model to use.")
    # Model Hyperparameters
    p.add_argument("--estimating-equation", type=int, default=1,
                    help="Index of the function of the features x, used in estimating equations. See top of this file for example functions.")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Tolerance used to exit the Newton-Raphson loop.")
    return p.parse_args()


def check_args(args):
    mandatory_args = {'mode', 'model_file', 'test_data', 'train_data',
                      'predictions_file', 'algorithm', 'estimating_equation',
                      'tolerance'}

    if not mandatory_args.issubset(set(dir(args))):
        raise Exception(("You're missing essential arguments!"
                         "We need these to run your code."))

    if args.model_file is None:
        raise Exception("--model-file should be specified in either mode")

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified during training")
        if args.model_file is None:
            raise Exception("--model-file should be specified during training")
        if args.train_data is None:
            raise Exception("--train-data should be specified during training")
        elif not os.path.exists(args.train_data):
            raise Exception("data file specified by --train-data does not exist.")
    elif args.mode.lower() == "test":
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified during testing")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")
        if args.test_data is None:
            raise Exception("--test-data should be specified during testing")
        elif not os.path.exists(args.test_data):
            raise Exception("data file specified by --test-data does not exist.")
    else:
        raise Exception("Invalid mode")


def test(args):
    """ Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _, _ = load_data(args.test_data)
    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    preds = model.predict(X)
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """

    X, y, nclasses = load_data(args.train_data)

    # build the appropriate model
    if args.algorithm == 'l2':
        model = models.MCLogisticWithL2(nfeatures=X.shape[1], nclasses=nclasses)
    elif args.algorithm == 'logistic':
        model = models.MCLogistic(nfeatures=X.shape[1], nclasses=nclasses)
    else:
        raise Exception("Algorithm argument not recognized")

    # Configure estimating equation
    if args.estimating_equation == 1:
        func = w1
    else:
        func = w2

    # Fit Model
    model.fit(X=X, y=y, A=func, tol=args.tolerance)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


if __name__ == "__main__":
    ARGS = get_args()
    check_args(ARGS)

    if ARGS.mode.lower() == 'train':
        train(ARGS)
    elif ARGS.mode.lower() == 'test':
        test(ARGS)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
