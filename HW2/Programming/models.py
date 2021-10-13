""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import math


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, A, tol):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            A: A function of the features x, used in estimating equations.
            tol: tolerance used to exit the Newton-Raphson loop.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X

class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses

class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        # TODO: Implement this!
        self.W = np.zeros((nfeatures + 1, nclasses - 1))


    def fit(self, *, X, y, A, tol):
        # TODO: Implement this!
        [numExamples, numFeatures] = np.shape(X)
        X0 = np.vstack(np.ones((X.shape[1], 1)))
        while(True):
            predictions = self.predict(X)
            jacobian = np.zeros((numFeatures + 1, 1))
            hessian = np.zeros((numFeatures + 1, numFeatures + 1))

            for i in range(numExamples):
                for j in range(numFeatures + 1):
                    jacobian[j, 0] += X0[i, j] * (y - predictions[i])

            for i in range(numFeatures + 1):
                for j in range(numFeatures + 1):
                    for k in range(numFeatures + 1):
                        hessian[i, j] += X0[k, i] * X0[k, j] * predictions[k] * (1 - predictions[k])

            Wold = np.copy(self.W)
            W = Wold - np.matmul(np.linalg.inv(hessian), jacobian)

            if np.linalg.norm(W - Wold, axis = 1) < tol :
                break


    def predict(self, X):
        # TODO: Implement this!
        X = super()._fix_test_feats(X)
        X = np.vstack(np.ones((X.shape[1], 1)))
        predictions = np.linalg.matmul(X, W)
        for i in predictions:
            i = 1 / (1 + math.exp(-i))
        return np.transpose(predictions);

class MCLogisticWithL2(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        # TODO: Implement this!
        self.W = np.zeros((nfeatures, nclasses - 1))


    def fit(self, *, X, y, A, tol):
        # TODO: Implement this!
        raise NotImplementedError()


    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        predictions = np.zeros((X.shape[0], 1), dtype=np.int)

        return predictions;