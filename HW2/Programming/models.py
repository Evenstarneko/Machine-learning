""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


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
        self.W = np.zeros((nfeatures + 1, len(nclasses) - 1))


    def fit(self, *, X, y, A, tol):
        [numExamples, numFeatures] = np.shape(X)
        X = super()._fix_test_feats(X).toarray()
        X0 = np.ones((X.shape[0], 1))
        X0 = np.hstack((X0, X))
        epoch = 0
        while(True):
            predictions = self._predict(X0)
            jacobian = np.zeros((numFeatures + 1, 1))
            hessian = np.zeros((numFeatures + 1, numFeatures + 1))

            for i in range(numExamples):
                for j in range(numFeatures + 1):
                    jacobian[j, 0] += X0[i, j] * ( predictions[i] - y[i])  
            jacobian /= numExamples
            
            for i in range(numExamples):
                hessian += np.transpose(np.reshape(X0[i, :], (1, X0.shape[1]))) @ np.reshape(X0[i, :], (1, X0.shape[1])) * predictions[i] * (1 - predictions[i])   
            hessian /= numExamples
            Wold = np.copy(self.W)
            self.W = Wold - np.matmul(np.linalg.pinv(hessian), jacobian)
            
            loss = 0
            for i in range(numExamples):
                if y[i] == 1:
                    loss += - np.log(predictions[i])
                else:
                    loss += - np.log(1-predictions[i])
            loss /= numExamples
            
            print('epoch ' + str(epoch) + ': ' + str(loss))
            
            epoch += 1
            if np.linalg.norm(self.W - Wold, axis = 0) < tol :
                break

    def _predict(self, X):
        predictions = np.matmul(X, self.W)
        for i in range(predictions.shape[0]):
            predictions[i, 0] = 1 / (1 + np.exp(-predictions[i, 0]))
        predictions = np.transpose(predictions)
        predictions1 = np.zeros(predictions.shape[1])
        for i in range(predictions.shape[1]):
            predictions1[i] = predictions[0, i]
        return predictions1;
    
    def predict(self, X):
        X = super()._fix_test_feats(X).toarray()
        X0 = np.ones((X.shape[0], 1))
        X0 = np.hstack((X0, X))
        predictions = self._predict(X0)
        predictions = predictions > 0.5
        return predictions.astype(int);
    
    
        

class MCLogisticWithL2(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nfeatures + 1, len(nclasses) - 1))


    def fit(self, *, X, y, A, tol):
        [numExamples, numFeatures] = np.shape(X)
        X = super()._fix_test_feats(X).toarray()
        X0 = np.ones((X.shape[0], 1))
        X0 = np.hstack((X0, X))
        lamb = 10
        epoch = 0
        while(True):
            predictions = self._predict(X0)
            jacobian = np.zeros((numFeatures + 1, 1))
            hessian = np.zeros((numFeatures + 1, numFeatures + 1))

            for i in range(numExamples):
                for j in range(numFeatures + 1):
                    jacobian[j, 0] += X0[i, j] * ( predictions[i] - y[i])  
            jacobian[1:-1, 0] += self.W[1:-1, 0]*lamb  
            jacobian /= numExamples
            
            for i in range(numExamples):
                hessian += np.transpose(np.reshape(X0[i, :], (1, X0.shape[1]))) @ np.reshape(X0[i, :], (1, X0.shape[1])) * predictions[i] * (1 - predictions[i]) 
            I = np.identity(hessian.shape[0])
            I[0,0] = 0
            hessian += I*lamb
            hessian /= numExamples
            Wold = np.copy(self.W)
            self.W = Wold - np.matmul(np.linalg.inv(hessian), jacobian)
            
            loss = 0
            for i in range(numExamples):
                if y[i] == 1:
                    loss += - np.log(predictions[i])
                else:
                    loss += - np.log(1-predictions[i])
            loss /= numExamples
            
            print('epoch ' + str(epoch) + ': ' + str(loss))
            
            epoch += 1
            if np.linalg.norm(self.W - Wold, axis = 0) < tol :
                break

    def _predict(self, X):
        predictions = np.matmul(X, self.W)
        for i in range(predictions.shape[0]):
            predictions[i, 0] = 1 / (1 + np.exp(-predictions[i, 0]))
        predictions = np.transpose(predictions)
        predictions1 = np.zeros(predictions.shape[1])
        for i in range(predictions.shape[1]):
            predictions1[i] = predictions[0, i]
        return predictions1;
    
    def predict(self, X):
        X = super()._fix_test_feats(X).toarray()
        X0 = np.ones((X.shape[0], 1))
        X0 = np.hstack((X0, X))
        predictions = self._predict(X0)
        predictions = predictions > 0.5
        return predictions.astype(int);