import numpy as np
import cv2 as cv

class PreprocessImage(object):
    """description of class"""

    @classmethod
    def preprocess(cls, img):
        shape = img.shape
        resizedShape = (500, int(shape[1] * 500.0 / shape[0]))
        img = cv.resize(img, resizedShape)
        shape = img.shape
        intensity = (0.11 * img[:,:,0] + 0.59 * img[:,:,1] + 0.30 * img[:,:,2])
        intensity = intensity.reshape((shape[0], shape[1], 1))
        edge = cv.Canny(img, 100, 100, apertureSize = 3)
        # cv.imshow('img', edge)
        # cv.waitKey(0)
        edge = edge
        edge = edge.reshape((shape[0], shape[1], 1))
        result = np.append(img, intensity, axis = -1)
        result = np.append(result, edge, axis = -1)
        result = np.moveaxis(result, -1, 0)
        result[[0,2]] = result[[2,0]]
        return result.astype(np.uint8)
    
    @classmethod
    def preprocess2(cls, img):
        img = cv.resize(img, ((224, 224)))
        intensity = (0.11 * img[:,:,0] + 0.59 * img[:,:,1] + 0.30 * img[:,:,2])
        intensity = intensity.reshape((224, 224, 1))
        edge = cv.Canny(img, 100, 100, apertureSize = 3)
        # cv.imshow('img', edge)
        # cv.waitKey(0)
        edge = edge
        edge = edge.reshape((224, 224, 1))
        result = np.append(img, intensity, axis = -1)
        result = np.append(result, edge, axis = -1)
        result = np.moveaxis(result, -1, 0)
        result[[0,2]] = result[[2,0]]
        return result.astype(np.uint8)