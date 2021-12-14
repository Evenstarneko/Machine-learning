import numpy as np
import cv2 as cv

class PreprocessImage(object):
    """description of class"""

    @classmethod
    def preprocess(cls, img, normalize):
        shape = img.shape
        intensity = (0.11 * img[:,:,0] + 0.59 * img[:,:,1] + 0.30 * img[:,:,2])
        intensity = intensity.reshape((shape[0], shape[1], 1))
        edge = cv.Canny(img, 100, 100, apertureSize = 3)
        # cv.imshow('img', edge)
        # cv.waitKey(0)
        edge = edge
        edge = edge.reshape((shape[0], shape[1], 1))
        if normalize is True:
            intensity = intensity / 255.0
            edge = edge / 255.0
            img = img / 255.0
            result = np.append(img, intensity, axis = -1)
            result = np.append(result, edge, axis = -1)
            result = np.moveaxis(result, -1, 0)
            result[[0,2]] = result[[2,0]]
            return result.astype(float)
        else:
            result = np.append(img, intensity, axis = -1)
            result = np.append(result, edge, axis = -1)
            result = np.moveaxis(result, -1, 0)
            result[[0,2]] = result[[2,0]]
            return result.astype(np.uint8)