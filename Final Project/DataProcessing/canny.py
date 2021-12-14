import cv2 as cv
import numpy as np

def canny():
    img = cv.imread('bt.000.png', 0)
    edges = cv.Canny(img, 100.0, 100.0, 5)
    return edges
