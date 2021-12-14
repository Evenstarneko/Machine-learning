import numpy as np
import cv2 as cv

def displayKeypoints(img1, img2, x, xp):
    [height, width, channels] = img1.shape

    for p1, p2 in zip(x, xp):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv.circle(img1, (int(p1[0]), int(p1[1])), 3, color, -1)
        img2 = cv.circle(img2, (int(p2[0]), int(p2[1])), 3, color, -1)

    cv.imshow('image1', img1)
    cv.imshow('image2', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()