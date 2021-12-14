import numpy as np
import cv2 as cv

def drawCropbox(img, pts1, pts2):
    img = cv.polylines(img, [pts1], True, (255, 255, 0), thickness = 2)
    img = cv.rectangle(img, tuple(pts2[0]), tuple(pts2[1]), (0, 0, 255), thickness = 2)
    cv.imshow('img', img)
    cv.waitKey(0)

def drawCropbox2(img, pts):
    img = cv.rectangle(img, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), thickness = 2)
    cv.imshow('img', img)
    cv.waitKey(0)