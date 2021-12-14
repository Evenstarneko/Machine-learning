import numpy as np
import os
import cv2 as cv
import math

import ExtractKeypoints
import DisplayKeypoints
import DrawCropbox
from PreprocessImage import PreprocessImage

def main():
    imageCropped = [[], [], [], [], []]
    cropbox = [[], [], [], [], []]
    label = [[], [], [], [], []]

    originalImgPath = './Data/original'
    croppedImgPath = './Data/cropped'
    savePath = './Data'

    numData = 0
    for croppedImgName in os.listdir(croppedImgPath):
        imgCropped = cv.imread(os.path.join(croppedImgPath, croppedImgName))
        originalImgName = croppedImgName[0:-9]     # cutting '.chip.jpg' at the end
        imgOriginal = cv.imread(os.path.join(originalImgPath, originalImgName))
        if imgOriginal is not None:
            [pts1, pts2] = ExtractKeypoints.extractKeypoints(imgOriginal, imgCropped, 25)
            if pts1 is  None or pts2 is None and pts1.shape[0] >= 10 or pts2.shape[0] < 10:
                continue
            [H, status] = cv.findHomography(pts2, pts1, method = cv.RANSAC)
            if H is None:
                continue
            # DisplayKeypoints.displayKeypoints(imgOriginal, imgCropped, pts1, pts2)

            angle = abs(math.atan2(H[1,0], H[0,0])) * 180 / math.pi
            if angle < 5:
                originalShape = imgOriginal.shape
                croppedShape = imgCropped.shape
                if croppedShape[0] != 200 or croppedShape[1] != 200:
                    continue
                croppedPts = np.array([[0, 0, 1], [0, croppedShape[1], 1], [croppedShape[0], croppedShape[1], 1], [croppedShape[0], 0, 1]])
                originalPts = np.matmul(H, croppedPts.transpose()).transpose()
                originalPts = np.round(originalPts[:, 0:2] / originalPts[:, 2:3]).astype(np.int32)
                x1 = np.min(originalPts[:, 0]).clip(0, originalShape[1])
                x2 = np.max(originalPts[:, 0]).clip(0, originalShape[1])
                y1 = np.min(originalPts[:, 1]).clip(0, originalShape[0])
                y2 = np.max(originalPts[:, 1]).clip(0, originalShape[0])
                # DrawCropbox.drawCropbox(imgOriginal, originalPts, np.array([[x1, y1], [x2, y2]]))

                labels = originalImgName.split('_', 3)
                if not(labels[0].isdigit() and labels[1].isdigit() and labels[2].isdigit()):
                    continue
                age = int(labels[0])
                gender = int(labels[1])
                race = int(labels[2])

                ageClass = -1
                if age <= 3:
                    ageClass = 0
                elif age <= 6:
                    ageClass = 1
                elif age <= 10:
                    ageClass = 2
                elif age <= 15:
                    ageClass = 3
                elif age <= 20:
                    ageClass = 4
                elif age <= 25:
                    ageClass = 5
                elif age <= 30:
                    ageClass = 6
                elif age <= 40:
                    ageClass = 7
                elif age <= 50:
                    ageClass = 8
                elif age <= 60:
                    ageClass = 9
                elif age <= 80:
                    ageClass = 10
                else:
                    ageClass = 11

                i = numData % 5;
                j = int(numData / 5.0)

                resizedShape = (500, (int)(originalShape[1] * 500.0 / originalShape[0]))
                image = cv.resize(imgOriginal, resizedShape)
                image = PreprocessImage.preprocess(image, False)
                path = os.path.join(savePath, str(i), str(j))
                np.savez_compressed(path, a = image)
                imageCropped[i].append(PreprocessImage.preprocess(cv.resize(imgCropped, (224, 224)), False))
                cropbox[i].append(np.array([x1, y1, x2, y2]) * 500.0 / originalShape[0])
                label[i].append(np.array([ageClass, gender, race]))
            
                numData = numData + 1
                if numData % 100 == 0:
                    print(numData)


    for i in range(5):
        path = os.path.join(savePath, str(i))
        np.savez_compressed(os.path.join(path, 'imageCropped'), a = np.array(imageCropped[i], dtype = np.uint8))
        np.savez_compressed(os.path.join(path, 'cropbox'), a = np.array(cropbox[i], dtype = int))
        np.savez_compressed(os.path.join(path, 'label'), a = np.array(label[i], dtype = int))

    print('end')


if __name__ == '__main__':
    main()