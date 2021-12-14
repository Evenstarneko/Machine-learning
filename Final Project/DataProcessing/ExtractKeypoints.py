import numpy as np
import cv2 as cv

def extractKeypoints(img1, img2, N):
    sift = cv.SIFT_create()
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

    if keypoint1 is None or keypoint2 is None or len(keypoint1) < N or len(keypoint2) < N:
        return np.array([]), np.array([])

    bf = cv.BFMatcher_create()
    matches = bf.knnMatch(descriptor1, descriptor2, k = 2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    matches = sorted(good, key = lambda x:x.distance)[:N]

    points1 = np.float32([keypoint1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoint2[m.trainIdx].pt for m in matches])

    #img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    #for p1, p2 in zip(points1.swapaxes(0, 1), points2.swapaxes(0, 1)):
    #    color = tuple(np.random.randint(0, 255, 3).tolist())
    #    img1 = cv.circle(img1, (int(p1[0]), int(p1[1])), 3, color, -1)
    #    img2 = cv.circle(img2, (int(p2[0]), int(p2[1])), 3, color, -1)

    #img0 = np.concatenate((img1, img2), axis = 0)
    #cv.imshow('image', img0)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return points1, points2