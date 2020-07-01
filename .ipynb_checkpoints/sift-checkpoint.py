import cv2
import numpy as np
from matplotlib import pyplot as plt

img1_gray = cv2.imread("./img/1.jpg")
img2_gray = cv2.imread("./img/2.jpg")

def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    np.save('./params/post1', post1)
    post3 = np.int32([kp2[pp].pt for pp in p2])
    np.save('./params/post3', post3)
    post2 = post3 + (w1, 0)
    np.save('./params/post2', post2)

    index = 0
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
        cv2.putText(vis, str(index) + ") " + str(x1)+','+str(y1), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        cv2.putText(vis, str(index) + ") " + str(post3[index][0]) + ',' + str(y2), (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        index+=1

    vis2 = vis[:,:,::-1]
    plt.imshow(vis2)
    cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)
#     plt.imshow(vis)

# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SURF_create()

# 直接找到关键点并计算
# kp为关键点列表，des为numpy的数组，为关键点数目×128
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

print(len(goodMatch))
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()