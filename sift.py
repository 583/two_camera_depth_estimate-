import cv2
import numpy as np
from matplotlib import pyplot as plt

# 载入图像
img1_gray = cv2.imread("./img/1.jpg")
img2_gray = cv2.imread("./img/2.jpg")

# 绘制特征点信息
def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    # 提取图像前两个维度，即长度和宽度信息，不包含颜色数据信息
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    # 生成一张黑色大图
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # 大图左边填充图1，右边填充图2
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray
    # 取得第0张图片的p1特征点的索引和对应在第1张图片中的该特征点的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    # 将图像1中的x坐标保存到post1集合
    post1 = np.int32([kp1[pp].pt for pp in p1])
    np.save('./params/post1', post1)
    # 将图像2中的x相对于整合大图的坐标保存到post1集合
    post3 = np.int32([kp2[pp].pt for pp in p2])
    np.save('./params/post3', post3)
    # 将图像2中的x相对于图片2的坐标保存到post1集合
    post2 = post3 + (w1, 0)
    np.save('./params/post2', post2)
    # 开始连线显示坐标
    index = 0
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
        cv2.putText(vis, str(index) + ") " + str(x1)+','+str(y1), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        cv2.putText(vis, str(index) + ") " + str(post3[index][0]) + ',' + str(y2), (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        index+=1

    # vis2 = vis[:,:,::-1]
    # plt.imshow(vis2)
    cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)
#     plt.imshow(vis)

# 特征点提取
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SURF_create()

# 对两幅图像检测特征和描述符，直接找到关键点并计算
# kp为关键点列表，des为numpy的数组，为关键点数目×128
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 创建BF匹配器对象
bf = cv2.BFMatcher(cv2.NORM_L2)
# 返回k个最匹配的
matches = bf.knnMatch(des1, des2, k=2)

# 预处理匹配集合
goodMatch = []
for m, n in matches:
    # 把所获取的特征点的欧氏距离小于0.5的点加入匹配集合
    # 结果各个参数的意义如下：
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

print(len(goodMatch))

# 显示匹配特征点集合
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()