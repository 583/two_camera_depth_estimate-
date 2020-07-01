import numpy as np

post1 = np.load('./params/post1.npy')
post3 = np.load('./params/post3.npy')
disparity = []

for (x1, y1), (x2, y2) in zip(post1, post3):
    disparity.append((x1-x2, y1-y2))
    print(379*707.0912/(x1-x2))

print(disparity)