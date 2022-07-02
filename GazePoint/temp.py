import importlib
import os
import sys
import numpy as np
import cv2
def b():

    package = importlib.import_module(".test","model")
    r = package.add()
    aa = package.A()
    aa.bb()
    print(r)


img = cv2.imread("imgs/1.jpg")
img = cv2.resize(img,(1920,1080))
img.fill(255)
cv2.imwrite("imgs/1920_1080.jpg",img)
img= cv2.resize(img,(1440,900))
cv2.imwrite("imgs/1440_900.jpg",img)
# # 使用Numpy创建一张A4(2105×1487)纸
# img = np.zeros((1920,1080,3))
# # 使用白色填充图片区域,默认为黑色
# img.fill(255)
#
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)  #当choice为0打开笔记本的第一个摄像头，choice为视频路径打开视频
print(cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    # 调整窗口大小
    #
    # cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    # cv2.resizeWindow("frame", 1920, 1080)    # 设置长和宽
    # cv2.imshow("frame",img)
    # cv2.rectangle(img,(200,400),(800,1400),(0,0,255),2)
    # # cv2.circle(img, (540,960), 3, (0, 255, 0), -1)
    # cv2.circle(img, (800, 1680), 3, (0, 255, 0), -1)  #点是反着的(h,w)
    # cv2.circle(img,(1080, 800), 10, (0, 255, 0), 2)
    cv2.circle(img,(1600,1000),10, (0, 255, 0), 2)
    cv2.imshow("frame",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()