import cv2
import dlib
import argparse
from imutils import face_utils
import numpy as np
import utils_test as utils
from easydict import EasyDict as edict
import os
from scipy.spatial import distance as dist


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face/shape_predictor_68_face_landmarks.dat')

parser = argparse.ArgumentParser(description="程序参数")
parser.add_argument("--choice",type=str,default='0',help='video file or camera')
parser.add_argument("--threshold",type=float,default=0.22,help='EAR_threshold')

args = parser.parse_args()
# print(args.choice,type(args.choice))
choice = args.choice
if choice == '0':
    choice = 0
EAR_threshold = args.threshold

img_root = "imgs"
faceRoot = img_root + "/face"
rightRoot = img_root + "/right"
leftRoot = img_root + "/left"
if not os.path.exists(faceRoot):
    os.makedirs(faceRoot)
if not os.path.exists(leftRoot):
    os.makedirs(leftRoot)
if not os.path.exists(rightRoot):
    os.makedirs(rightRoot)


cap = cv2.VideoCapture(choice)  #当choice为0打开笔记本的第一个摄像头，choice为视频路径打开视频
print(cap.isOpened())
imgNum = 0
while cap.isOpened():
    ret,frame = cap.read()
    #print("Shape",frame.shape)  h,w,c
    rects = detector(frame, 0)  # 人脸检测   len(rects)人脸个数
    #print("人脸个数", len(rects))  # 查看rects的长度，即检测到的人脸的个数
    # #如果检测多个人脸
    # if len(rects) > 1:
    #     break

    for rect in rects:
        shape = predictor(frame, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates
        # print(shape)

        #如果没眨眼
        if not utils.isBlink(points,EAR_threshold):
            rot_frame = utils.rotationFrame(frame,points)

            #图像预处理方式1，可返回裁剪图像，grid，三个box,但需要手动根据距离GetFaceBox函数中的length来确认box
            rot_rects = detector(rot_frame, 0)  # 对旋转后的frame人脸检测
            for rot_rect in rot_rects:
                rot_shape = predictor(rot_frame, rot_rect)  # 对旋转后的frame关键点检测
                rot_points = face_utils.shape_to_np(rot_shape)  # 转换为numpy格式
                #调用normalize2函数
                faceImg, lefteyeImg,righteyeImg,grid,faceConer,leftEyeCorner,rightEyeCorner = utils.normlize2(rot_frame,rot_points)
                # cv2.circle(rot_frame, rot_points[39], 2, (0, 0, 255), -1)
                # cv2.circle(rot_frame, rot_points[36], 2, (0, 0, 255), -1)
                cv2.imshow("Image",faceImg)

            # #图像预处理2，只返回裁剪图像
            # face, lefteye, righteye = utils.normalize1(points, frame)
            # cv2.imshow("face",face)
        else:
            print("眨眼！！！！")
        # cv2.imwrite(faceRoot + "/f_" + str(imgNum) + ".jpg", face)
        # cv2.imwrite(leftRoot + "/le_" + str(imgNum) + ".jpg", lefteye)
        # cv2.imwrite(rightRoot + "/re_" + str(imgNum) + ".jpg", righteye)
        # imgNum += 1
        # print(p_1,p_2)
        # cv2.circle(frame,p_1,2,(0,0,255),-1)
        # cv2.circle(frame, p_2, 2, (0, 0, 255), -1)
        # cv2.circle(frame, p_3, 2, (0, 0, 255), -1)
        #
        # cv2.circle(frame, left_corner, 2, (0, 0, 255), -1)
        # cv2.circle(frame, right_corner, 2, (0, 0, 255), -1)

    #按q结束
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()