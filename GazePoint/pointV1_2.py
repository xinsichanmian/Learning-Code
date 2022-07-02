import cv2
import dlib
import argparse
from imutils import face_utils
import numpy as np
import utils_test as utils
from easydict import EasyDict as edict
import os
from scipy.spatial import distance as dist

def rotationFrame(frame,points):
    p_1 = points[28 - 1]

    left_corner = points[43 - 1]  #左眼眼角
    right_corner = points[40 - 1] #右眼眼角
    dx = right_corner[0] - left_corner[0]
    dy = right_corner[1] - left_corner[1]

    rot_angle = np.degrees(np.arctan2(dy, dx)) - 180  # 旋转角度
    #print("旋转角度", rot_angle)

    center = (int(p_1[0]), int(p_1[1]))  #旋转中心，绕改点旋转
    # print("Type(c)",type(center),center)
    rot_M = cv2.getRotationMatrix2D(center, rot_angle, 1)
    #旋转后，对齐的的帧
    rot_frame = cv2.warpAffine(frame, rot_M, (frame.shape[1], frame.shape[0]))

    return rot_frame

def normlize2(rot_frame,rot_points):
    lcenter = (rot_points[42] + rot_points[45]) // 2  # 左眼中心坐标
    rcenter = (rot_points[36] + rot_points[39]) // 2  # 右眼中心坐标
    # print(leftEyeCenter,rightEyeCenter)
    #为面部box计算face_length
    face_length = rot_points[26][0]-rot_points[0][0]
    #print("face长度",face_length)

    lefteye_width = rot_points[45][0] - rot_points[42][0]
    #print("lefteye_width",lefteye_width)
    leftbox = utils.GetEyeBox(lcenter,lefteye_width)  # 从旋转后的rot_frame中获取左眼box
    lefteyeImg = utils.CropImg(rot_frame,
                               leftbox.begin[0],
                               leftbox.begin[1],
                               leftbox.width,
                               leftbox.height)
    lefteyeImg = cv2.resize(lefteyeImg, (60, 36))

    righteye_width = rot_points[39][0] - rot_points[36][0]
    #print("right_width",righteye_width)
    rightbox = utils.GetEyeBox(rcenter,righteye_width)  # 右眼box
    righteyeImg = utils.CropImg(rot_frame,
                                rightbox.begin[0],
                                rightbox.begin[1],
                                rightbox.width,
                                rightbox.height)
    righteyeImg = cv2.resize(righteyeImg, (60, 36))

    facebox = utils.GetFaceBox(lcenter, rcenter,face_length)  # 面部box
    # print("leftbox:", leftbox, "rightbox:", rightbox, "facebox:", facebox)

    faceImg = utils.CropImg(rot_frame,
                            facebox.begin[0],
                            facebox.begin[1],
                            facebox.width,
                            facebox.height)
    faceImg = cv2.resize(faceImg, (224, 224))

    image = edict()
    image.width = 640
    image.height = 480
    grid = utils.GetGrid(image, facebox)

    # 面部眶、左眼眶、右眼眶的左上角、右下角的位置
    faceConer = list(map(float, [facebox.begin[0], facebox.begin[1], facebox.begin[0] + facebox.width,
                                 facebox.begin[1] + facebox.height]))
    leftEyeCorner = list(map(float, [leftbox.begin[0], leftbox.begin[1], leftbox.begin[0] + leftbox.width,
                                     leftbox.begin[1] + leftbox.height]))
    rightEyeCorner = list(map(float, [rightbox.begin[0], rightbox.begin[1], rightbox.begin[0] + rightbox.width,
                                      rightbox.begin[1] + rightbox.height]))

    return faceImg, lefteyeImg, righteyeImg, grid, faceConer, leftEyeCorner, rightEyeCorner
    #return faceImg, lefteyeImg,righteyeImg,grid,faceConer,leftEyeCorner,rightEyeCorner

#预处理1，返回裁剪后的面部、左右眼图像
def normalize1(frame,points):
    # Adapted from imutils package
    left_eye_coord = (0.70, 0.35)

    lcenter = (points[46 - 1] + points[43 - 1]) / 2
    rcenter = (points[40 - 1] + points[37 - 1]) / 2

    center_f = (int((lcenter[0] + rcenter[0]) / 2), int((lcenter[1] + rcenter[1]) / 2))

    center_l = (int(lcenter[0]), int(lcenter[1]))
    center_r = (int(rcenter[0]), int(rcenter[1]))

    # compute the angle between the eye centroids
    dY = rcenter[1] - lcenter[1]
    dX = rcenter[0] - lcenter[0]
    angle = np.degrees(np.arctan2(dY, dX))
    #print("旋转角度ffff",angle)
    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    right_eye_x = 1.0 - left_eye_coord[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    new_dist = (right_eye_x - left_eye_coord[0])
    new_dist_f = new_dist * 245  # 放缩距离
    scale_f = new_dist_f / dist

    new_dist_e = new_dist * 245  # 放缩距离
    scale_e = new_dist_e / dist

    # grab the rotation matrix for rotating and scaling the face
    M_f = cv2.getRotationMatrix2D(center_f, angle, scale_f)  # M 为变换矩阵

    # update the translation component of the matrix
    tX_f = 224 * 0.5
    tY_f = 224 * left_eye_coord[1]
    M_f[0, 2] += (tX_f - center_f[0])
    M_f[1, 2] += (tY_f - center_f[1])

    M_le = cv2.getRotationMatrix2D(center_l, angle, scale_e)  # M 为变换矩阵
    M_re = cv2.getRotationMatrix2D(center_r, angle, scale_e)
    # update the translation component of the matrix
    tX_e = 60 * 0.5
    tY_e = 60 * left_eye_coord[1]
    M_le[0, 2] += (tX_e - center_l[0])
    M_le[1, 2] += (tY_e - center_l[1])

    M_re[0, 2] += (tX_e - center_r[0])
    M_re[1, 2] += (tY_e - center_r[1])

    # apply the affine transformation
    lefteye = cv2.warpAffine(frame, M_le, (60, 36),  # 设置成112*112格式的输入(60, 36)
                          flags=cv2.INTER_CUBIC)
    righteye = cv2.warpAffine(frame, M_re, (60, 36),  # 设置成112*112格式的输入
                          flags=cv2.INTER_CUBIC)
    # apply the affine transformation
    # 设置成112*112格式的输出
    face = cv2.warpAffine(frame, M_f, (224, 224), flags=cv2.INTER_CUBIC)
    return face,lefteye,righteye

# EAR 计算
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates

    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio

    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def isBlink(points,EAR_threshold):
    #左右眼轮廓的关键点坐标
    leftEye = points[left_eye_start:left_eye_end + 1]
    rightEye = points[right_eye_start:right_eye_end + 1]

    leftEAR = eye_aspect_ratio(leftEye)  # 左眼EAR
    rightEAR = eye_aspect_ratio(rightEye)  # 右眼EAR
    eyeEAR = (leftEAR + rightEAR) / 2  # EAR平均值

    #当大于设置的EAR阈值时，处于未眨眼状态返回False，否则返回True
    if eyeEAR > EAR_threshold:
        return False

    return True

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

# img_root = "imgs"
# faceRoot = img_root + "/face"
# rightRoot = img_root + "/right"
# leftRoot = img_root + "/left"
# if not os.path.exists(faceRoot):
#     os.makedirs(faceRoot)
# if not os.path.exists(leftRoot):
#     os.makedirs(leftRoot)
# if not os.path.exists(rightRoot):
#     os.makedirs(rightRoot)

#右眼关键点
right_eye_start = 37-1
right_eye_end = 42-1
#左眼关键点
left_eye_start = 43-1
left_eye_end = 48-1

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
        if not isBlink(points,EAR_threshold):
            rot_frame = rotationFrame(frame,points)

            #图像预处理方式1，可返回裁剪图像，grid，三个box,但需要手动根据距离GetFaceBox函数中的length来确认box
            rot_rects = detector(rot_frame, 0)  # 对旋转后的frame人脸检测
            for rot_rect in rot_rects:
                rot_shape = predictor(rot_frame, rot_rect)  # 对旋转后的frame关键点检测
                rot_points = face_utils.shape_to_np(rot_shape)  # 转换为numpy格式
                #faceImg, lefteyeImg,righteyeImg,grid,faceConer,leftEyeCorner,rightEyeCorner = normlize2(rot_frame,rot_points)
                faceImg, lefteyeImg,righteyeImg = normalize1(rot_frame, rot_points)
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