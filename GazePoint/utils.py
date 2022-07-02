import cv2
import numpy as np
from easydict import EasyDict as edict
import dlib
from imutils import face_utils
import os
from scipy.spatial import distance as dist

def GetFaceBox_original(lefteye, righteye):
  face = edict()
  face.center = (lefteye + righteye)/2
  #length = 112*0.9
  length = 160 #224 OK
  #face.begin = (face.center - np.array([0.5*length, 0.3*length])).astype("int")
  face.begin = (face.center - np.array([0.5 * length, 0.3 * length])).astype("int")
  face.width = int(length)
  face.height = int(length)
  return bound(face)

def GetEyeBox_original(center):
  # eye
  eye = edict()
  eye.center = center
  eye.width = 60*0.8
  times = eye.width / 60
  eye.height = int(36*times)
  eye.begin = (eye.center - np.array([0.5*eye.width, 0.5*eye.height])).astype("int")
  return bound(eye)


def GetFaceBox(lefteye, righteye,length):
  face = edict()
  face.center = (lefteye + righteye)/2
  #length = 112*0.9
  #length = 160 #224 OK
  length = length * 1.1
  #face.begin = (face.center - np.array([0.5*length, 0.3*length])).astype("int")
  face.begin = (face.center - np.array([0.5 * length, 0.3 * length])).astype("int")
  face.width = int(length)
  face.height = int(length)
  return bound(face)

def GetEyeBox(center,width):
  # eye
  eye = edict()
  eye.center = center
  eye.width = width*1.3  #*1.3 for right     1.4 for left
  times = eye.width / 60
  eye.height = int(36*times)
  eye.begin = (eye.center - np.array([0.5*eye.width, 0.5*eye.height])).astype("int")
  return bound(eye)


def GetGrid(image, face):
  img = np.zeros((image.width, image.height))
  img[face.begin[0]:face.begin[0] + face.width, face.begin[1]:face.begin[1]+face.height] = np.ones([face.width,face.height])
  img = cv2.resize(img, (25, 25))
  return img


def bound(axis):
  image = edict()
  image.width = 640    #原始图像的宽
  image.height = 480   #原始图像的高
  axis.begin[0] = max(axis.begin[0], 0)
  axis.begin[1] = max(axis.begin[1], 0)

  if axis.begin[0] + axis.width > image.width:
    axis.begin[0] = image.width - axis.width

  if axis.begin[1] + axis.height > image.height:
    axis.begin[1] = image.height - axis.height
  return axis


def CropImg(img, X, Y, W, H):
    """
    X, Y is the corrdinate of the left-top corner of images.
    W, H is weight and high.
    """

    Y_lim, X_lim  = img.shape[0], img.shape[1]
    H =  min(H, Y_lim)
    W = min(W, X_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H

    return img[Y:(Y+H),X:(X+W)]

#预处理1，返回裁剪后的面部、左右眼图像
def normalize1(points, frame):
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
    lefteye = cv2.warpAffine(frame, M_le, (60, 36),  # 设置成112*112格式的输入
                          flags=cv2.INTER_CUBIC)
    righteye = cv2.warpAffine(frame, M_re, (60, 36),  # 设置成112*112格式的输入
                          flags=cv2.INTER_CUBIC)
    # apply the affine transformation
    # 设置成112*112格式的输出
    face = cv2.warpAffine(frame, M_f, (224, 224), flags=cv2.INTER_CUBIC)
    return face,lefteye,righteye

#旋转帧图像
def rotationFrame(frame,points):
    p_1 = points[28 - 1]  #基于人脸68关键点设置的旋转中心点

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

#图像预处理方法2，可返回面部、左右眼裁剪图像及对应的box（用于AFF-Net），还可返回grid（用于Itracker）
def normlize2(rot_frame,rot_points):
    lcenter = (rot_points[42] + rot_points[45]) // 2  # 左眼中心坐标
    rcenter = (rot_points[36] + rot_points[39]) // 2  # 右眼中心坐标
    # print(leftEyeCenter,rightEyeCenter)
    #为面部box计算face_length
    face_length = rot_points[26][0]-rot_points[0][0]
    #print("face长度",face_length)

    lefteye_width = rot_points[45][0] - rot_points[42][0]
    #print("lefteye_width",lefteye_width)
    leftbox = GetEyeBox(lcenter,lefteye_width)  # 从旋转后的rot_frame中获取左眼box
    lefteyeImg = CropImg(rot_frame,
                               leftbox.begin[0],
                               leftbox.begin[1],
                               leftbox.width,
                               leftbox.height)
    lefteyeImg = cv2.resize(lefteyeImg, (60, 36))

    righteye_width = rot_points[39][0] - rot_points[36][0]
    #print("right_width",righteye_width)
    rightbox = GetEyeBox(rcenter,righteye_width)  # 右眼box
    righteyeImg = CropImg(rot_frame,
                                rightbox.begin[0],
                                rightbox.begin[1],
                                rightbox.width,
                                rightbox.height)
    righteyeImg = cv2.resize(righteyeImg, (60, 36))

    facebox = GetFaceBox(lcenter, rcenter,face_length)  # 面部box
    # print("leftbox:", leftbox, "rightbox:", rightbox, "facebox:", facebox)

    faceImg = CropImg(rot_frame,
                            facebox.begin[0],
                            facebox.begin[1],
                            facebox.width,
                            facebox.height)
    faceImg = cv2.resize(faceImg, (224, 224))

    image = edict()
    image.width = 640
    image.height = 480
    grid = GetGrid(image, facebox)

    # 面部眶、左眼眶、右眼眶的左上角、右下角的位置
    faceConer = list(map(float, [facebox.begin[0], facebox.begin[1], facebox.begin[0] + facebox.width,
                                 facebox.begin[1] + facebox.height]))
    leftEyeCorner = list(map(float, [leftbox.begin[0], leftbox.begin[1], leftbox.begin[0] + leftbox.width,
                                     leftbox.begin[1] + leftbox.height]))
    rightEyeCorner = list(map(float, [rightbox.begin[0], rightbox.begin[1], rightbox.begin[0] + rightbox.width,
                                      rightbox.begin[1] + rightbox.height]))

    return faceImg, lefteyeImg, righteyeImg, grid, faceConer, leftEyeCorner, rightEyeCorner
    #return faceImg, lefteyeImg,righteyeImg,grid,faceConer,leftEyeCorner,rightEyeCorner

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

# 眼睛轮廓的关键点
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # (42, 48)
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)
# 检测是否眨眼，眨眼为Ture，不眨眼为False
def isBlink(points,EAR_threshold):
    # 右眼关键点
    right_eye_start = 37 - 1
    right_eye_end = 42 - 1
    # 左眼关键点
    left_eye_start = 43 - 1
    left_eye_end = 48 - 1

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