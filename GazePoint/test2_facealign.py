import cv2
import dlib
import argparse
from imutils import face_utils
import numpy as np
import utils_test as utils
from easydict import EasyDict as edict
import os
from scipy.spatial import distance as dist
import importlib
import torch


def loadModel(modelname,parameterpath,dataset,device):
    #动态加入模型名
    model = importlib.import_module("."+modelname,"model")
    net = model.GazeModel()
    net.to(device)
    #加载模型参数
    net.load_state_dict(torch.load(os.path.join(parameterpath, modelname + "_" + dataset + ".pth")))
    net.eval()

    return net

def gazeEstimation(net,modelname,faceImg, lefteyeImg,righteyeImg,grid,box_rects,device):
    faceImg = faceImg / 255
    faceImg = torch.from_numpy(faceImg.transpose(2, 0, 1)).type(torch.FloatTensor)
    faceImg = faceImg.view(1,faceImg.size(0),faceImg.size(1),faceImg.size(2)).to(device)
    #变换成BCHW格式，以用于net的输入
    lefteyeImg = lefteyeImg/255
    lefteyeImg = torch.from_numpy(lefteyeImg.transpose(2, 0, 1)).type(torch.FloatTensor)
    lefteyeImg = lefteyeImg.view(1,lefteyeImg.size(0),lefteyeImg.size(1),lefteyeImg.size(2)).to(device)
    output = 0.0
    if modelname == "AFFNet":
        righteyeImg = cv2.flip(righteyeImg, 1)  # 1代表水平翻转
        righteyeImg = righteyeImg / 255
        righteyeImg = torch.from_numpy(righteyeImg.transpose(2,0,1)).type(torch.FloatTensor)
        righteyeImg = righteyeImg.view(1,righteyeImg.size(0),righteyeImg.size(1),righteyeImg.size(2)).to(device)

        box_rects = torch.from_numpy(box_rects).type(torch.FloatTensor)
        box_rects = box_rects.view(1,-1).to(device)

        #print("size",righteyeImg.size(),faceImg.size(),box_rects.size())
        output = net(lefteyeImg,righteyeImg,faceImg,box_rects)
        # print(output)
        output = output.cpu().detach().numpy()
        # print("222",output2, type(output2))

    elif modelname == "FullFace":
        output = net(faceImg)
        output = output.cpu().detach().numpy()

    elif modelname == "Itracker":
        righteyeImg = righteyeImg / 255
        righteyeImg = torch.from_numpy(righteyeImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        righteyeImg = righteyeImg.view(1, righteyeImg.size(0), righteyeImg.size(1), righteyeImg.size(2)).to(device)

        grid = np.expand_dims(grid, 0)
        grid = torch.from_numpy(grid).type(torch.FloatTensor).to(device)

        output = net(faceImg,lefteyeImg,righteyeImg,grid)
        output = output.cpu().detach().numpy()

    else:  #Resnet
        righteyeImg = righteyeImg / 255
        righteyeImg = torch.from_numpy(righteyeImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        righteyeImg = righteyeImg.view(1, righteyeImg.size(0), righteyeImg.size(1), righteyeImg.size(2)).to(device)

        output = net(faceImg,lefteyeImg,righteyeImg)
        output = output.cpu().detach().numpy()


    return output

def gazeEstimation2(net,modelname,faceImg, lefteyeImg, righteyeImg, device):
    if modelname == "FullFace":
        faceImg = cv2.resize(faceImg, (448, 448)) / 255
        faceImg = torch.from_numpy(faceImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        faceImg = faceImg.view(1, faceImg.size(0), faceImg.size(1), faceImg.size(2)).to(device)
        output = net(faceImg)
        output = output.cpu().detach().numpy()
        return output
    else:
        faceImg = cv2.resize(faceImg,(224,224))/255
        faceImg = torch.from_numpy(faceImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        faceImg = faceImg.view(1, faceImg.size(0), faceImg.size(1), faceImg.size(2)).to(device)
        # 变换成BCHW格式，以用于net的输入
        lefteyeImg = cv2.resize(lefteyeImg,(224,224))/255
        lefteyeImg = torch.from_numpy(lefteyeImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        lefteyeImg = lefteyeImg.view(1, lefteyeImg.size(0), lefteyeImg.size(1), lefteyeImg.size(2)).to(device)

        righteyeImg = cv2.resize(righteyeImg, (224, 224)) / 255
        righteyeImg = torch.from_numpy(righteyeImg.transpose(2, 0, 1)).type(torch.FloatTensor)
        righteyeImg = righteyeImg.view(1, righteyeImg.size(0), righteyeImg.size(1), righteyeImg.size(2)).to(device)
        output = net(faceImg,lefteyeImg,righteyeImg)
        output = output.cpu().detach().numpy()
        return output
    return np.array([0,0])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face/shape_predictor_68_face_landmarks.dat')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="程序参数")
parser.add_argument("--choice",type=str,default='0',help='video file or camera')
parser.add_argument("--threshold",type=float,default=0.05,help='EAR_threshold')  #0.22
parser.add_argument("--modelname",type=str, default='Itracker',help='Model choice,AFFNet,FullFace,Itracker and Resnet')
parser.add_argument("--dataset",type=str,default='mpii',help="model is trained by the dataset,mpii or diap")
parser.add_argument("--parameterpath",type=str,default='./pth/MPII_mm',help="parameter path of gaze model")

args = parser.parse_args()

choice = args.choice
if choice == '0':
    choice = 0
EAR_threshold = args.threshold
modelname = args.modelname
dataset = args.dataset
parameterpath = args.parameterpath

eye_shape = (60, 36)
face_shape = (224,224)
if modelname == "AFFNet":
    eye_shape = (112, 112)
    face_shape = (224, 224)
elif modelname == "FullFace":
    face_shape = (448,448)
else:  # Itracker, Resnet
    eye_shape = (224, 224)
    face_shape = (224, 224)



img_root = "imgs"
faceRoot = img_root + "/face"
rightRoot = img_root + "/right"
leftRoot = img_root + "/left"
gridRoot = img_root + "/grid"
if not os.path.exists(faceRoot):
    os.makedirs(faceRoot)
if not os.path.exists(leftRoot):
    os.makedirs(leftRoot)
if not os.path.exists(rightRoot):
    os.makedirs(rightRoot)
if not os.path.exists(gridRoot):
    os.makedirs(gridRoot)

img = cv2.imread("imgs/1920_1080.jpg")

cap = cv2.VideoCapture(choice)  #当choice为0打开笔记本的第一个摄像头，choice为视频路径打开视频
print(cap.isOpened())
imgNum = 0

# cv2.imshow("frame", img)
while cap.isOpened():
    ret,frame = cap.read()
    #print("Shape",frame.shape)  h,w,c
    rects = detector(frame, 0)  # 人脸检测   len(rects)人脸个数
    #print("人脸个数", len(rects))  # 查看rects的长度，即检测到的人脸的个数
    # #如果检测多个人脸
    # if len(rects) > 1:
    #     break
    temp_img = img
    for rect in rects:
        shape = predictor(frame, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates
        # print(shape)
        #如果没眨眼
        if not utils.isBlink(points,EAR_threshold):

            # faceImg, lefteyeImg, righteyeImg = utils.normalize1(points, frame)
            # cv2.imshow("img", faceImg)
            # lefteyeImg = cv2.flip(lefteyeImg, 1)  # 1代表水平翻转
            # righteyeImg = cv2.flip(righteyeImg, 1)  # 1代表水平翻转
            #   faceImg = cv2.flip(faceImg, 1)  # 1代表水平翻转
            # cv2.imshow("img",lefteyeImg)
            # cv2.imshow("img222",righteyeImg)
            # # # cv2.imshow("Image",righteyeImg)
            # # 将裁剪后的眼睛图像变成3通道的灰度图像用于网络输入
            # lefteyeImg = cv2.cvtColor(lefteyeImg, cv2.COLOR_RGB2GRAY)
            # lefteyeImg = np.stack((lefteyeImg, lefteyeImg, lefteyeImg), 0)  # 堆叠三层，变成3通道的灰度图像
            # lefteyeImg = lefteyeImg.transpose(1, 2, 0)
            #
            # # 将裁剪后的眼睛图像变成3通道的灰度图像用于网络输入
            # righteyeImg = cv2.cvtColor(righteyeImg, cv2.COLOR_RGB2GRAY)
            # righteyeImg = np.stack((righteyeImg, righteyeImg, righteyeImg), 0)
            # righteyeImg = righteyeImg.transpose(1, 2, 0)

            # 加载模型
            # net = loadModel(modelname, parameterpath, dataset, device)
            # gaze = gazeEstimation2(net, modelname, faceImg, lefteyeImg, righteyeImg, device)
            # point_w = int((gaze[0][0] * 1920) / (34.54*10))
            # point_h = int((gaze[0][1] * 1080 ) / (19.43*10))
            # # point_w = int(gaze[0][0] / 0.2238053125 * 10)
            # # point_h = int(gaze[0][1] / 0.22380524999 * 10)
            # # # cv2.rectangle(img, (200, 400), (800, 1400), (0, 0, 255), 2)
            # cv2.circle(img, (point_h, point_w), 8 , (0, 255, 0), -1)
            # 0.2238053125,0.22380524999999998

            rot_frame = utils.rotationFrame(frame,points)
            #图像预处理方式1，可返回裁剪图像，grid，三个box,但需要手动根据距离GetFaceBox函数中的length来确认box
            rot_rects = detector(rot_frame, 0)  # 对旋转后的frame人脸检测
            for rot_rect in rot_rects:
                rot_shape = predictor(rot_frame, rot_rect)  # 对旋转后的frame关键点检测
                rot_points = face_utils.shape_to_np(rot_shape)  # 转换为numpy格式
                #调用normalize2函数
                faceImg, lefteyeImg,righteyeImg,grid,faceConer,leftEyeCorner,rightEyeCorner = utils.normlize2(rot_frame,rot_points,eye_shape,face_shape)
                cv2.imshow("img", faceImg)
                #cv2.imshow("img",lefteyeImg)
                box_rects = np.array(faceConer+leftEyeCorner+rightEyeCorner)  #将将面部眶、眼角框的左上和右下角位置坐标拼接list使用+可拼接
                #print(type(box_rects),type(faceImg),type(grid),type(lefteyeImg))

                # lefteyeImg = cv2.flip(lefteyeImg, 1)  # 1代表水平翻转
                # righteyeImg = cv2.flip(righteyeImg, 1)  # 1代表水平翻转
                #qfaceImg = cv2.flip(faceImg, 1)  # 1代表水q

                # cv2.imshow("Image",righteyeImg)
                #将裁剪后的眼睛图像变成3通道的灰度图像用于网络输入
                lefteyeImg = cv2.cvtColor(lefteyeImg, cv2.COLOR_RGB2GRAY)
                lefteyeImg = np.stack((lefteyeImg, lefteyeImg, lefteyeImg), 0)  #堆叠三层，变成3通道的灰度图像
                lefteyeImg = lefteyeImg.transpose(1,2,0)

                # cv2.imshow("img", lefteyeImg)
                # 将裁剪后的眼睛图像变成3通道的灰度图像用于网络输入
                righteyeImg = cv2.cvtColor(righteyeImg, cv2.COLOR_RGB2GRAY)
                righteyeImg = np.stack((righteyeImg,righteyeImg,righteyeImg),0)
                righteyeImg = righteyeImg.transpose(1,2,0)

                #加载模型
                net = loadModel(modelname,parameterpath,dataset,device)
                gaze = gazeEstimation(net,modelname,faceImg,lefteyeImg,righteyeImg,grid,box_rects,device)
                print(gaze)
                #gaze = gazeEstimation2(net, faceImg, lefteyeImg, righteyeImg, device)
                point_w = int(((gaze[0][0] +0 ) * 1920) / (34.54 *10 ))
                point_h = int(((gaze[0][1]  ) * 1080 ) / (19.43 * 10))
                # point_w = int((gaze[0][0]*10) /0.2238053125 )
                # point_h = int((gaze[0][1]*10) /0.22380524999)
                print("point",point_w,point_h)
                #cv2.rectangle(img, (200, 400), (800, 1400), (0, 0, 255), 2)
                cv2.circle(temp_img, (point_w,point_h), 10, (0, 255, 0), 2)
            #     #0.2238053125,0.22380524999999998


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
    #cv2.imshow("img",faceImg)
    cv2.imshow("background", temp_img)
    #按q结束
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()