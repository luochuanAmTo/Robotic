# -*- coding:utf-8 -*-

from naoqi import ALProxy
import cv2
import numpy as np
import math
from PIL import Image
import vision_definitions


def get_image_from_camera(camera_id, camera_proxy, video_client):
    # 获取图片, 一帧一帧组成视频流
    camera_proxy.setActiveCamera(camera_id)
    # 返回的frame中， 第一维为图像的宽，第二维为图片的高，第三维为图片的通道数，第六维为图片本身数组
    frame = camera_proxy.getImageRemote(video_client)
    frameWidth = frame[0]
    frameHeight = frame[1]
    frameChannels = frame[2]

    # 将图片转换成numpy数组，并且reshape成标准的形状，方便我们使用cv2来展示
    frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frameHeight, frameWidth, frameChannels])
    return frameArray


# if __name__ == '__main__':
#     vd_proxy = get_Proxy("ALVideoDevice", ip)
#     videoClient = vd_proxy.subscribe("python_GVM", resolution, colorSpace, fps)
#     while True:
#         img = get_image_from_camera(1, vd_proxy, videoClient)
#         cv2.imshow("res", img)
#         cv2.waitKey(1)

def getInitMoveConfig():
    maxStepX = 0.14
    maxStepY = 0.14
    maxStepTheta = 0.4
    maxStepFrequency = 0.7
    stepHeight = 0.06
    TorsoWx = 0.0
    TorsoWy = 0.0
    initMoveConfig = [["MaxStepX", maxStepX],
                      ["MaxStepY", maxStepY],
                      ["MaxStepTheta", maxStepTheta],
                      ["MaxStepFrequency", maxStepFrequency],
                      ["StepHeight", stepHeight],
                      ["TorsoWx", TorsoWx],
                      ["TorsoWy", TorsoWy]]
    return initMoveConfig


def getMoveConfig():
    maxStepX = 0.33
    maxStepY = 0.22
    maxStepTheta = 0.45
    maxStepFrequency = 1.8
    stepHeight = 0.04
    TorsoWx = 0.0
    TorsoWy = 0.0
    moveConfig = [["MaxStepX", maxStepX],
                  ["MaxStepY", maxStepY],
                  ["MaxStepTheta", maxStepTheta],
                  ["MaxStepFrequency", maxStepFrequency],
                  ["StepHeight", stepHeight],
                  ["TorsoWx", TorsoWx],
                  ["TorsoWy", TorsoWy]]
    return moveConfig


def getVideoConfig():
    resolution = vision_definitions.kQVGA
    colorSpace = vision_definitions.kBGRColorSpace
    fps = 30
    videoConfig = [resolution, colorSpace, fps]

    return videoConfig


class MyClass:
    def __init__(self, IP, Port=9559):
        # GeneratedClass.__init__(self)
        self.IP = IP
        self.motion = ALProxy("ALMotion", IP, Port)
        self.posture = ALProxy("ALRobotPosture", IP, Port)
        self.camera = ALProxy("ALVideoDevice", IP, Port)
        self.life = ALProxy("ALAutonomousLife", IP, Port)
        self.tts = ALProxy("ALTextToSpeech", IP, Port)
        self.memory = ALProxy("ALMemory", IP, Port)

    def onLoad(self):
        self.life.setState("disabled")
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)
        self.motion.moveInit()
        self.motion.setMoveArmsEnabled(True, True)
        self.motion.setStiffnesses("LShoulderPitch", 0.0)
        self.motion.setStiffnesses("RShoulderPitch", 0.0)
        self.motion.setStiffnesses("LElbowYaw", 0.0)
        self.motion.setStiffnesses("RElbowYaw", 0.0)
        self.motion.setAngles("HeadPitch", 10 * math.pi / 180.0, 0.8)
        self.camera.setActiveCamera(1)

    # def onUnload(self):
    #     pass

    def yyyonInput_onStart(self):
        videoConfig = getVideoConfig()
        videoStream = self.camera.subscribe("python_GVM", videoConfig[0], videoConfig[1], videoConfig[2])
        self.camera.setCamerasParameter(videoStream, 22, 2)
        imageNumber = 0
        # carema=self.camera
        # img=get_image_from_camera(1,carema,videoStream)
        # cv2.imshow("res", img)
        # cv2.waitKey(1)
        x = y = []
        robotInitPosition = self.motion.getRobotPosition(0)
        initTheta = robotInitPosition[2]
        currentPosition = [0, 0, 0]

        while True:
            robotNewPosition = self.motion.getRobotPosition(0)
            currentPosition[0] = ((robotNewPosition[0] - robotInitPosition[0]) * 100) * math.cos(initTheta) + \
                                 ((robotNewPosition[1] - robotInitPosition[1]) * 100) * math.sin(initTheta)
            currentPosition[1] = ((robotNewPosition[1] - robotInitPosition[1]) * 100) * math.cos(initTheta) - \
                                 ((robotNewPosition[0] - robotInitPosition[0]) * 100) * math.sin(initTheta)
            currentPosition[2] = (robotNewPosition[2] - robotInitPosition[2]) * (180.0 / math.pi)

            x.append(currentPosition[0])
            y.append(currentPosition[1])

            naoImage = self.camera.getImageRemote(videoStream)
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            dataArray = naoImage[6]

            RGBImage = Image.frombytes("RGB", (imageWidth, imageHeight), dataArray)
            dataFrame = np.asarray(RGBImage)

            HSVImage = cv2.cvtColor(dataFrame, cv2.COLOR_BGR2HSV)
            HSVImage = cv2.convertScaleAbs(HSVImage, alpha=2, beta=0)
            lowerThreshold = np.array([0, 0, 100])
            upperThreshold = np.array([180, 50, 255])
            mask = cv2.inRange(HSVImage, lowerThreshold, upperThreshold)
            # cv2.imwrite("D:\\Pycharm Programme\\shencangblue\\imgs", mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # img=self.get_image_from_camera(1,self.camera,RGBImage)
            # cv2.imshow("my",img)
            exDetectedLines = cv2.HoughLines(mask, 1, np.pi / 260, 80)
            if exDetectedLines is None:
                # print "NO LINE"
                detectedLines = [[0, 0]]
            else:
                detectedLines = exDetectedLines[:, 0, :]
                for rho, theta in detectedLines[:]:
                    cosine = np.cos(theta)
                    sine = np.sin(theta)

                    xBias = rho * cosine
                    yBias = rho * sine

                    x1 = int(xBias - 1000 * sine)
                    y1 = int(yBias + 1000 * cosine)
                    x2 = int(xBias + 1000 * sine)
                    y2 = int(yBias - 1000 * cosine)
                    # print("naoqi888")
                    cv2.line(dataFrame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("test", dataFrame)
            # print(dataFrame)
            # img=self.get_image_from_camera(1,self.camera,dataFrame)
            # cv2.imshow("test",img)
            cv2.waitKey(1)

            singleLine = detectedLines[0]
            print "Line: " + str(singleLine)
            rotationValue = 0.2

            if currentPosition[0] <= 5:
                moveConfig = getInitMoveConfig()
            else:
                moveConfig = getMoveConfig()

            if currentPosition[0] >= 610:
                self.motion.move(0.0, 0.0, 0.0, moveConfig)
                self.motion.rest()
                break

            if currentPosition[0] < 10:
                self.motion.move(0.1, 0.0, 0.0, moveConfig)
            else:
                if singleLine[0]:
                    if singleLine[1] <= 1.04 or 1.57 < singleLine[1] <= 2.09:
                        # Turn right
                        self.motion.move(0.3, 0.0, -rotationValue, moveConfig)
                    else:
                        # Turn left
                        self.motion.move(0.3, 0.0, rotationValue, moveConfig)
                else:
                    # print "NOs LINE"
                    if currentPosition[0] <= 610:
                        self.motion.move(0.2, 0.0, 0.0, moveConfig)
                    else:
                        self.motion.move(0.0, 0.0, 0.0, moveConfig)
                        self.motion.rest()
                        break

        self.camera.unsubscribe(videoStream)
        self.onStopped()

    def onInput_onStop(self):
        self.onUnload()
        self.onStopped()


def headtouch(start):

    while True:
        headTouchedButtonFlag = start.memory.getData("FrontTactilTouched")
        if headTouchedButtonFlag == 1.0:
            start.yyyonInput_onStart()


IP = "192.168.43.48"
start = MyClass(IP)
start.onLoad()
headtouch(start)

start.onInput_onStop()