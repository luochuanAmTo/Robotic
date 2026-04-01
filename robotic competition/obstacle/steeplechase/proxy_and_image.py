# coding:utf-8
import vision_definitions
from naoqi import ALProxy
import cv2
import numpy as np

# 配置参数
CONFIG = {
    "ip": "192.168.43.229",
    "resolution": vision_definitions.kVGA,
    # "resolution": vision_definitions.k4VGA,
    "colorSpace": vision_definitions.kBGRColorSpace,
    "fps": 20,

}
color_ranges = {
        'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))],
        'blue': (np.array([80, 50, 50]), np.array([130, 255, 255])),
        'yellow': (np.array([25,50, 90]), np.array([30, 255, 255]))
    }

# 封装代理
def get_Proxy(modelName, IP, port=9559):
    proxy = ALProxy(modelName, IP, port)
    return proxy


def get_image_from_camera(camera_id, camera_proxy, videoClient):
    # 获取图片, 一帧一帧组成视频流
    camera_proxy.setActiveCamera(camera_id)

    # 返回的frame中， 第一维为图像的宽，第二维为图片的高，第三维为图片的通道数，第六维为图片本身数组
    frame = camera_proxy.getImageRemote(videoClient)
    frameWidth = frame[0]
    frameHeight = frame[1]
    frameChannels = frame[2]

    # 将图片转换成numpy数组，并且reshape成标准的形状，方便我们使用cv2来展示
    frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frameHeight, frameWidth, frameChannels])
    return frameArray


if __name__ == '__main__':
    pass
    # vd_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
    # videoClient = vd_proxy.subscribeCamera("python_GVM", 1, CONFIG["resolution"], CONFIG["colorSpace"], CONFIG["fps"])
    # while 1:
    #     img = get_image_from_camera(1, vd_proxy, videoClient)
    #     cv2.imshow("res", img)
    #     cv2.waitKey(1)
