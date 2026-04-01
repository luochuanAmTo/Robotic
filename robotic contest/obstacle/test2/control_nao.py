# coding:utf-8

from proxy_and_image import *


# 调整机器人部位角度
def change_the_postion(motionProxy, name, targetAngles):
    motionProxy.angleInterpolationWithSpeed(name, targetAngles, 0.2)
    return True


if __name__ == "__main__":
    ip = CONFIG["ip"]
    mt_proxy = get_Proxy("ALMotion", ip)
    mt_proxy.moveTo(0.3, 0, 0)
