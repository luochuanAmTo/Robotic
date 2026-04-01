# coding:utf-8
import vision_definitions
from naoqi import ALProxy
import cv2
import numpy as np

# 配置参数
CONFIG = {
    "ip": "192.168.43.153",
    "resolution": vision_definitions.kVGA,  # VGA分辨率 640x480
    # "resolution": vision_definitions.k4VGA,
    "colorSpace": vision_definitions.kBGRColorSpace,
    "fps": 20,
}

# 颜色范围定义
color_ranges = {
    'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))],
    'blue': (np.array([80, 50, 50]), np.array([130, 255, 255])),
    'yellow': (np.array([25, 50, 90]), np.array([30, 255, 255]))
}

# 白色检测范围（HSV）
WHITE_RANGE = {
    'white': (np.array([0, 0, 200]), np.array([180, 30, 255]))  # 检测白色区域
}


# 封装代理
def get_Proxy(modelName, IP, port=9559):
    """
    获取NAO机器人的代理对象
    :param modelName: 代理模型名称
    :param IP: 机器人IP地址
    :param port: 端口号，默认9559
    :return: 代理对象
    """
    try:
        proxy = ALProxy(modelName, IP, port)
        return proxy
    except Exception as e:
        print "获取代理失败: {}, 错误: {}".format(modelName, str(e))
        return None


def get_image_from_camera(camera_id, camera_proxy, videoClient):
    """
    从指定摄像头获取图像
    :param camera_id: 摄像头ID (0: 上摄像头, 1: 下摄像头)
    :param camera_proxy: 摄像头代理对象
    :param videoClient: 视频客户端ID
    :return: 图像数组
    """
    try:
        # 设置活动摄像头
        camera_proxy.setActiveCamera(camera_id)

        # 获取图片, 一帧一帧组成视频流
        # 返回的frame中：
        # frame[0]: 图像宽度
        # frame[1]: 图像高度
        # frame[2]: 图像通道数
        # frame[6]: 图像数据数组
        frame = camera_proxy.getImageRemote(videoClient)

        if frame is None:
            print "获取图像失败：frame为空"
            return None

        frameWidth = frame[0]
        frameHeight = frame[1]
        frameChannels = frame[2]

        # 将图片转换成numpy数组，并且reshape成标准的形状，方便我们使用cv2来展示
        frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frameHeight, frameWidth, frameChannels])

        return frameArray

    except Exception as e:
        print "获取摄像头图像时出错: {}".format(str(e))
        return None


def setup_camera_for_ground_detection(motionProxy):
    """
    设置摄像头角度以便于地面检测
    :param motionProxy: 运动代理对象
    """
    try:
        # 设置头部角度，使下摄像头能够看到机器人前方的地面
        # HeadPitch: 正值向下看，负值向上看
        # HeadYaw: 正值向左转，负值向右转
        motionProxy.angleInterpolation(
            ["HeadPitch", "HeadYaw"],
            [0.4, 0.0],  # 头部稍微向下，朝向正前方
            [1.0, 1.0],  # 1秒内完成动作
            True
        )
        print "摄像头角度设置完成，适用于地面检测"
    except Exception as e:
        print "设置摄像头角度失败: {}".format(str(e))


def validate_camera_connection(IP):
    """
    验证摄像头连接
    :param IP: 机器人IP地址
    :return: True如果连接成功，False否则
    """
    try:
        vd_proxy = get_Proxy("ALVideoDevice", IP)
        if vd_proxy is None:
            return False

        # 尝试获取摄像头信息
        camera_info = vd_proxy.getCameraNames()
        print "可用摄像头: {}".format(camera_info)
        return True

    except Exception as e:
        print "摄像头连接验证失败: {}".format(str(e))
        return False


def test_camera_capture(IP, camera_id=1, duration=5):
    """
    测试摄像头捕获功能
    :param IP: 机器人IP地址
    :param camera_id: 摄像头ID (0: 上摄像头, 1: 下摄像头)
    :param duration: 测试持续时间（秒）
    """
    try:
        vd_proxy = get_Proxy("ALVideoDevice", IP)
        if vd_proxy is None:
            print "无法获取视频设备代理"
            return

        # 订阅摄像头
        video_client = vd_proxy.subscribeCamera(
            "test_client",
            camera_id,
            CONFIG["resolution"],
            CONFIG["colorSpace"],
            CONFIG["fps"],
        )

        print "开始测试摄像头 {} ({})，持续 {} 秒...".format(
            camera_id,
            "下摄像头" if camera_id == 1 else "上摄像头",
            duration
        )

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            frame = get_image_from_camera(camera_id, vd_proxy, video_client)
            if frame is not None:
                frame_count += 1
                cv2.imshow("Camera Test", frame)

                # 显示帧信息
                cv2.putText(frame,
                            "Frame: {} | Camera: {}".format(frame_count, camera_id),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if cv2.waitKey(30) == 27:  # ESC键退出
                    break
            else:
                print "获取帧失败"

        print "测试完成，共捕获 {} 帧".format(frame_count)

        # 清理资源
        vd_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()

    except Exception as e:
        print "摄像头测试失败: {}".format(str(e))


if __name__ == '__main__':
    import time

    print "正在验证摄像头连接..."
    if validate_camera_connection(CONFIG["ip"]):
        print "摄像头连接验证成功"

        # 测试下摄像头
        print "测试下摄像头（用于地面检测）..."
        test_camera_capture(CONFIG["ip"], camera_id=1, duration=5)
    else:
        print "摄像头连接验证失败，请检查IP地址和网络连接"