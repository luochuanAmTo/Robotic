# coding:utf-8
import math
import random
import time

from proxy_and_image import *
from recognized_cylinder import *
from control_nao import change_the_postion

frameHeight = 0
frameWidth = 0
frameChannels = 0
frameArray = None
cameraPitchRange = 47.64 / 180 * math.pi
cameraYawRange = 60.97 / 180 * math.pi

row = "HeadPitch"
angle = 0.5235987755982988
maxstepx = 0.10
maxstepy = 0.11
maxsteptheta = 0.3
maxstepfrequency = 0.6

stepheight = 0.02
torsowx = 0.0
torsowy = 0.0

# 初始化函数
def initialize_robot(IP):

    AutonomousLifeProxy = get_Proxy("ALAutonomousLife", IP)
    AutonomousLifeProxy.setState("disabled")

    motionProxy = get_Proxy("ALMotion", IP)
    motionProxy.stiffnessInterpolation("Body", 1, 1.5)
    motionProxy.angleInterpolation(["HeadPitch", "HeadYaw"], [0, 0], [0.3, 0.3], True)

    postureProxy = get_Proxy("ALRobotPosture", IP)
    postureProxy.goToPosture("StandInit", 1.5)

    return motionProxy, postureProxy

#视觉处理函数
def process_vision(vd_proxy, mt_proxy, posture_proxy):

    # 订阅摄像头
    video_client = vd_proxy.subscribeCamera(
        "vision_" + str(random.random()),
        0,  # 使用上部摄像头
        CONFIG["resolution"],
        CONFIG["colorSpace"],
        CONFIG["fps"],
    )

    # 运动参数配置
    BASE_SPEED = 1.5  # 基础前进速度
    LATERAL_DIST = 0.5  # 横向移动距离
    MIN_WIDTH = 60  # 触发横向移动的最小宽度
    TARGET_ORDER = ['red', 'blue', 'yellow']  # 目标顺序

    # 状态变量
    current_stage = 0  # 当前阶段 0-红 1-蓝 2-黄+
    is_movement_done = False  # 移动完成标记
    moveto_done=False
    last_detect_time = time.time()  # 最后检测时间

    try:
        while True:
            # 获取图像帧
            frame = get_image_from_camera(0, vd_proxy, video_client)
            if frame is None:
                print("获取图像失败")
                continue

            # 执行柱体检测
            processed_img, pillars = detect_cylinder(frame, color_ranges)

            # 寻找当前阶段目标
            current_target = next(
                (p for p in pillars
                 if p['color'] == TARGET_ORDER[current_stage]
                 and p['width'] >= MIN_WIDTH),
                None
            )

            # 可视化标注
            if current_target:
                cv2.putText(processed_img,
                            "{} Pillar".format(TARGET_ORDER[current_stage]),
                            (current_target['center'][0] - 50, current_target['center'][1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 运动控制逻辑
            if current_target and not is_movement_done:
                # 根据阶段执行横向移动
                if current_stage in [0, 2]:  # 红、黄阶段左移
                    print("Stage {}: 向左横向移动".format(current_stage + 1))
                    mt_proxy.moveTo(0,
                                    LATERAL_DIST,
                                    0,
                                    [
                                        ["MaxStepX", maxstepx],
                                        ["MaxStepY", maxstepy],
                                        ["MaxStepTheta", maxsteptheta],
                                        ["MaxStepFrequency", maxstepfrequency],
                                        ["StepHeight", stepheight],
                                        ["TorsoWx", torsowx],
                                        ["TorsoWy", torsowy],
                                    ],
                                    )
                elif current_stage == 1:  # 蓝阶段右移
                    print("Stage {}: 向右横向移动".format(current_stage + 1))
                    mt_proxy.moveTo(0,
                                    -LATERAL_DIST,
                                    0,
                                    [
                                        ["MaxStepX", maxstepx],
                                        ["MaxStepY", maxstepy],
                                        ["MaxStepTheta", maxsteptheta],
                                        ["MaxStepFrequency", maxstepfrequency],
                                        ["StepHeight", stepheight],
                                        ["TorsoWx", torsowx],
                                        ["TorsoWy", torsowy],
                                    ],
                                    )

                is_movement_done = True
                moveto_done=True
                last_detect_time = time.time()


                if moveto_done is True:
                    mt_proxy.moveTo(BASE_SPEED,
                                    0,
                                    0,
                                    [
                                        ["MaxStepX", maxstepx],
                                        ["MaxStepY", maxstepy],
                                        ["MaxStepTheta", maxsteptheta],
                                        ["MaxStepFrequency", maxstepfrequency],
                                        ["StepHeight", stepheight],
                                        ["TorsoWx", torsowx],
                                        ["TorsoWy", torsowy],
                                    ],
                                    )
                # 更新阶段
                if current_stage < 2:
                    current_stage += 1
                    is_movement_done = False
                    moveto_done=False
                # 10秒无检测重置
                if time.time() - last_detect_time > 10:
                    current_stage = 0
                    is_movement_done = False
                    moveto_done=False
                    print("重置检测状态")

            else:
                mt_proxy.moveTo(0.3,
                                0,
                                0,
                                [
                                    ["MaxStepX", maxstepx],
                                    ["MaxStepY", maxstepy],
                                    ["MaxStepTheta", maxsteptheta],
                                    ["MaxStepFrequency", maxstepfrequency],
                                    ["StepHeight", stepheight],
                                    ["TorsoWx", torsowx],
                                    ["TorsoWy", torsowy],
                                ],
                                )

            # 显示处理结果
            cv2.imshow("Pillar Detection", processed_img)
            if cv2.waitKey(30) == 27:  # ESC退出
                break

    except Exception as e:
        print("运行时错误: {}".format(str(e)))
    finally:
        # 释放资源
        vd_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()




        print("视觉处理结束")


# 使用示例
if __name__ == "__main__":
    motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
    motionProxy.setSmartStiffnessEnabled(1)
    change_the_postion(motionProxy, row, angle)
    motionProxy.moveTo(
        1,
        0,
        0.1,
        [
            ["MaxStepX", maxstepx],
            ["MaxStepY", maxstepy],
            ["MaxStepTheta", maxsteptheta],
            ["MaxStepFrequency", maxstepfrequency],
            ["StepHeight", stepheight],
            ["TorsoWx", torsowx],
            ["TorsoWy", torsowy],
        ],
    )
    vd_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
    process_vision(vd_proxy, motionProxy, postureProxy)