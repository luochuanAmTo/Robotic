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

# 稍微增大步幅参数用于避障
maxstepx = 0.12  # 增加了0.02
maxstepy = 0.13  # 增加了0.02
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


def get_avoidance_directions():
    """获取用户输入的避障方向序列"""
    print("请输入3次避障的方向序列:")
    print("l = 向左避障, r = 向右避障")
    print("例如: lrl (先左后右再左)")

    while True:
        user_input = raw_input("请输入避障序列: ").strip().lower()

        if len(user_input) == 3 and all(c in ['l', 'r'] for c in user_input):
            directions = []
            for c in user_input:
                if c == 'l':
                    directions.append('left')
                else:
                    directions.append('right')
            print("避障序列设置为: {}".format(' -> '.join(directions)))
            return directions
        else:
            print("输入错误！请输入3个字符，只能是l或r")


# 视觉处理函数
def process_vision(vd_proxy, mt_proxy, posture_proxy, avoidance_directions):
    # 订阅摄像头
    video_client = vd_proxy.subscribeCamera(
        "vision_" + str(random.random()),
        0,  # 使用上部摄像头
        CONFIG["resolution"],
        CONFIG["colorSpace"],
        CONFIG["fps"],
    )

    # 运动参数配置
    BASE_SPEED = 0.3  # 基础前进速度
    LATERAL_DIST = 0.5  # 横向移动距离（避障距离）
    MIN_WIDTH = 60  # 触发避障的最小宽度

    # 状态变量
    current_avoidance = 0  # 当前避障次数 (0-2)
    is_avoiding = False  # 是否正在避障
    avoidance_completed = False  # 避障动作是否完成
    last_detect_time = time.time()  # 最后检测时间

    print("开始巡航，寻找障碍物...")
    print("避障序列: {}".format(' -> '.join(avoidance_directions)))

    try:
        while current_avoidance < 3:  # 完成3次避障
            # 获取图像帧
            frame = get_image_from_camera(0, vd_proxy, video_client)
            if frame is None:
                print("获取图像失败")
                continue

            # 执行柱体检测
            processed_img, pillars = detect_cylinder(frame, color_ranges)

            # 寻找任何足够大的障碍物
            obstacle = None
            for pillar in pillars:
                if pillar['width'] >= MIN_WIDTH:
                    obstacle = pillar
                    break

            # 可视化标注
            if obstacle and not is_avoiding:
                cv2.putText(processed_img,
                            "Obstacle Detected - Avoid {}".format(avoidance_directions[current_avoidance]),
                            (obstacle['center'][0] - 80, obstacle['center'][1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示当前避障进度
            cv2.putText(processed_img,
                        "Avoidance: {}/3".format(current_avoidance),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 避障逻辑
            if obstacle and not is_avoiding and not avoidance_completed:
                print("第{}次避障: 检测到障碍物，开始向{}避障".format(
                    current_avoidance + 1, avoidance_directions[current_avoidance]))

                # 执行避障动作
                if avoidance_directions[current_avoidance] == 'left':
                    # 向左避障
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
                else:
                    # 向右避障
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

                is_avoiding = True
                avoidance_completed = True
                last_detect_time = time.time()

                # 避障后继续前进一段距离
                print("避障完成，继续前进...")
                mt_proxy.moveTo(1.5,  # 前进距离
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

                # 更新避障计数
                current_avoidance += 1
                is_avoiding = False
                avoidance_completed = False

                print("第{}次避障完成，等待下一个障碍物...".format(current_avoidance))
                time.sleep(1)  # 短暂停顿

            else:
                # 没有检测到障碍物或正在避障时，正常前进
                if not is_avoiding:
                    mt_proxy.moveTo(BASE_SPEED,
                                    0,
                                    0,
                                    [
                                        ["MaxStepX", 0.10],  # 正常前进时使用原始步幅
                                        ["MaxStepY", 0.11],
                                        ["MaxStepTheta", maxsteptheta],
                                        ["MaxStepFrequency", maxstepfrequency],
                                        ["StepHeight", stepheight],
                                        ["TorsoWx", torsowx],
                                        ["TorsoWy", torsowy],
                                    ],
                                    )

            # 超时重置（防止卡死）
            if time.time() - last_detect_time > 15:
                print("长时间未检测到障碍物，重置状态")
                is_avoiding = False
                avoidance_completed = False
                last_detect_time = time.time()

            # 显示处理结果
            cv2.imshow("Adaptive Obstacle Avoidance", processed_img)
            if cv2.waitKey(30) == 27:  # ESC退出
                break

        print("所有避障任务完成！")

        # 最终前进
        print("继续前进...")
        mt_proxy.moveTo(2.0,
                        0,
                        0,
                        [
                            ["MaxStepX", 0.10],
                            ["MaxStepY", 0.11],
                            ["MaxStepTheta", maxsteptheta],
                            ["MaxStepFrequency", maxstepfrequency],
                            ["StepHeight", stepheight],
                            ["TorsoWx", torsowx],
                            ["TorsoWy", torsowy],
                        ],
                        )

    except Exception as e:
        print("运行时错误: {}".format(str(e)))
    finally:
        # 释放资源
        vd_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()
        print("视觉处理结束")


# 使用示例
if __name__ == "__main__":
    # 获取避障方向序列
    avoidance_directions = get_avoidance_directions()

    # 初始化机器人
    motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
    motionProxy.setSmartStiffnessEnabled(1)
    change_the_postion(motionProxy, row, angle)

    # 初始前进
    print("开始初始前进...")
    motionProxy.moveTo(
        1,
        0,
        0.1,
        [
            ["MaxStepX", 0.10],  # 使用原始步幅
            ["MaxStepY", 0.11],
            ["MaxStepTheta", maxsteptheta],
            ["MaxStepFrequency", maxstepfrequency],
            ["StepHeight", stepheight],
            ["TorsoWx", torsowx],
            ["TorsoWy", torsowy],
        ],
    )

    # 开始视觉处理和避障
    vd_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
    process_vision(vd_proxy, motionProxy, postureProxy, avoidance_directions)