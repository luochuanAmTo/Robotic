# coding:utf-8
import math
import random
import time

from kick_ball import kick_ball
from proxy_and_image import *
from recognized_ball import *
from control_nao import change_the_postion
from detect_goal_and_obstacle import process_goal_image
from fsm import AlignmentStateMachine

frameHeight = 0
frameWidth = 0
frameChannels = 0
frameArray = None
cameraPitchRange = 47.64 / 180 * math.pi
cameraYawRange = 60.97 / 180 * math.pi

row = "HeadPitch"
angle = 0.5235987755982988
maxstepx = 0.04
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
    motionProxy.stiffnessInterpolation("Body", 1, 1)
    motionProxy.angleInterpolation(["HeadPitch", "HeadYaw"], [0, 0], [0.3, 0.3], True)

    postureProxy = get_Proxy("ALRobotPosture", IP)
    postureProxy.goToPosture("StandInit", 1.0)

    return motionProxy, postureProxy

# 视觉处理函数
def process_vision(vd_proxy, mt_proxy, postureProxy, config):
    ball_video_client = vd_proxy.subscribeCamera(
        "ball_" + str(random.random()),
        1,  # 下部摄像头
        config["resolution"],
        config["colorSpace"],
        config["fps"],
    )

    goal_video_client = vd_proxy.subscribeCamera(
        "goal_" + str(random.random()),
        0,  # 上部摄像头
        config["resolution"],
        config["colorSpace"],
        config["fps"],
    )

    head_min_angle = -0.5  # 头部最左侧角度
    head_max_angle = 0.5  # 头部最右侧角度
    head_step = 0.1  # 每次转动的角度
    head_angle = 0  # 初始化头部角度
    direction = 1  # 初始化转头方向，1 表示向右，-1 表示向左

    try:
        while True:
            ball_img = get_image_from_camera(1, vd_proxy, ball_video_client)
            goal_img = get_image_from_camera(0, vd_proxy, goal_video_client)

            # 图像处理逻辑
            width = ball_img.shape[1]
            height = ball_img.shape[0]
            #print("Image width: , height: ", width, height)
            # 确保阈值参数是 NumPy 数组
            low_black = np.array(config["black_low"])
            high_black = np.array(config["black_high"])
            low_white = np.array(config["white_low"])
            high_white = np.array(config["white_high"])
            low_yellow = np.array(config["yellow_low"])
            high_yellow = np.array(config["yellow_high"])

            # 在图像上绘制范围边界线
            x_min_line = int(width / 2)
            #x_max_line = int(width * 14 / 25)
            x_max_line = int(width * 2 / 3)
            y_min_line = int(height * 4 / 7)
            y_max_line = int(height * 4 / 5)

            cv2.line(ball_img, (x_min_line, 0), (x_min_line, height), (255, 0, 0), 2)
            cv2.line(ball_img, (x_max_line, 0), (x_max_line, height), (255, 0, 0), 2)
            cv2.line(ball_img, (0, y_min_line), (width, y_min_line), (255, 0, 0), 2)
            cv2.line(ball_img, (0, y_max_line), (width, y_max_line), (255, 0, 0), 2)

            # 霍夫圆检测,返回结果为一个圆心的坐标(类型：tuple), 半径(类型double)
            cir_center, radius, combined_mask = detect_circle(
                ball_img,
                low_black,
                high_black,
                low_white,
                high_white,
            )

            # [0]为坐标x，[1]为坐标y
            if cir_center is not None:
                # 打印检测到的球的半径
                print("Detected ball radius: ", radius)
                # 标注球心位置
                cv2.circle(ball_img, (int(cir_center[0]), int(cir_center[1])), 5, (0, 0, 255), -1)

                # 找到球后，记录当前头部角度
                found_angle = head_angle
                # 第一象限，球在左上方
                if height * 4 / 7 - cir_center[1] > 0:
                    print("向前走")
                    mt_proxy.moveTo(0.05, 0, 0)
                    if width / 2 - cir_center[0] > 0:
                        print("左")
                        mt_proxy.moveTo(0, 0.04, 0)
                    elif width * 3 / 4 - cir_center[0] < 0:
                        print("右")
                        mt_proxy.moveTo(0, -0.03, 0)
                elif width / 2 - cir_center[0] > 0 > height * 4 / 7 - cir_center[1]:
                    print("后退，左！")
                    mt_proxy.moveTo(-0.03, 0.03, 0)
                elif (
                        width * 14 / 25 - cir_center[0] < 0
                        and height * 4 / 7 - cir_center[1] < 0
                ):
                    print("后退，右！")
                    mt_proxy.moveTo(-0.04, -0.04, 0)
                elif height * 4 / 5 - cir_center[1] < 0:
                    print("后退！")
                    mt_proxy.moveTo(-0.02, 0, 0)
                elif (
                            width / 2 - cir_center[0] < 0 and height * 4 / 7 - cir_center[1] < 0
                            and width * 2 / 3 - cir_center[0] > 0
                            and height * 4 / 5 - cir_center[1] > 0
                    ):
                    # 确认球在适当的距离内

                        #goal_img = get_image_from_camera(0, vd_proxy, goal_video_client)
                        cv2.imshow("Goal Image", goal_img)
                        cv2.waitKey(1)

                        goal_img, mid_x = process_goal_image(goal_img, low_yellow, high_yellow)


                        # if mid_x is not None:
                        #     # 根据中点调整机器人位置，使机器人、球和中点在一条直线上
                        #     if cir_center[0] - mid_x > 80:  # 加入一定的容错范围
                        #         print("向右转")
                        #         mt_proxy.moveTo(0, 0, -0.03)
                        #         #break  # 转动后退出循环，进行球的位置调整
                        #     elif mid_x - cir_center[0] > 80:
                        #         print("向左转")
                        #         mt_proxy.moveTo(0, 0, 0.03)
                        #         #break  # 转动后退出循环，进行球的位置调整
                        #     else:
                        #         print("球门角度正确")
                        #         cv2.imwrite("goal.jpg", goal_img)
                        #
                        # else:
                        #     print("未能识别到球门或障碍物")
                        #     cv2.imwrite("failgoal.jpg", goal_img)

                        # 假设 mt_proxy 和 goal_img 已经被初始化
                        #alignment_sm = AlignmentStateMachine(mt_proxy)
                        #alignment_sm = AlignmentStateMachine(mt_proxy, low_yellow, high_yellow)

                        alignment_sm = AlignmentStateMachine(
                            mt_proxy=mt_proxy,
                            low_yellow=low_yellow,
                            high_yellow=high_yellow,
                            head_min_angle=head_min_angle,
                            head_max_angle=head_max_angle,
                            head_step=head_step
                        )

                # 在合适的地方调用 update 方法，并传入 goal_img
                        alignment_sm.update(cir_center, mid_x, goal_img)

                        # 恢复头部角度
                        change_the_postion(mt_proxy, row, angle)

                        # 调整姿势回到站立状态
                        postureProxy = get_Proxy("ALRobotPosture", CONFIG["ip"])
                        postureProxy.goToPosture("Stand", 0.5)

                        # 停2秒
                        time.sleep(2)
                        kick_ball(mt_proxy)

                        # 第四象限，球在右脚下方，由于我们的预设程序是右脚踢球，执行踢球程序，并根据实际情况进行微调，
                        # cv2.imwrite方法保存下识别结果图片
                        mt_proxy.stiffnessInterpolation("Body", 1, 1)
                        mt_proxy.angleInterpolation(
                            ["HeadPitch", "HeadYaw"], [0, 0], [0.3, 0.3], True
                        )

                        print("保存图片成功！")
                        cv2.imwrite("ball.jpg", ball_img)
                        break

            else:
                # 没有找到球时，转动头部寻找
                head_angle += direction * head_step
                if head_angle > head_max_angle or head_angle < head_min_angle:
                    direction *= -1  # 改变转动方向
                    head_angle += direction * head_step  # 修正角度到有效范围内
                change_the_postion(mt_proxy, "HeadYaw", head_angle)
                print("转头寻找: ", head_angle)

                change_the_postion(mt_proxy, row, angle)


            # 显示原始图像和掩码
            result = cv2.bitwise_and(ball_img, ball_img, mask=combined_mask)

            cv2.imshow("ball", ball_img)
            cv2.waitKey(1)
            # 根据处理结果执行相应动作...
    finally:
        vd_proxy.unsubscribe(ball_video_client)
        vd_proxy.unsubscribe(goal_video_client)
        print("ok")

# 主函数
def main():
    motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
    motionProxy.setSmartStiffnessEnabled(1)
    change_the_postion(motionProxy, row, angle)

    motionProxy.moveTo(
        0.3,
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

    motionProxy.moveTo(
        0,
        0,
        -0.35,
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

    motionProxy.moveTo(0, 0.05, 0)

    vd_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])

    process_vision(vd_proxy, motionProxy, postureProxy, CONFIG)

    print("完成任务")

if __name__ == "__main__":
    main()
