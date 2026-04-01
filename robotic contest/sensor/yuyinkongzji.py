# -*- coding: utf-8 -*-
"""
NAO机器人键盘控制程序
Python 2.7 版本
通过键盘按键控制NAO机器人动作
按下1键：抬起左臂
按下2键：抬起右臂
"""

import time
import sys
import math
from naoqi import ALProxy
import msvcrt  # Windows键盘输入
import threading


class NAOKeyboardController:
    def __init__(self, robot_ip="192.168.1.100", robot_port=9559):
        """
        初始化NAO键盘控制器

        Args:
            robot_ip: NAO机器人的IP地址
            robot_port: NAO机器人的端口号（默认9559）
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.last_speech_time = 0  # 上次说话时间
        self.speech_cooldown = 1  # 语音冷却时间（秒）
        self.arm_lifted = {"left": False, "right": False}  # 胳膊抬起状态

        # 初始化代理
        try:
            self.tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
            self.motion = ALProxy("ALMotion", robot_ip, robot_port)  # 运动代理
            self.posture = ALProxy("ALRobotPosture", robot_ip, robot_port)  # 姿势代理
            print("成功连接到NAO机器人: {0}:{1}".format(robot_ip, robot_port))
        except Exception as e:
            print("连接NAO机器人失败: {0}".format(e))
            sys.exit(1)

        # 键盘按键映射
        self.key_actions = {
            '1': {
                'name': "抬起左臂",
                'action': self.lift_left_arm
            },
            '2': {
                'name': "抬起右臂",
                'action': self.lift_right_arm
            },
            '3': {
                'name': "放下左臂",
                'action': self.lower_left_arm
            },
            '4': {
                'name': "放下右臂",
                'action': self.lower_right_arm
            },
            '5': {
                'name': "放下双臂",
                'action': self.reset_arms
            },
            'q': {
                'name': "退出程序",
                'action': self.cleanup_and_exit
            }
        }

        # 初始化机器人姿势
        self.initialize_robot()

    def initialize_robot(self):
        """
        初始化机器人姿势
        """
        try:
            # 唤醒机器人
            self.motion.wakeUp()
            time.sleep(0.5)

            # 站直
            self.posture.goToPosture("Stand", 0.5)
            time.sleep(1)

            # 初始胳膊位置（放下状态）
            self.reset_arms()
            print("机器人初始化完成")

        except Exception as e:
            print("初始化机器人失败: {0}".format(e))

    def reset_arms(self):
        """
        重置胳膊到初始位置（放下状态）
        """
        try:
            # 设置胳膊位置（放下状态）
            # 左胳膊关节角度（弧度）
            left_arm_joints = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
            left_arm_angles = [1.5, 0.15, -1.0, -0.5]  # 放下姿势

            # 右胳膊关节角度（弧度）
            right_arm_joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
            right_arm_angles = [1.5, -0.15, 1.0, 0.5]  # 放下姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 移动胳膊到初始位置
            self.motion.setAngles(left_arm_joints, left_arm_angles, fraction_max_speed)
            self.motion.setAngles(right_arm_joints, right_arm_angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted = {"left": False, "right": False}

            time.sleep(0.5)
            self.speak_response("双臂已放下")

        except Exception as e:
            print("重置胳膊失败: {0}".format(e))

    def lift_left_arm(self):
        """
        抬起左胳膊
        """
        try:
            if self.arm_lifted["left"]:
                self.speak_response("左胳膊已经抬起来了")
                return

            # 左胳膊抬起位置
            joints = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
            angles = [0.0, 0.3, -1.0, -0.3, -1.0]  # 抬起姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 抬起胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted["left"] = True

            print("左胳膊已抬起")
            self.speak_response("好的，抬起左臂")

        except Exception as e:
            print("抬起左胳膊失败: {0}".format(e))

    def lift_right_arm(self):
        """
        抬起右胳膊
        """
        try:
            if self.arm_lifted["right"]:
                self.speak_response("右胳膊已经抬起来了")
                return

            # 右胳膊抬起位置
            joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
            angles = [0.0, -0.3, 1.0, 0.3, 1.0]  # 抬起姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 抬起胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted["right"] = True

            print("右胳膊已抬起")
            self.speak_response("好的，抬起右臂")

        except Exception as e:
            print("抬起右胳膊失败: {0}".format(e))

    def lower_left_arm(self):
        """
        放下左胳膊
        """
        try:
            if not self.arm_lifted["left"]:
                self.speak_response("左胳膊已经放下了")
                return

            # 左胳膊放下位置
            joints = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
            angles = [1.5, 0.15, -1.0, -0.5]  # 放下姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 放下胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted["left"] = False

            print("左胳膊已放下")
            self.speak_response("左臂放下")

        except Exception as e:
            print("放下左胳膊失败: {0}".format(e))

    def lower_right_arm(self):
        """
        放下右胳膊
        """
        try:
            if not self.arm_lifted["right"]:
                self.speak_response("右胳膊已经放下了")
                return

            # 右胳膊放下位置
            joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
            angles = [1.5, -0.15, 1.0, 0.5]  # 放下姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 放下胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted["right"] = False

            print("右胳膊已放下")
            self.speak_response("右臂放下")

        except Exception as e:
            print("放下右胳膊失败: {0}".format(e))

    def speak_response(self, text):
        """
        使用NAO的语音合成说出文本
        """
        try:
            current_time = time.time()
            if current_time - self.last_speech_time > self.speech_cooldown:
                self.tts.say(text)
                self.last_speech_time = current_time
        except Exception as e:
            print("语音播报失败: {0}".format(e))

    def check_keyboard_input(self):
        """
        检查键盘输入
        """
        try:
            # 检查是否有按键按下
            if msvcrt.kbhit():
                # 获取按键
                key = msvcrt.getch().lower()

                # 检查是否是有效的按键
                if key in self.key_actions:
                    action_info = self.key_actions[key]

                    # 显示按键信息
                    print("[{0}] 按下按键: {1} - {2}".format(
                        time.strftime("%H:%M:%S"),
                        key,
                        action_info['name']
                    ))

                    # 执行动作
                    action_info['action']()

                    return True

        except Exception as e:
            print("读取键盘输入失败: {0}".format(e))

        return False

    def start_keyboard_monitoring(self, update_interval=0.1):
        """
        开始键盘监测

        Args:
            update_interval: 更新间隔（秒）
        """
        print("=" * 60)
        print("NAO机器人键盘控制系统")
        print("=" * 60)
        print("可用按键:")
        print("1: 抬起左臂")
        print("2: 抬起右臂")
        print("3: 放下左臂")
        print("4: 放下右臂")
        print("5: 放下双臂（重置）")
        print("q: 退出程序")
        print("=" * 60)
        print("等待键盘输入...\n")

        try:
            while True:
                self.check_keyboard_input()
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n程序被中断")
            self.cleanup()

    def display_status(self):
        """
        显示当前状态
        """
        print("\n" + "=" * 40)
        print("当前状态:")
        print("左臂状态: {0}".format("抬起" if self.arm_lifted["left"] else "放下"))
        print("右臂状态: {0}".format("抬起" if self.arm_lifted["right"] else "放下"))
        print("=" * 40)

    def cleanup_and_exit(self):
        """
        清理资源并退出程序
        """
        self.cleanup()
        print("程序退出")
        sys.exit(0)

    def cleanup(self):
        """
        清理资源，重置机器人姿势
        """
        try:
            print("清理资源...")
            # 放下所有胳膊
            if self.arm_lifted["left"]:
                self.lower_left_arm()
            if self.arm_lifted["right"]:
                self.lower_right_arm()

            # 让机器人休息
            self.motion.rest()
            print("清理完成")

        except Exception as e:
            print("清理资源时出错: {0}".format(e))


def main():
    """
    主函数
    """
    # 设置NAO机器人的IP地址
    # 请根据实际网络设置修改IP地址
    robot_ip = "192.168.43.247"  # 替换为您的NAO机器人的实际IP

    try:
        # 创建键盘控制器
        controller = NAOKeyboardController(robot_ip=robot_ip)

        # 显示状态
        controller.display_status()

        # 显示使用说明
        print("\n控制说明:")
        print("按 '1' 键: 机器人会说'好的，抬起左臂'并抬起左臂")
        print("按 '2' 键: 机器人会说'好的，抬起右臂'并抬起右臂")
        print("按 '3' 键: 放下左臂")
        print("按 '4' 键: 放下右臂")
        print("按 '5' 键: 重置双臂")
        print("按 'q' 键: 退出程序")
        print("\n请按下相应按键控制机器人...")

        # 开始键盘监测
        controller.start_keyboard_monitoring(update_interval=0.1)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print("程序运行出错: {0}".format(e))


if __name__ == "__main__":
    main()