# -*- coding: utf-8 -*-
"""
NAO机器人触觉传感器监测程序
Python 2.7 版本
监测头部、手背、脚面的按压
按压手背时会抬起相应胳膊
"""

import time
import sys
import math
from naoqi import ALProxy


class NAOTactileMonitor:
    def __init__(self, robot_ip="192.168.1.100", robot_port=9559):
        """
        初始化NAO触觉监测器

        Args:
            robot_ip: NAO机器人的IP地址
            robot_port: NAO机器人的端口号（默认9559）
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sensor_states = {}  # 存储传感器状态
        self.last_speech_time = 0  # 上次说话时间
        self.speech_cooldown = 1  # 语音冷却时间（秒）
        self.arm_lifted = {"left": False, "right": False}  # 胳膊抬起状态

        # 初始化代理
        try:
            self.memory = ALProxy("ALMemory", robot_ip, robot_port)
            self.tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
            self.motion = ALProxy("ALMotion", robot_ip, robot_port)  # 运动代理
            self.posture = ALProxy("ALRobotPosture", robot_ip, robot_port)  # 姿势代理
            print("成功连接到NAO机器人: {0}:{1}".format(robot_ip, robot_port))
        except Exception as e:
            print("连接NAO机器人失败: {0}".format(e))
            sys.exit(1)

        # 定义要监测的触觉传感器
        self.tactile_sensors = {
            # 头部传感器
            "head_front": "Device/SubDeviceList/Head/Touch/Front/Sensor/Value",
            "head_middle": "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value",
            "head_rear": "Device/SubDeviceList/Head/Touch/Rear/Sensor/Value",

            # 左手传感器
            "left_hand_back": "Device/SubDeviceList/LHand/Touch/Back/Sensor/Value",

            # 右手传感器
            "right_hand_back": "Device/SubDeviceList/RHand/Touch/Back/Sensor/Value",

            # 左脚传感器
            "left_foot_left": "Device/SubDeviceList/LFoot/Bumper/Left/Sensor/Value",
            "left_foot_right": "Device/SubDeviceList/LFoot/Bumper/Right/Sensor/Value",

            # 右脚传感器
            "right_foot_left": "Device/SubDeviceList/RFoot/Bumper/Left/Sensor/Value",
            "right_foot_right": "Device/SubDeviceList/RFoot/Bumper/Right/Sensor/Value",
        }

        # 传感器分组和对应的语音提示
        self.sensor_groups = {
            "head": {
                "name": "头部",
                "sensors": ["head_front", "head_middle", "head_rear"],
                "last_pressed": 0,
                "action": self.head_action
            },
            "left_hand": {
                "name": "左手背",
                "sensors": ["left_hand_back"],
                "last_pressed": 0,
                "action": self.left_hand_action
            },
            "right_hand": {
                "name": "右手背",
                "sensors": ["right_hand_back"],
                "last_pressed": 0,
                "action": self.right_hand_action
            },
            "left_foot": {
                "name": "左脚面",
                "sensors": ["left_foot_left", "left_foot_right"],
                "last_pressed": 0,
                "action": self.foot_action
            },
            "right_foot": {
                "name": "右脚面",
                "sensors": ["right_foot_left", "right_foot_right"],
                "last_pressed": 0,
                "action": self.foot_action
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

        except Exception as e:
            print("重置胳膊失败: {0}".format(e))

    def lift_arm(self, side="left"):
        """
        抬起指定侧的胳膊

        Args:
            side: "left" 或 "right"
        """
        try:
            if side == "left":
                # 左胳膊抬起位置
                joints = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
                angles = [0.0, 0.3, -1.0, -0.3, -1.0]  # 抬起姿势
            else:
                # 右胳膊抬起位置
                joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
                angles = [0.0, -0.3, 1.0, 0.3, 1.0]  # 抬起姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 抬起胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted[side] = True

            print("{0}胳膊已抬起".format("左" if side == "left" else "右"))

        except Exception as e:
            print("抬起{0}胳膊失败: {1}".format("左" if side == "left" else "右", e))

    def lower_arm(self, side="left"):
        """
        放下指定侧的胳膊

        Args:
            side: "left" 或 "right"
        """
        try:
            if side == "left":
                # 左胳膊放下位置
                joints = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
                angles = [1.5, 0.15, -1.0, -0.5]  # 放下姿势
            else:
                # 右胳膊放下位置
                joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
                angles = [1.5, -0.15, 1.0, 0.5]  # 放下姿势

            # 设置最大速度分数
            fraction_max_speed = 0.3

            # 放下胳膊
            self.motion.setAngles(joints, angles, fraction_max_speed)

            # 更新状态
            self.arm_lifted[side] = False

            print("{0}胳膊已放下".format("左" if side == "left" else "右"))

        except Exception as e:
            print("放下{0}胳膊失败: {1}".format("左" if side == "left" else "右", e))

    def head_action(self):
        """
        头部被按压时的动作
        """
        # 头部被按压时不需要特殊动作，只需语音响应
        pass

    def left_hand_action(self):
        """
        左手背被按压时的动作
        """
        if self.arm_lifted["left"]:
            # 如果左手已经抬起，则放下
            self.lower_arm("left")
            self.speak_response("左手放下")
        else:
            # 如果左手没有抬起，则抬起
            self.lift_arm("left")
            self.speak_response("左手抬起")

    def right_hand_action(self):
        """
        右手背被按压时的动作
        """
        if self.arm_lifted["right"]:
            # 如果右手已经抬起，则放下
            self.lower_arm("right")
            self.speak_response("右手放下")
        else:
            # 如果右手没有抬起，则抬起
            self.lift_arm("right")
            self.speak_response("右手抬起")

    def foot_action(self):
        """
        脚面被按压时的动作
        """
        # 脚面被按压时不需要特殊动作，只需语音响应
        pass

    def get_sensor_value(self, sensor_path):
        """
        获取传感器值

        Args:
            sensor_path: 传感器路径

        Returns:
            传感器值（通常0表示未按压，1表示按压）
        """
        try:
            return self.memory.getData(sensor_path)
        except Exception as e:
            print("获取传感器数据失败 {0}: {1}".format(sensor_path, e))
            return 0

    def check_tactile_sensors(self, threshold=0.5):
        """
        检查所有触觉传感器

        Args:
            threshold: 触发阈值
        """
        current_time = time.time()

        for group_name, group_info in self.sensor_groups.iteritems():
            is_pressed = False

            # 检查组内的所有传感器
            for sensor_key in group_info["sensors"]:
                sensor_path = self.tactile_sensors.get(sensor_key)
                if sensor_path:
                    value = self.get_sensor_value(sensor_path)
                    if value > threshold:
                        is_pressed = True
                        break

            # 如果是新按压，且超过冷却时间
            if (is_pressed and
                    current_time - group_info["last_pressed"] > self.speech_cooldown and
                    current_time - self.last_speech_time > self.speech_cooldown):
                # 更新按压时间
                group_info["last_pressed"] = current_time
                self.last_speech_time = current_time

                # 语音播报
                speech_text = "{0}被按压".format(group_info["name"])
                self.speak_response(speech_text)

                # 执行相应动作
                if "action" in group_info and callable(group_info["action"]):
                    group_info["action"]()

                # 显示状态
                print("[{0}] {1}".format(
                    time.strftime("%H:%M:%S"),
                    speech_text
                ))

    def speak_response(self, text):
        """
        使用NAO的语音合成说出文本
        """
        try:
            self.tts.say(text)
        except Exception as e:
            print("语音播报失败: {0}".format(e))

    def start_monitoring(self, update_interval=0.1):
        """
        开始触觉监测

        Args:
            update_interval: 更新间隔（秒）
        """
        print("=" * 60)
        print("NAO机器人触觉传感器监测系统")
        print("=" * 60)
        print("监测以下部位的按压:")
        print("1. 头部（前、中、后） - 语音响应")
        print("2. 手背（左、右手） - 语音响应并抬起/放下相应胳膊")
        print("3. 脚面（左、右脚） - 语音响应")
        print("=" * 60)
        print("开始监测...")
        print("按压机器人的相应部位:")
        print("- 按压头部/脚面: 语音提示")
        print("- 按压手背: 抬起或放下相应胳膊")
        print("按 Ctrl+C 停止监测\n")

        try:
            while True:
                self.check_tactile_sensors()
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n停止触觉监测...")
            self.cleanup()

    def test_all_sensors(self):
        """
        测试所有传感器并显示当前状态
        """
        print("测试所有触觉传感器状态:")
        print("-" * 50)

        for sensor_key, sensor_path in self.tactile_sensors.iteritems():
            try:
                value = self.memory.getData(sensor_path)
                status = "按压中" if value > 0.5 else "未按压"
                print("{0:20s}: {1:.3f} [{2}]".format(sensor_key, value, status))
            except Exception as e:
                print("{0:20s}: 读取失败 - {1}".format(sensor_key, str(e)))

        print("-" * 50)

    def continuous_pressure_test(self, duration=10, interval=0.2):
        """
        连续压力测试

        Args:
            duration: 测试持续时间（秒）
            interval: 测试间隔（秒）
        """
        print("开始连续压力测试，持续时间: {0}秒".format(duration))
        print("请按压机器人的不同部位")

        start_time = time.time()
        press_count = {
            "head": 0,
            "left_hand": 0,
            "right_hand": 0,
            "left_foot": 0,
            "right_foot": 0
        }

        try:
            while time.time() - start_time < duration:
                current_time = time.time()

                for group_name, group_info in self.sensor_groups.iteritems():
                    is_pressed = False

                    for sensor_key in group_info["sensors"]:
                        sensor_path = self.tactile_sensors.get(sensor_key)
                        if sensor_path:
                            value = self.get_sensor_value(sensor_path)
                            if value > 0.5:
                                is_pressed = True
                                break

                    if is_pressed:
                        # 如果是新按压
                        if current_time - group_info["last_pressed"] > self.speech_cooldown:
                            group_info["last_pressed"] = current_time
                            press_count[group_name] += 1

                            # 显示但不语音播报（避免太吵）
                            print("[{0:.1f}s] {1}被按压".format(
                                current_time - start_time,
                                group_info["name"]
                            ))

                time.sleep(interval)

            # 显示测试结果
            print("\n" + "=" * 50)
            print("压力测试结果:")
            for group_name, count in press_count.iteritems():
                group_info = self.sensor_groups[group_name]
                print("{0}: {1}次按压".format(group_info["name"], count))
            print("=" * 50)

            # 语音播报总结
            total_presses = sum(press_count.values())
            if total_presses > 0:
                summary = "测试完成，总共检测到{0}次按压".format(total_presses)
                self.speak_response(summary)

        except KeyboardInterrupt:
            print("\n测试被中断")

    def cleanup(self):
        """
        清理资源，重置机器人姿势
        """
        try:
            print("清理资源...")
            # 放下所有胳膊
            if self.arm_lifted["left"]:
                self.lower_arm("left")
            if self.arm_lifted["right"]:
                self.lower_arm("right")

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
        # 创建触觉监测器
        monitor = NAOTactileMonitor(robot_ip=robot_ip)

        # 测试传感器状态
        monitor.test_all_sensors()

        # 选择模式
        print("\n请选择模式:")
        print("1. 实时监测模式（语音响应并控制胳膊）")
        print("2. 压力测试模式（不语音响应）")
        print("3. 直接开始监测")

        try:
            choice = raw_input("请输入选择 (1, 2, 或直接回车选择3): ").strip()
        except:
            choice = "3"

        if choice == "1":
            # 实时监测模式
            interval = 0.1
            print("\n进入实时监测模式，机器人会语音响应并控制胳膊")
            monitor.start_monitoring(update_interval=interval)

        elif choice == "2":
            # 压力测试模式
            duration = 10
            interval = 0.2
            print("\n进入压力测试模式，机器人不会语音响应")
            monitor.continuous_pressure_test(duration=duration, interval=interval)

        else:
            # 默认模式
            interval = 0.1
            print("\n进入默认监测模式")
            monitor.start_monitoring(update_interval=interval)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print("程序运行出错: {0}".format(e))


if __name__ == "__main__":
    main()