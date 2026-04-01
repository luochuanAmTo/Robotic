# coding:utf-8
import math
import random
import time
import numpy as np
from threading import Thread, Lock

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

stepheight = 0.05
torsowx = 0.0
torsowy = 0.0

# 目标路径点
TARGET_POINTS = [
    (0.6, 1.5),
    (-0.6, 3.0),
    (0.6, 4.5),
    (-0.6, 6.0)
]

# 场地参数
FIELD_WIDTH = 2.0  # 2m宽
FIELD_LENGTH = 6.0  # 6m长

# 控制参数
POSITION_TOLERANCE = 0.05  # 位置误差容忍度 5cm
ORIENTATION_TOLERANCE = 0.1  # 朝向误差容忍度 约5.7度
MAX_CORRECTION_ANGLE = 0.2  # 最大单次朝向矫正角度


class RobotPositionTracker:
    """机器人位置跟踪器"""

    def __init__(self, motionProxy):
        self.motionProxy = motionProxy
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.lock = Lock()
        self.tracking = False

    def start_tracking(self):
        """开始位置跟踪"""
        self.tracking = True
        self.tracking_thread = Thread(target=self._update_position)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()

    def stop_tracking(self):
        """停止位置跟踪"""
        self.tracking = False

    def _update_position(self):
        """实时更新机器人位置"""
        while self.tracking:
            try:
                # 获取机器人的里程计数据
                position = self.motionProxy.getRobotPosition(True)  # True表示使用传感器融合

                with self.lock:
                    self.current_x = position[0]
                    self.current_y = position[1]
                    self.current_theta = position[2]

                time.sleep(0.1)  # 10Hz更新频率
            except Exception as e:
                print("位置更新错误: {e}")
                time.sleep(0.5)

    def get_current_position(self):
        """获取当前位置"""
        with self.lock:
            return self.current_x, self.current_y, self.current_theta

    def reset_position(self, x=0.0, y=0.0, theta=0.0):
        """重置位置"""
        with self.lock:
            self.current_x = x
            self.current_y = y
            self.current_theta = theta


def calculate_distance(x1, y1, x2, y2):
    """计算两点间距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_angle(angle):
    """将角度归一化到[-pi, pi]范围"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_target_orientation(current_x, current_y, target_x, target_y):
    """计算到目标点的朝向角度"""
    dx = target_x - current_x
    dy = target_y - current_y
    return math.atan2(dy, dx)


def move_to_target_with_correction(motionProxy, tracker, target_x, target_y):
    """移动到目标点并进行实时偏航矫正"""
    print("开始移动到目标点: ({target_x:.2f}, {target_y:.2f})")

    while True:
        # 获取当前位置
        current_x, current_y, current_theta = tracker.get_current_position()

        # 计算到目标的距离
        distance = calculate_distance(current_x, current_y, target_x, target_y)

        print("当前位置: ({current_x:.3f}, {current_y:.3f}, {math.degrees(current_theta):.1f}°), "
              "距离目标: {distance:.3f}m")

        # 检查是否到达目标
        if distance < POSITION_TOLERANCE:
            print("到达目标点: ({target_x:.2f}, {target_y:.2f})")
            motionProxy.stopWalk()
            break

        # 计算目标朝向
        target_theta = calculate_target_orientation(current_x, current_y, target_x, target_y)

        # 计算朝向偏差
        theta_error = normalize_angle(target_theta - current_theta)

        print("目标朝向: {math.degrees(target_theta):.1f}°, "
              "朝向偏差: {math.degrees(theta_error):.1f}°")

        # 如果朝向偏差过大，先进行朝向矫正
        if abs(theta_error) > ORIENTATION_TOLERANCE:
            print("进行朝向矫正...")
            correction_angle = max(-MAX_CORRECTION_ANGLE,
                                   min(MAX_CORRECTION_ANGLE, theta_error))

            # 原地转向矫正
            motionProxy.moveTo(0, 0, correction_angle, [
                ["MaxStepTheta", 0.3],
                ["MaxStepFrequency", 0.5]
            ])

            time.sleep(0.5)  # 等待转向完成
            continue

        # 计算移动步长
        step_distance = min(distance, 0.2)  # 最大单步0.2m
        step_x = step_distance * math.cos(target_theta - current_theta)
        step_y = step_distance * math.sin(target_theta - current_theta)

        # 执行移动
        motionProxy.moveTo(step_x, step_y, 0, [
            ["MaxStepX", maxstepx],
            ["MaxStepY", maxstepy],
            ["MaxStepTheta", maxsteptheta],
            ["MaxStepFrequency", maxstepfrequency],
            ["StepHeight", stepheight],
            ["TorsoWx", torsowx],
            ["TorsoWy", torsowy],
        ])

        time.sleep(0.5)  # 等待步伐完成


def correct_orientation_to_forward(motionProxy, tracker):
    """矫正朝向到正前方（沿Y轴正方向）"""
    print("矫正朝向到正前方...")
    current_x, current_y, current_theta = tracker.get_current_position()
    target_theta = 0.0  # 正前方
    theta_error = normalize_angle(target_theta - current_theta)

    if abs(theta_error) > ORIENTATION_TOLERANCE:
        motionProxy.moveTo(0, 0, theta_error, [
            ["MaxStepTheta", 0.3],
            ["MaxStepFrequency", 0.5]
        ])
        time.sleep(1.0)


def initialize_robot(IP):
    """初始化机器人"""
    AutonomousLifeProxy = get_Proxy("ALAutonomousLife", IP)
    AutonomousLifeProxy.setState("disabled")

    motionProxy = get_Proxy("ALMotion", IP)
    motionProxy.stiffnessInterpolation("Body", 1, 1.5)
    motionProxy.angleInterpolation(["HeadPitch", "HeadYaw"], [0, 0], [0.3, 0.3], True)

    postureProxy = get_Proxy("ALRobotPosture", IP)
    postureProxy.goToPosture("StandInit", 1.5)

    return motionProxy, postureProxy


def execute_path_mission(motionProxy, tracker):
    """执行完整的路径任务"""
    print("=" * 50)
    print("开始执行路径任务")
    print("场地大小: {FIELD_WIDTH}m x {FIELD_LENGTH}m")
    print("目标路径: {TARGET_POINTS}")
    print("=" * 50)

    # 重置位置到起始点
    tracker.reset_position(0.0, 0.0, 0.0)

    # 依次走到每个目标点
    for i, (target_x, target_y) in enumerate(TARGET_POINTS):
        print("\n--- 阶段 {i + 1}: 前往 ({target_x}, {target_y}) ---")

        # 移动到目标点
        move_to_target_with_correction(motionProxy, tracker, target_x, target_y)

        # 到达后矫正朝向到正前方
        correct_orientation_to_forward(motionProxy, tracker)

        # 短暂停留
        time.sleep(1.0)

        current_x, current_y, current_theta = tracker.get_current_position()
        print("阶段 {i + 1} 完成，当前位置: ({current_x:.3f}, {current_y:.3f}, {math.degrees(current_theta):.1f}°)")

    print("\n" + "=" * 50)
    print("路径任务全部完成！")
    print("=" * 50)


if __name__ == "__main__":
    try:
        # 初始化机器人
        motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
        motionProxy.setSmartStiffnessEnabled(1)

        # 设置头部角度
        change_the_postion(motionProxy, row, angle)

        # 创建位置跟踪器
        tracker = RobotPositionTracker(motionProxy)

        # 开始位置跟踪
        tracker.start_tracking()
        time.sleep(1.0)  # 等待跟踪器启动

        # 执行路径任务
        execute_path_mission(motionProxy, tracker)

        # 停止位置跟踪
        tracker.stop_tracking()

        # 最终停止
        motionProxy.stopWalk()
        print("任务完成，机器人已停止")

    except Exception as e:
        print("程序执行错误: {e}")
        if 'tracker' in locals():
            tracker.stop_tracking()
        if 'motionProxy' in locals():
            motionProxy.stopWalk()