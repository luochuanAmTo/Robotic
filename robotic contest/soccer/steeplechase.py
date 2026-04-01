#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO机器人障碍跑代码
基于NAOqi SDK实现自主避障和S型路径规划
"""

import sys
import time
import math
import numpy as np
import math
from naoqi import ALProxy
import vision_definitions
import cv2

class NAOObstacleRun:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化NAO机器人及相关模块"""
        try:
            # 初始化基本代理模块
            self.motion = ALProxy("ALMotion", ip, port)
            self.posture = ALProxy("ALRobotPosture", ip, port)
            self.memory = ALProxy("ALMemory", ip, port)
            self.tts = ALProxy("ALTextToSpeech", ip, port)
            self.video = ALProxy("ALVideoDevice", ip, port)
            self.sonar = ALProxy("ALSonar", ip, port)

            # 初始化行为管理器
            self.behavior = ALProxy("ALBehaviorManager", ip, port)

            # 初始化标志变量
            self.is_running = False
            self.start_time = 0
            self.obstacles_detected = []
            self.path_points = []

            # 摄像头参数
            self.camera_id = 0  # 0: 上摄像头, 1: 下摄像头
            self.resolution = vision_definitions.kVGA  # 640x480
            self.color_space = vision_definitions.kRGBColorSpace
            self.fps = 15

            # 订阅摄像头
            self.camera_name = "ObstacleCamera"
            self.camera_client = self.video.subscribe(
                self.camera_name,
                self.resolution,
                self.color_space,
                self.fps
            )

            # 启动声纳传感器 (辅助避障)
            self.sonar.subscribe("ObstacleSonar")

            # 设置行走参数
            self.max_step_x = 0.14  # 最大前进步长 (m)
            self.max_step_y = 0.14  # 最大侧移步长 (m)
            self.max_step_theta = 0.3  # 最大转向角度 (rad)
            self.step_frequency = 0.6  # 步频

            # 场地参数 (单位: 米)
            self.track_length = 6.0
            self.track_width = 2.0
            self.obstacle_diameter = 0.3

            # 颜色识别阈值 (RGB)
            self.color_thresholds = {
                "red": ([150, 0, 0], [255, 50, 50]),
                "blue": ([0, 0, 150], [50, 50, 255]),
                "yellow": ([150, 150, 0], [255, 255, 50])
            }

            self.tts.say("初始化完成")
            print("NAO机器人障碍跑初始化完成")

        except Exception as e:
            print("初始化失败: {}".format(e))
            sys.exit(1)

    def __del__(self):
        """清理资源"""
        try:
            # 取消订阅摄像头和声纳
            self.video.unsubscribe(self.camera_client)
            self.sonar.unsubscribe("ObstacleSonar")
            # 停止行走
            self.motion.stopMove()
        except:
            pass

    def prepare_for_run(self):
        """准备竞赛前的设置"""
        # 唤醒机器人并设置刚度
        self.motion.wakeUp()
        self.motion.setStiffnesses("Body", 1.0)

        # 调整到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

        # 配置行走参数 - 使用稳定步态
        self.motion.setMotionConfig([["MAX_STEP_X", self.max_step_x],
                                     ["MAX_STEP_Y", self.max_step_y],
                                     ["MAX_STEP_THETA", self.max_step_theta],
                                     ["MAX_STEP_FREQUENCY", self.step_frequency],
                                     ["STEP_HEIGHT", 0.02]])

        # 设置避障能力
        self.motion.setExternalCollisionProtectionEnabled("All", True)

        # 准备头部位置 - 稍微低头以便更好地看到障碍物
        self.motion.setAngles("HeadPitch", 0.2, 0.2)
        self.motion.setAngles("HeadYaw", 0.0, 0.2)

        self.tts.say("准备开始比赛")

    def detect_obstacles(self):
        """检测场地中的障碍物"""
        self.tts.say("开始检测障碍物")

        # 清空先前检测的障碍物
        self.obstacles_detected = []

        # 旋转头部扫描场地
        head_yaw_angles = [-0.8, -0.4, 0.0, 0.4, 0.8]  # 从左到右扫描

        for yaw in head_yaw_angles:
            # 移动头部
            self.motion.setAngles("HeadYaw", yaw, 0.1)
            self.motion.setAngles("HeadPitch", 0.2, 0.1)
            time.sleep(1.0)  # 等待头部移动到位

            # 获取图像
            image = self.video.getImageRemote(self.camera_client)
            if image is None:
                continue

            # 处理图像数据
            width = image[0]
            height = image[1]
            image_array = np.frombuffer(image[6], dtype=np.uint8).reshape(height, width, 3)

            # 颜色分割并检测障碍物
            for color, (lower, upper) in self.color_thresholds.items():
                # 创建颜色掩码
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(image_array, lower, upper)

                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # 计算轮廓面积，过滤噪声
                    area = cv2.contourArea(contour)
                    if area > 500:  # 面积阈值
                        # 获取轮廓中心
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # 估计障碍物距离 (基于简单的图像尺寸估计)
                            # 这里需要进一步校准以获得更准确的距离估计
                            distance = self._estimate_distance(area)

                            # 估计障碍物位置 (x, y, z)坐标
                            obstacle_position = self._image_to_world(cx, cy, distance, yaw)

                            # 存储障碍物信息
                            obstacle = {
                                "color": color,
                                "position": obstacle_position,
                                "diameter": self.obstacle_diameter
                            }

                            # 检查是否是新的障碍物 (避免重复检测)
                            if self._is_new_obstacle(obstacle):
                                self.obstacles_detected.append(obstacle)
                                print("检测到{}障碍物，位置: {}".format(color, obstacle_position))

        # 恢复头部位置
        self.motion.setAngles("HeadYaw", 0.0, 0.1)

        # 确认是否检测到全部障碍物
        if len(self.obstacles_detected) >= 3:
            self.tts.say("已检测到全部障碍物")
            return True
        else:
            self.tts.say("未检测到全部障碍物，请重试")
            return False

    def _estimate_distance(self, contour_area):
        """根据轮廓面积估计距离 (需要针对NAO摄像头进行校准)"""
        # 这是一个简化的估计模型，实际应用中需要进行校准
        # 假设障碍物是圆柱体，直径30cm
        # 基于透视投影原理，距离与图像中的面积成反比
        # 参数k需要通过实验校准获得
        k = 50000.0  # 校准参数
        distance = k / np.sqrt(contour_area)
        return min(max(distance, 0.3), 5.0)  # 限制估计距离在合理范围内

    def _image_to_world(self, image_x, image_y, distance, head_yaw):
        """将图像坐标转换为世界坐标"""
        # 获取摄像头在NAO坐标系中的位置
        camera_position = self.motion.getPosition("CameraTop", 2, True)

        # 计算目标在机器人坐标系中的方位
        image_center_x = 320  # 图像中心x坐标
        fov_h = 60.9 * math.pi / 180.0  # 水平视场角(弧度)

        # 计算水平角度偏移
        angle_offset = ((image_x - image_center_x) / 320.0) * (fov_h / 2.0)

        # 加上头部偏转角度
        total_angle = head_yaw + angle_offset

        # 计算世界坐标
        x = distance * math.cos(total_angle)
        y = distance * math.sin(total_angle)

        # 获取机器人当前位置和方向
        robot_pose = self.motion.getRobotPosition(True)
        robot_x, robot_y, robot_theta = robot_pose

        # 将障碍物坐标从机器人坐标系转换到世界坐标系
        world_x = robot_x + x * math.cos(robot_theta) - y * math.sin(robot_theta)
        world_y = robot_y + x * math.sin(robot_theta) + y * math.cos(robot_theta)

        return (world_x, world_y, 0)  # z坐标设为0 (障碍物在地面上)

    def _is_new_obstacle(self, obstacle):
        """检查是否是新的障碍物 (避免重复检测)"""
        if not self.obstacles_detected:
            return True

        for existing in self.obstacles_detected:
            # 计算欧氏距离
            p1 = existing["position"]
            p2 = obstacle["position"]
            distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            # 如果距离小于一定阈值，认为是同一个障碍物
            if distance < 0.5:  # 50cm阈值
                return False

        return True

    def plan_s_shaped_path(self):
        """规划S形避障路径"""
        if len(self.obstacles_detected) < 3:
            self.tts.say("障碍物数量不足，无法规划路径")
            return False

        # 对障碍物按照x坐标排序 (从起点到终点)
        sorted_obstacles = sorted(self.obstacles_detected, key=lambda o: o["position"][0])

        # 确定起点和终点
        start_point = (0.0, 0.0, 0.0)  # 起点 (机器人当前位置)
        end_point = (self.track_length, 0.0, 0.0)  # 终点

        # 清空路径点
        self.path_points = []
        self.path_points.append(start_point)

        # 为每个障碍物计算绕行点 (实现S型绕行)
        for i, obstacle in enumerate(sorted_obstacles):
            x, y, z = obstacle["position"]

            # 为实现S型路径，为相邻障碍物选择不同的绕行方向
            # 第一个障碍物从右侧绕行
            side_offset = self.obstacle_diameter * 0.7  # 与障碍物的安全距离

            if i % 2 == 0:  # 偶数索引障碍物，从右侧绕行
                y_offset = -side_offset
            else:  # 奇数索引障碍物，从左侧绕行
                y_offset = side_offset

            # 添加绕行点
            waypoint = (x, y + y_offset, 0.0)
            self.path_points.append(waypoint)

        # 添加终点
        self.path_points.append(end_point)

        print("已规划S型路径，路径点数量: {}".format(len(self.path_points)))
        return True

    def navigate_path(self):
        """沿规划路径导航"""
        if not self.path_points:
            self.tts.say("没有可导航的路径")
            return False

        self.tts.say("开始导航")
        self.start_time = time.time()
        self.is_running = True

        # 遍历所有路径点
        for i, target in enumerate(self.path_points[1:]):  # 跳过第一个点(起点)
            if not self.is_running:
                break

            print("导航到第{}个路径点: {}".format(i + 1, target))

            # 获取当前位置
            current_pose = self.motion.getRobotPosition(True)
            current_x, current_y, current_theta = current_pose

            # 计算目标方向
            target_x, target_y, _ = target
            dx = target_x - current_x
            dy = target_y - current_y
            target_theta = math.atan2(dy, dx)

            # 首先转向目标方向
            self._turn_to(target_theta)

            # 计算与目标的距离
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # 向目标点移动
            self._move_to_target(target)

            # 检查是否到达终点
            if i == len(self.path_points) - 2:  # 最后一个目标点
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.tts.say("到达终点，用时{:.1f}秒".format(elapsed_time))
                print("比赛完成，用时: {:.1f}秒".format(elapsed_time))

                # 停止行走并做一个胜利姿势
                self.motion.stopMove()
                self.behavior.runBehavior("animations/Stand/Gestures/Happy_1")

                self.is_running = False
                return True

        return self.is_running

    def _turn_to(self, target_theta):
        """转向指定方向"""
        # 获取当前方向
        current_theta = self.motion.getRobotPosition(True)[2]

        # 计算需要转动的角度
        delta_theta = target_theta - current_theta

        # 将角度规范化到[-pi, pi]范围内
        while delta_theta > math.pi:
            delta_theta -= 2.0 * math.pi
        while delta_theta < -math.pi:
            delta_theta += 2.0 * math.pi

        # 执行转向
        self.motion.moveTo(0, 0, delta_theta)

    def _move_to_target(self, target):
        """移动到目标点，同时避开障碍物和边界"""
        target_x, target_y, _ = target

        while self.is_running:
            # 获取当前位置
            current_pose = self.motion.getRobotPosition(True)
            current_x, current_y, current_theta = current_pose

            # 计算与目标的距离和方向
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # 如果已经足够接近目标，则停止
            if distance < 0.2:  # 20cm阈值
                self.motion.stopMove()
                return True

            # 计算前进方向
            target_theta = math.atan2(dy, dx)

            # 将目标方向转换为机器人坐标系
            delta_theta = target_theta - current_theta

            # 规范化角度
            while delta_theta > math.pi:
                delta_theta -= 2.0 * math.pi
            while delta_theta < -math.pi:
                delta_theta += 2.0 * math.pi

            # 检查是否需要重新转向
            if abs(delta_theta) > 0.3:  # 如果方向偏差超过0.3弧度(约17度)
                self._turn_to(target_theta)
                continue

            # 检查前方障碍物 - 使用声纳传感器辅助避障
            left_sonar = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            right_sonar = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

            # 如果探测到前方有障碍物，暂停并重新规划
            if min(left_sonar, right_sonar) < 0.4:  # 40cm阈值
                self.motion.stopMove()
                print("检测到前方障碍物，重新规划路径")
                # 这里可以添加局部路径重规划逻辑
                time.sleep(1.0)
                continue

            # 检查边界 - 通过视觉识别白色边界线
            # 这里需要添加边界检测逻辑
            if self._detect_boundary():
                self.motion.stopMove()
                print("检测到边界线，调整行走方向")
                # 远离边界
                self._avoid_boundary()
                continue

            # 计算前进步长 (根据距离调整速度)
            step_x = min(self.max_step_x, distance / 10.0)

            # 执行行走
            self.motion.move(step_x, 0, delta_theta / 10.0)
            time.sleep(0.1)  # 短暂暂停，给机器人时间执行动作

    def _detect_boundary(self):
        """检测跑道边界"""
        # 使用下摄像头检测白色边界线
        # 这里简化处理，实际应用中需要更复杂的图像处理算法

        # 临时切换到下摄像头
        current_camera = self.camera_id
        self.video.setParam(vision_definitions.kCameraSelectID, 1)  # 切换到下摄像头
        time.sleep(0.1)

        # 获取图像
        image = self.video.getImageRemote(self.camera_client)

        # 恢复原来的摄像头
        self.video.setParam(vision_definitions.kCameraSelectID, current_camera)

        if image is None:
            return False

        # 处理图像数据
        width = image[0]
        height = image[1]
        image_array = np.frombuffer(image[6], dtype=np.uint8).reshape(height, width, 3)

        # 创建白色掩码 (检测白色边界线)
        lower_white = np.array([200, 200, 200], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(image_array, lower_white, upper_white)

        # 检查图像底部是否有足够的白色像素 (边界线)
        bottom_region = white_mask[height - 50:height, :]
        white_pixel_ratio = np.count_nonzero(bottom_region) / float(bottom_region.size)

        return white_pixel_ratio > 0.2  # 如果超过20%的像素是白色，认为检测到边界

    def _avoid_boundary(self):
        """避开边界线"""
        # 获取当前位置
        current_pose = self.motion.getRobotPosition(True)
        _, _, current_theta = current_pose

        # 向场地中心方向移动
        # 这里使用简单的策略：转向90度，然后向前移动一小段距离
        self.motion.moveTo(0, 0, math.pi / 2)  # 转向90度
        self.motion.moveTo(0.2, 0, 0)  # 向前移动20cm

        # 恢复原来的方向
        self.motion.moveTo(0, 0, -math.pi / 2)

    def check_fallen(self):
        """检查机器人是否摔倒"""
        return not self.motion.robotIsWakeUp()

    def recover_from_fall(self):
        """从摔倒中恢复"""
        self.tts.say("尝试恢复")
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)

    def stop_run(self):
        """停止比赛"""
        self.is_running = False
        self.motion.stopMove()
        self.tts.say("比赛结束")

    def run_competition(self):
        """执行完整的比赛流程"""
        try:
            # 准备比赛
            self.prepare_for_run()

            # 等待开始信号

            # 检测障碍物


            # 规划S形路径


            # 开始导航
            self.tts.say("开始比赛")
            navigation_result = self.navigate_path()

            # 比赛结束后复位
            self.motion.stopMove()
            self.posture.goToPosture("StandInit", 0.5)

            return navigation_result

        except Exception as e:
            print("比赛执行过程中出错: {}".format(e))
            self.stop_run()
            return False
        finally:
            # 确保机器人停止
            self.motion.stopMove()


def main():
    """主函数"""
    print("NAO机器人障碍跑程序启动")

    # 创建NAO障碍跑对象
    try:
        # 使用实际的NAO机器人IP地址
        # 对于物理机器人，使用其IP地址
        # 对于模拟器，通常使用"127.0.0.1"
        nao_ip = "192.168.43.221"  # 请替换为您的NAO机器人IP地址
        nao_port = 9559

        obstacle_run = NAOObstacleRun(nao_ip, nao_port)

        # 执行比赛
        result = obstacle_run.run_competition()

        if result:
            print("比赛成功完成")
        else:
            print("比赛未能成功完成")

    except Exception as e:
        print("程序执行错误: {}".format(e))

    print("NAO机器人障碍跑程序结束")


if __name__ == "__main__":
    main()