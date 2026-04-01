#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO机器人超声波障碍跑代码
基于NAOqi SDK实现超声波传感的障碍物检测与S型路径导航
"""

import sys
import time
import math

import cv2
import numpy as np
import math
from naoqi import ALProxy


class NAOUltrasonicObstacleRun:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化NAO机器人及相关模块"""
        try:
            # 初始化基本代理模块
            self.motion = ALProxy("ALMotion", ip, port)
            self.posture = ALProxy("ALRobotPosture", ip, port)
            self.memory = ALProxy("ALMemory", ip, port)
            self.tts = ALProxy("ALTextToSpeech", ip, port)
            self.sonar = ALProxy("ALSonar", ip, port)
            self.leds = ALProxy("ALLeds", ip, port)
            self.video = ALProxy("ALVideoDevice", ip, port)  # 用于辅助检测边界线

            # 设置语言（可选）
            self.tts.setLanguage("Chinese")

            # 初始化标志变量
            self.is_running = False
            self.start_time = 0
            self.obstacles_map = []  # 存储检测到的障碍物地图
            self.path_points = []  # 存储规划的路径点

            # 场地参数 (单位: 米)
            self.track_length = 6.0
            self.track_width = 2.0
            self.obstacle_diameter = 0.3

            # 行走参数
            self.max_step_x = 0.04  # 最大前进步长 (m)
            self.max_step_y = 0.14  # 最大侧移步长 (m)
            self.max_step_theta = 0.3  # 最大转向角度 (rad)
            self.step_frequency = 0.6  # 步频

            # 激活超声波传感器
            self.sonar.subscribe("ObstacleRunner")

            # 超声波传感器参数
            self.sonar_threshold = 0.5  # 超声波检测阈值 (m)
            self.sonar_scan_resolution = 15  # 扫描分辨率 (度)

            # 用于边界检测的相机参数
            self.camera_name = "BoundaryCamera"
            self.camera_client = self.video.subscribe(
                self.camera_name,
                2,  # kQVGA: 320x240
                11,  # kRGBColorSpace
                15  # fps
            )

            self.tts.say("初始化完成")
            print("NAO机器人超声波障碍跑初始化完成")

        except Exception as e:
            print("初始化失败: {}".format(e))
            sys.exit(1)

    def __del__(self):
        """清理资源"""
        try:
            # 取消订阅超声波传感器和相机
            self.sonar.unsubscribe("ObstacleRunner")
            self.video.unsubscribe(self.camera_client)
            # 停止行走
            self.motion.stopMove()
        except:
            pass

    def prepare_for_run(self):
        """准备竞赛前的设置"""
        try:
            # 唤醒机器人并设置刚度
            self.motion.wakeUp()
            self.motion.setStiffnesses("Body", 1.0)

            # 调整到初始姿势
            self.posture.goToPosture("StandInit", 0.5)

            # 配置行走参数 - 使用兼容的API
            # 新版NAOqi可能不支持moveConfig，改用以下方式单独设置参数
            self.motion.setMotionConfig([["MAX_STEP_X", self.max_step_x],
                                         ["MAX_STEP_Y", self.max_step_y],
                                         ["MAX_STEP_THETA", self.max_step_theta],
                                         ["MAX_STEP_FREQUENCY", self.step_frequency],
                                         ["STEP_HEIGHT", 0.02],
                                         ["TORSO_WX", 0.0],
                                         ["TORSO_WY", 0.0]])



            # 设置避障能力
            self.motion.setExternalCollisionProtectionEnabled("All", True)

            # 眼睛LED反馈 - 设置为蓝色表示准备就绪
            self.leds.fadeRGB("FaceLeds", 0, 0, 1.0, 0.5)


            return True

        except Exception as e:
            print("准备比赛时出错:", e)
            return False

    def scan_environment(self):
        """使用超声波传感器扫描环境以检测障碍物"""


        # 清空障碍物地图
        self.obstacles_map = []

        # 初始头部位置 - 略微低头
        self.motion.setAngles("HeadPitch", 0.2, 0.2)

        # 扫描角度范围 (弧度)
        scan_range = [-math.pi / 6, math.pi / 6]  # 从左90度到右90度

        # 转换扫描分辨率从度到弧度
        resolution_rad = math.radians(self.sonar_scan_resolution)

        # 计算扫描点数量
        scan_points = int((scan_range[1] - scan_range[0]) / resolution_rad) + 1

        # 对每个扫描点进行检测
        for i in range(scan_points):
            # 计算当前头部角度
            head_yaw = scan_range[0] + i * resolution_rad

            # 设置头部角度
            self.motion.setAngles("HeadYaw", head_yaw, 0.2)
            time.sleep(0.5)  # 等待头部移动到位并稳定超声波读数

            # 获取超声波传感器值
            left_sonar = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            right_sonar = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

            # 使用左右超声波的平均值作为当前方向的距离测量
            # 这是一个简化处理，实际上更准确的方法是根据传感器位置和角度计算
            sonar_distance = min(left_sonar, right_sonar)

            # 确定是否检测到障碍物 (距离小于阈值)
            if sonar_distance < self.sonar_threshold:
                # 获取机器人当前位置
                robot_pose = self.motion.getRobotPosition(True)
                robot_x, robot_y, robot_theta = robot_pose

                # 计算障碍物的世界坐标 (简化计算)
                # 需要考虑机器人的位置和头部角度
                obstacle_distance = sonar_distance + 0.1  # 加上从传感器到机器人中心的距离修正
                obstacle_angle = robot_theta + head_yaw

                obstacle_x = robot_x + obstacle_distance * math.cos(obstacle_angle)
                obstacle_y = robot_y + obstacle_distance * math.sin(obstacle_angle)

                # 检查是否是新障碍物 (避免重复检测)
                is_new = True
                for obs in self.obstacles_map:
                    obs_x, obs_y = obs[:2]
                    dist = math.sqrt((obstacle_x - obs_x) ** 2 + (obstacle_y - obs_y) ** 2)
                    if dist < self.obstacle_diameter:
                        is_new = False
                        break

                # 如果是新障碍物，则添加到地图中
                if is_new:
                    # 存储障碍物位置和大小 [x, y, diameter]
                    self.obstacles_map.append([obstacle_x, obstacle_y, self.obstacle_diameter])

                    # 使用眼睛LED反馈障碍物检测
                    self.leds.fadeRGB("FaceLeds", 1.0, 0.5, 0.0, 0.3)  # 橙色
                    time.sleep(0.3)
                    self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)  # 恢复蓝色

                    print("检测到障碍物，位置: ({:.2f}, {:.2f})".format(obstacle_x, obstacle_y))

        # 恢复头部位置
        self.motion.setAngles("HeadYaw", 0.0, 0.2)

        # 检查是否检测到至少3个障碍物
        if len(self.obstacles_map) >= 1:
            self.tts.say("检测到{}个障碍物".format(len(self.obstacles_map)))
            return True
        else:
            # 如果检测到的障碍物不足3个，进行第二次扫描以确认
            time.sleep(1.0)
            self.tts.say("二次扫描")

            # 向前移动一小段距离以获得不同角度的扫描
            self.motion.moveTo(0.3, 0, 0)

            # 进行第二次扫描
            return self.scan_environment()

    def verify_obstacles(self):
        """通过接近检查验证障碍物位置"""
        if len(self.obstacles_map) < 1:
            return False

        self.tts.say("开始验证障碍物位置")

        # 选择最近的三个障碍物
        obstacles = sorted(self.obstacles_map, key=lambda x: x[0])[:3]
        verified_obstacles = []

        for i, obstacle in enumerate(obstacles):
            obstacle_x, obstacle_y, _ = obstacle

            # 计算从当前位置到障碍物附近的路径点
            robot_pose = self.motion.getRobotPosition(True)
            robot_x, robot_y, robot_theta = robot_pose

            # 计算接近障碍物的安全距离点
            approach_distance = self.obstacle_diameter + 0.3  # 障碍物直径 + 安全距离

            # 根据障碍物和机器人位置计算接近角度
            dx = obstacle_x - robot_x
            dy = obstacle_y - robot_y
            approach_angle = math.atan2(dy, dx)

            # 计算接近点坐标
            approach_x = obstacle_x - approach_distance * math.cos(approach_angle)
            approach_y = obstacle_y - approach_distance * math.sin(approach_angle)

            # 移动到接近点以验证障碍物
            self._move_to_position(approach_x, approach_y)

            # 到达接近点后，使用超声波再次检测
            left_sonar = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            right_sonar = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

            # 如果超声波检测到障碍物，则确认障碍物存在
            if min(left_sonar, right_sonar) < approach_distance:
                # 超声波检测确认了障碍物

                # 精确定位障碍物 - 旋转头部进行扫描以确定最精确的位置
                refined_position = self._refine_obstacle_position()
                if refined_position:
                    verified_obstacles.append([refined_position[0], refined_position[1], self.obstacle_diameter])
                else:
                    # 如果无法精确定位，使用原始估计位置
                    verified_obstacles.append(obstacle)

                # 使用LED反馈确认障碍物
                self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.5)  # 绿色
                time.sleep(0.5)
                self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)  # 恢复蓝色
            else:
                # 未能通过超声波确认障碍物，可能是误检
                self.leds.fadeRGB("FaceLeds", 1.0, 0.0, 0.0, 0.5)  # 红色
                time.sleep(0.5)
                self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)  # 恢复蓝色

        # 更新障碍物地图
        self.obstacles_map = verified_obstacles

        # 返回到起始位置
        self._move_to_position(0, 0)

        return len(verified_obstacles) >= 3

    def _refine_obstacle_position(self):
        """通过旋转头部和超声波传感器精确定位障碍物"""
        # 初始头部角度
        scan_range = [-math.pi / 4, math.pi / 4]  # 从左45度到右45度
        resolution_rad = math.radians(5)  # 5度分辨率

        scan_points = int((scan_range[1] - scan_range[0]) / resolution_rad) + 1
        min_distance = float('inf')
        min_angle = 0

        # 进行精细扫描以找到最小距离
        for i in range(scan_points):
            head_yaw = scan_range[0] + i * resolution_rad
            self.motion.setAngles("HeadYaw", head_yaw, 0.1)
            time.sleep(0.2)

            left_sonar = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            right_sonar = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

            avg_distance = (left_sonar + right_sonar) / 2.0

            if avg_distance < min_distance:
                min_distance = avg_distance
                min_angle = head_yaw

        # 恢复头部位置
        self.motion.setAngles("HeadYaw", 0.0, 0.1)

        # 如果找到了障碍物 (距离在有效范围内)
        if min_distance < self.sonar_threshold:
            # 获取机器人当前位置
            robot_pose = self.motion.getRobotPosition(True)
            robot_x, robot_y, robot_theta = robot_pose

            # 计算障碍物的精确位置
            obstacle_angle = robot_theta + min_angle
            obstacle_x = robot_x + min_distance * math.cos(obstacle_angle)
            obstacle_y = robot_y + min_distance * math.sin(obstacle_angle)

            return (obstacle_x, obstacle_y)

        return None

    def plan_s_shaped_path(self):
        """规划S形避障路径"""
        if len(self.obstacles_map) < 3:

            return False

        # 清空路径点
        self.path_points = []

        # 按x坐标排序障碍物 (沿跑道方向)
        obstacles_sorted = sorted(self.obstacles_map, key=lambda obs: obs[0])

        # 定义起点和终点
        start_point = [0.0, 0.0, 0.0]  # [x, y, theta]
        end_point = [self.track_length, 0.0, 0.0]

        # 添加起点
        self.path_points.append(start_point)

        # 为S形路径交替设置障碍物绕行方向
        for i, obstacle in enumerate(obstacles_sorted):
            obs_x, obs_y, obs_diameter = obstacle

            # 安全绕行距离
            safe_distance = obs_diameter * 0.7

            # 设置绕行方向 (S形路径)
            if i % 2 == 0:  # 偶数索引障碍物，从右侧绕行
                passing_y = -safe_distance
            else:  # 奇数索引障碍物，从左侧绕行
                passing_y = safe_distance

            # 创建绕行点
            # 添加接近点
            approach_point = [obs_x - 0.3, obs_y + passing_y, 0.0]
            self.path_points.append(approach_point)

            # 添加绕行点
            path_point = [obs_x, obs_y + passing_y, 0.0]
            self.path_points.append(path_point)

            # 添加离开点
            leave_point = [obs_x + 0.3, obs_y + passing_y, 0.0]
            self.path_points.append(leave_point)

        # 添加终点
        self.path_points.append(end_point)


        print("S形路径规划完成，共{}个路径点".format(len(self.path_points)))
        return True

    def navigate_path(self):
        """沿规划路径导航"""
        if not self.path_points:
            self.tts.say("没有规划路径，无法导航")
            return False


        self.start_time = time.time()
        self.is_running = True

        # 依次导航到每个路径点
        for i, point in enumerate(self.path_points[1:]):  # 跳过第一个点 (起点)
            if not self.is_running:
                break

            target_x, target_y, target_theta = point

            print("导航到第{}个路径点: ({:.2f}, {:.2f})".format(i + 1, target_x, target_y))

            # 移动到目标点，同时进行实时避障
            success = self._move_to_target(target_x, target_y)

            if not success:

                # 可以添加重新规划的逻辑
                return False

            # 检查是否到达终点
            if i == len(self.path_points) - 2:  # 最后一个目标点
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.tts.say("".format(elapsed_time))
                print("比赛完成，用时: {:.1f}秒".format(elapsed_time))

                # 眼睛LED反馈 - 设置为绿色表示成功
                self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 1.0)

                # 停止行走并做一个胜利姿势
                self.motion.stopMove()
                try:
                    self.motion.angleInterpolation(
                        ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"],
                        [[0.0], [0.0], [0.0], [0.0]],
                        [[1.0], [1.0], [1.0], [1.0]],
                        True
                    )
                except:
                    pass  # 如果姿势执行失败，继续执行

                self.is_running = False
                return True

        return self.is_running

    def _move_to_position(self, x, y):
        """移动到指定位置"""
        # 获取当前位置
        current_pose = self.motion.getRobotPosition(True)
        current_x, current_y, current_theta = current_pose

        # 计算目标相对于当前位置的方向和距离
        dx = x - current_x
        dy = y - current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        target_theta = math.atan2(dy, dx)

        # 先旋转到正确的方向
        delta_theta = target_theta - current_theta
        while delta_theta > math.pi:
            delta_theta -= 2.0 * math.pi
        while delta_theta < -math.pi:
            delta_theta += 2.0 * math.pi

        self.motion.moveTo(0, 0, delta_theta)

        # 然后直线移动到目标点
        self.motion.moveTo(distance, 0, 0)

        return True

    def _move_to_target(self, target_x, target_y):
        """移动到目标点，同时进行实时避障"""
        success = False
        retry_count = 0
        max_retries = 3

        while not success and retry_count < max_retries:
            try:
                # 获取当前位置
                current_pose = self.motion.getRobotPosition(True)
                current_x, current_y, current_theta = current_pose

                # 计算目标相对于当前位置的方向和距离
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # 如果已经足够接近目标，则认为到达
                if distance < 0.1:  # 10cm阈值
                    return True

                # 计算目标方向
                target_theta = math.atan2(dy, dx)

                # 计算需要旋转的角度
                delta_theta = target_theta - current_theta
                while delta_theta > math.pi:
                    delta_theta -= 2.0 * math.pi
                while delta_theta < -math.pi:
                    delta_theta += 2.0 * math.pi

                # 分段导航 - 先转向，然后移动
                if abs(delta_theta) > 0.2:  # 如果方向偏差超过约11度
                    # 先旋转到正确的方向
                    self.motion.moveTo(0, 0, delta_theta)
                else:
                    # 方向基本正确，设置移动距离
                    # 如果距离比较远，分段移动以便更好地检测障碍物
                    move_distance = min(distance, 0.3)  # 最多移动30cm

                    # 执行移动
                    self.motion.moveTo(move_distance, 0, 0)

                    # 检查是否检测到新的障碍物或边界
                    if self._check_obstacles_and_boundaries():
                        # 碰到障碍物或边界，尝试重新规划路径
                        print("检测到障碍物或边界，尝试局部路径规划")
                        self._local_path_planning(target_x, target_y)
                        retry_count += 1
                        continue

                # 检查移动后的位置，判断是否接近目标
                updated_pose = self.motion.getRobotPosition(True)
                updated_x, updated_y = updated_pose[0], updated_pose[1]

                updated_distance = math.sqrt((target_x - updated_x) ** 2 + (target_y - updated_y) ** 2)

                # 如果距离减小，则继续尝试
                if updated_distance < distance:
                    success = (updated_distance < 0.1)  # 如果距离小于10cm，则认为到达
                else:
                    # 距离没有减小，可能有障碍，尝试重新规划
                    retry_count += 1

            except Exception as e:
                print("移动过程中出错: {}".format(e))
                retry_count += 1

        return success

    def _check_obstacles_and_boundaries(self):
        """增强版障碍物与边界检测"""
        # 超声波检测
        left = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        right = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

        # 多方位检测
        if min(left, right) < 0.35:  # 前方35cm检测
            return True

        # 侧向检测（防止卡在障碍物之间）
        if max(left, right) < 0.5:  # 两侧50cm检测
            return True

        # 边界检测（保持原有逻辑）
        try:
            self.video.setParam(0, 1)
            image = self.video.getImageRemote(self.camera_client)
            self.video.setParam(0, 0)

            if image:
                img = np.frombuffer(image[6], np.uint8).reshape(image[1], image[0], 3)
                edge_mask = cv2.Canny(img, 50, 150)
                edge_density = np.sum(edge_mask > 0) / edge_mask.size
                if edge_density > 0.25:
                    return True
        except Exception as e:
            print("边界检测异常:", e)

        return False

    def _recalculate_path(self, current_pose, original_target):
        """动态路径重规划"""
        print("执行动态路径重规划")

        # 获取当前位置
        current_x, current_y, _ = current_pose
        target_x, target_y = original_target

        # 计算新的中间点
        mid_point = [
            (current_x + target_x) / 2,
            (current_y + target_y) / 2 + 0.3 * (-1 if current_y < target_y else 1),
            0
        ]

        # 插入新的路径点
        self.path_points.insert(0, mid_point)  # 添加到待处理队列最前面
        print("插入临时路径点 ({mid_point[0]:.2f}, {mid_point[1]:.2f})")
    def _local_path_planning(self, target_x, target_y):
        """在检测到障碍物时尝试局部路径规划"""
        # 获取当前位置
        current_pose = self.motion.getRobotPosition(True)
        current_x, current_y, current_theta = current_pose

        # 获取超声波传感器数据
        left_sonar = self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        right_sonar = self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")

        # 尝试向障碍物较少的一侧移动
        if left_sonar > right_sonar:
            # 左侧空间更大，向左移动
            self.motion.moveTo(0, 0.2, 0)
        else:
            # 右侧空间更大，向右移动
            self.motion.moveTo(0, -0.2, 0)

        # 向前移动一小段距离
        self.motion.moveTo(0.2, 0, 0)

    def run_competition(self):
        """执行完整的比赛流程"""
        try:
            # 准备比赛
            self.prepare_for_run()

            # 等待开始信号

            time.sleep(3)  # 模拟等待开始信号

            # 扫描环境
            if not self.scan_environment():

                return False

            # 验证障碍物位置
            if not self.verify_obstacles():

                return False

            # 规划S形路径
            if not self.plan_s_shaped_path():

                return False

            # 导航路径
            result = self.navigate_path()

            # 比赛结束后返回初始姿势
            self.motion.stopMove()
            self.posture.goToPosture("StandInit", 0.5)

            return result

        except Exception as e:
            print("比赛执行过程中出错: {}".format(e))
            self.is_running = False
            self.motion.stopMove()
            return False


def main():
    """主函数"""
    print("NAO机器人超声波障碍跑程序启动")

    # 创建NAO障碍跑对象
    try:
        # 使用实际的NAO机器人IP地址
        nao_ip = "192.168.43.221"  # 请替换为您的NAO机器人IP地址
        nao_port = 9559

        obstacle_run = NAOUltrasonicObstacleRun(nao_ip, nao_port)

        # 执行比赛
        result = obstacle_run.run_competition()

        if result:
            print("比赛成功完成")
        else:
            print("比赛未能成功完成")

    except Exception as e:
        print("程序执行错误: {}".format(e))

    print("NAO机器人超声波障碍跑程序结束")


if __name__ == "__main__":
    main()