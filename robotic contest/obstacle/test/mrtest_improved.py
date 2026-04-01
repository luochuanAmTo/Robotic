# coding:utf-8
import math
import random
import time
import cv2
import numpy as np

from proxy_and_image import *
from recognized_cylinder_improved import *
from control_nao import change_the_postion
from WhiteBoundaryDetector import WhiteBoundaryDetector  # 引入白色边界检测器

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

# 场地和避障参数
FIELD_WIDTH = 2.0  # 场地宽度（米）- 根据实际情况调整
ROBOT_WIDTH = 0.3  # 机器人宽度
SAFETY_MARGIN = 0.2  # 安全边距
MIN_OBSTACLE_WIDTH = 60  # 触发避障的最小障碍宽度
OBSTACLE_DISTANCE_THRESHOLD = 100  # 障碍距离判断阈值（像素）
BASE_SPEED = 0.3  # 基础前进速度
AVOIDANCE_DISTANCE = 0.6  # 避障横移距离
MAX_LATERAL_DISTANCE = 1.0  # 最大允许横向移动距离（米）

# 避障时的增大步幅参数
AVOIDANCE_MAXSTEPY = 0.18  # 避障时增大的横向步幅


class SmartObstacleAvoidanceSystem:
    def __init__(self):
        self.processed_obstacles = set()  # 记录已处理的障碍
        self.last_avoidance_time = 0
        self.avoidance_cooldown = 3.0  # 避障冷却时间（秒）
        self.robot_position_x = 0  # 机器人在场地中的X位置（横向）
        self.boundary_detector = WhiteBoundaryDetector()  # 添加白色边界检测器

        # 新增：距离跟踪
        self.initial_position_set = False
        self.cumulative_lateral_distance = 0.0  # 累计横向移动距离
        self.last_robot_y_position = 0.0  # 上次机器人Y坐标位置

    def reset_lateral_distance_tracking(self):
        """重置横向距离跟踪"""
        self.cumulative_lateral_distance = 0.0
        self.initial_position_set = True
        print("重置横向移动距离跟踪")

    def update_lateral_distance(self, lateral_movement):
        """更新累计横向移动距离"""
        self.cumulative_lateral_distance += abs(lateral_movement)
        print("累计横向移动距离: {:.2f}m / {:.1f}m".format(
            self.cumulative_lateral_distance, MAX_LATERAL_DISTANCE))

    def can_move_laterally(self, planned_movement):
        """检查是否可以进行横向移动（不超过1米限制）"""
        future_distance = self.cumulative_lateral_distance + abs(planned_movement)
        if future_distance > MAX_LATERAL_DISTANCE:
            print("警告: 横向移动将超过限制 ({:.2f}m > {:.1f}m)".format(
                future_distance, MAX_LATERAL_DISTANCE))
            return False
        return True

    def estimate_robot_field_position(self, image_width):
        """
        估算机器人在场地中的横向位置
        这里使用简单的中心假设，实际应用中可以结合IMU数据
        """
        # 简化假设：机器人在场地中央开始
        # 实际应用中应该结合IMU或其他定位方法
        return FIELD_WIDTH / 2

    def calculate_available_space(self, obstacle, image_width, robot_field_pos):
        """
        计算障碍物左右两边的可用空间
        """
        # 将像素坐标转换为实际距离（简化计算）
        obstacle_center_pixel = obstacle['center'][0]
        obstacle_width_pixel = obstacle['width']

        # 计算障碍物在场地中的相对位置
        obstacle_field_ratio = obstacle_center_pixel / image_width

        # 估算左右空间
        left_space = obstacle_field_ratio * FIELD_WIDTH - robot_field_pos
        right_space = FIELD_WIDTH - obstacle_field_ratio * FIELD_WIDTH - robot_field_pos

        # 考虑障碍物宽度和安全边距
        required_width = ROBOT_WIDTH + SAFETY_MARGIN

        return {
            'left_space': max(0, left_space - required_width),
            'right_space': max(0, right_space - required_width),
            'obstacle_pixel_pos': obstacle_center_pixel
        }

    def select_primary_obstacle(self, obstacles):
        """
        从多个障碍中选择主要处理目标
        优先级：距离机器人最近的大障碍
        """
        if not obstacles:
            return None

        # 过滤出足够大的障碍
        significant_obstacles = [obs for obs in obstacles if obs['width'] >= MIN_OBSTACLE_WIDTH]

        if not significant_obstacles:
            return None

        # 选择最近的障碍（基于y坐标，越大越近）
        primary_obstacle = max(significant_obstacles,
                               key=lambda x: x['center'][1])

        return primary_obstacle

    def determine_avoidance_direction(self, obstacle, image_width):
        """
        确定最佳避障方向（考虑边界限制和距离限制）
        """
        robot_field_pos = self.estimate_robot_field_position(image_width)
        space_info = self.calculate_available_space(obstacle, image_width, robot_field_pos)

        left_space = space_info['left_space']
        right_space = space_info['right_space']

        print("空间分析 - 左侧: {:.2f}m, 右侧: {:.2f}m".format(left_space, right_space))

        # 使用边界检测器进行安全检查
        robot_center_x = image_width // 2  # 假设机器人在图像中央
        safe_direction, safety_reason = self.boundary_detector.get_safe_avoidance_direction(
            obstacle['center'][0], robot_center_x)

        if safe_direction is None:
            print("警告：无安全避障方向 - {}".format(safety_reason))
            return 'forward', 0  # 继续前进，寻找更好的机会

        # 检查距离限制
        if not self.can_move_laterally(AVOIDANCE_DISTANCE):
            print("警告：横向移动距离已达限制，继续前进")
            return 'forward', 0

        # 如果边界检测器建议特定方向，优先考虑
        if safety_reason == "right_unsafe" and left_space > ROBOT_WIDTH:
            return 'left', left_space
        elif safety_reason == "left_unsafe" and right_space > ROBOT_WIDTH:
            return 'right', right_space

        # 两边都安全时，选择空间更大的一边
        if left_space > right_space and left_space > ROBOT_WIDTH:
            return 'left', left_space
        elif right_space > left_space and right_space > ROBOT_WIDTH:
            return 'right', right_space
        elif left_space > ROBOT_WIDTH:
            return 'left', left_space
        elif right_space > ROBOT_WIDTH:
            return 'right', right_space
        else:
            # 两边都没有足够空间，选择相对较大的一边
            return ('left', left_space) if left_space >= right_space else ('right', right_space)

    def get_obstacle_id(self, obstacle):
        """
        为障碍物生成唯一ID（基于位置和颜色）
        """
        center_x, center_y = obstacle['center']
        return "{}_{}_{}".format(obstacle['color'], center_x // 50, center_y // 50)

    def should_avoid_obstacle(self, obstacle):
        """
        判断是否需要避开该障碍
        """
        obstacle_id = self.get_obstacle_id(obstacle)
        current_time = time.time()

        # 检查是否已处理过
        if obstacle_id in self.processed_obstacles:
            return False

        # 检查冷却时间
        if current_time - self.last_avoidance_time < self.avoidance_cooldown:
            return False

        # 检查障碍大小
        if obstacle['width'] < MIN_OBSTACLE_WIDTH:
            return False

        return True


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


def enhanced_vision_processing(vd_proxy, mt_proxy, posture_proxy):
    """
    增强的视觉处理函数 - 智能障碍躲避 + 白色边界检测 + 距离限制
    """
    # 初始化智能避障系统
    avoidance_system = SmartObstacleAvoidanceSystem()

    # 订阅摄像头
    video_client = vd_proxy.subscribeCamera(
        "vision_" + str(random.random()),
        0,  # 使用上部摄像头
        CONFIG["resolution"],
        CONFIG["colorSpace"],
        CONFIG["fps"],
    )

    frame_count = 0

    try:
        print("启.")

        while True:
            # 获取图像帧
            frame = get_image_from_camera(0, vd_proxy, video_client)
            if frame is None:
                print("获取图像失败")
                continue

            image_height, image_width = frame.shape[:2]

            # 前15帧用于校准边界
            if frame_count < 15:
                avoidance_system.boundary_detector.calibrate_boundaries(frame)
                frame_count += 1

                # 在校准期间显示进度
                cv2.putText(frame, "Calibrating boundaries... {}/15".format(frame_count),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Boundary Calibration", frame)
                cv2.waitKey(100)
                continue

            # 初始化距离跟踪（校准完成后）
            if not avoidance_system.initial_position_set:
                avoidance_system.reset_lateral_distance_tracking()

            # 执行柱体检测
            processed_img, all_obstacles = detect_cylinder(frame, color_ranges)

            # 检查边界状态
            robot_center_x = image_width // 2  # 假设机器人在图像中央
            boundary_status, distance_to_boundary = avoidance_system.boundary_detector.check_boundary_violation(
                robot_center_x)

            # 添加边界可视化
            processed_img = avoidance_system.boundary_detector.visualize_boundaries_and_warnings(
                processed_img, robot_center_x)

            # 在图像上显示检测信息和距离信息
            cv2.putText(processed_img, "Obstacles: {}".format(len(all_obstacles)),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_img, "Lateral: {:.2f}m/{:.1f}m".format(
                avoidance_system.cumulative_lateral_distance, MAX_LATERAL_DISTANCE),
                        (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 紧急边界处理
            if boundary_status in ["violation", "danger"]:
                print("边界紧急状态: {}, 距离: {:.1f}px".format(boundary_status, distance_to_boundary))

                # 检查是否还能进行紧急调整
                if avoidance_system.can_move_laterally(0.3):
                    # 紧急调整 - 向场地中央移动
                    if robot_center_x < image_width // 2:
                        # 机器人偏左，向右移动
                        emergency_direction = "RIGHT"
                        lateral_distance = -0.3
                    else:
                        # 机器人偏右，向左移动
                        emergency_direction = "LEFT"
                        lateral_distance = 0.3

                    cv2.putText(processed_img, "EMERGENCY CORRECTION - {}".format(emergency_direction),
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # 执行紧急横移（使用增大的步幅）
                    mt_proxy.moveTo(0, lateral_distance, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", AVOIDANCE_MAXSTEPY],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])

                    # 更新距离跟踪
                    avoidance_system.update_lateral_distance(abs(lateral_distance))

                    time.sleep(1.0)  # 等待移动完成
                    continue
                else:
                    # 无法移动，只能停止
                    print("警告：已达横向移动限制，无法进行紧急边界调整")
                    cv2.putText(processed_img, "LATERAL LIMIT REACHED - STOPPING",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 智能障碍处理
            primary_obstacle = avoidance_system.select_primary_obstacle(all_obstacles)

            if primary_obstacle:
                # 标记主要障碍
                center = primary_obstacle['center']
                cv2.circle(processed_img, center, 10, (0, 255, 0), -1)
                cv2.putText(processed_img, "TARGET: {}".format(primary_obstacle['color']),
                            (center[0] - 50, center[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 判断是否需要避障
                if avoidance_system.should_avoid_obstacle(primary_obstacle):
                    # 确定避障方向（考虑边界和距离限制）
                    direction_result = avoidance_system.determine_avoidance_direction(
                        primary_obstacle, image_width)

                    if len(direction_result) == 2:
                        direction, available_space = direction_result
                    else:
                        direction = direction_result
                        available_space = 0

                    print("检测到 {} 障碍，选择 {} 方向避障".format(primary_obstacle['color'], direction))
                    print("可用空间: {:.2f}m".format(available_space))

                    if direction == 'forward':
                        # 继续前进，等待更好的避障机会
                        cv2.putText(processed_img, "WAITING FOR SAFE PATH",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        mt_proxy.moveTo(BASE_SPEED * 0.5, 0, 0,
                                        [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                         ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                         ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                    else:
                        # 执行避障动作
                        if direction == 'left':
                            lateral_distance = AVOIDANCE_DISTANCE
                            cv2.putText(processed_img, "AVOIDING LEFT",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            lateral_distance = -AVOIDANCE_DISTANCE
                            cv2.putText(processed_img, "AVOIDING RIGHT",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # 横向移动（使用增大的步幅）
                        mt_proxy.moveTo(0, lateral_distance, 0,
                                        [["MaxStepX", maxstepx], ["MaxStepY", AVOIDANCE_MAXSTEPY],
                                         ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                         ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])

                        # 更新距离跟踪
                        avoidance_system.update_lateral_distance(abs(lateral_distance))

                        # 记录避障信息
                        obstacle_id = avoidance_system.get_obstacle_id(primary_obstacle)
                        avoidance_system.processed_obstacles.add(obstacle_id)
                        avoidance_system.last_avoidance_time = time.time()

                        # 短暂停顿后继续前进
                        time.sleep(0.1)

                        # 继续前进
                        mt_proxy.moveTo(BASE_SPEED * 1.5, 0, 0,
                                        [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                         ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                         ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                else:
                    # 正常前进
                    cv2.putText(processed_img, "MOVING FORWARD",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    mt_proxy.moveTo(BASE_SPEED, 0, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
            else:
                # 没有检测到障碍，检查边界警告状态
                if boundary_status == "warning":
                    cv2.putText(processed_img, "BOUNDARY WARNING - SLOW FORWARD",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    mt_proxy.moveTo(BASE_SPEED * 0.7, 0, 0,  # 减速前进
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                else:
                    cv2.putText(processed_img, "CLEAR PATH - MOVING FORWARD",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    mt_proxy.moveTo(BASE_SPEED, 0, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])

            # 显示处理结果
            cv2.imshow("Smart NAO Obstacle Avoidance with Distance Limit", processed_img)
            if cv2.waitKey(30) == 27:  # ESC退出
                break

            frame_count += 1

    except Exception as e:
        print("运行时错误: {}".format(str(e)))
    finally:
        # 释放资源
        vd_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()
        print("智能避障系统结束")


# 主程序
if __name__ == "__main__":
    print("启动NAO智能障碍躲避系统...")

    try:
        # 初始化机器人（保持原有逻辑）
        motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
        motionProxy.setSmartStiffnessEnabled(1)

        # 调整头部角度（保持原有逻辑）
        change_the_postion(motionProxy, row, angle)

        # 初始前进（保持原有逻辑）
        print("初始化移动...")
        motionProxy.moveTo(0.3, 0, 0,
                           [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                            ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                            ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])

        # 启动视觉处理（新的智能系统）
        vd_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
        enhanced_vision_processing(vd_proxy, motionProxy, postureProxy)

        print("任务完成!")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print("系统错误: {}".format(str(e)))
    finally:
        print("清理资源...")
        try:
            motionProxy.rest()
        except:
            pass