# coding:utf-8
import math
import random
import time
import cv2
import numpy as np

from proxy_and_image import *
from recognized_cylinder_improved import *
from control_nao import change_the_postion

# 导入我们的增强边界检测器
# 注意：需要将前面的 EnhancedWhiteBoundaryDetector 类复制到单独文件中
# from enhanced_white_boundary import EnhancedWhiteBoundaryDetector

# 机器人控制参数
frameHeight = 0
frameWidth = 0
frameChannels = 0
frameArray = None
cameraPitchRange = 47.64 / 180 * math.pi
cameraYawRange = 60.97 / 180 * math.pi

row = "HeadPitch"
angle = 0.5235987755982988
maxstepx = 0.08
maxstepy = 0.12
maxsteptheta = 0.25
maxstepfrequency = 0.5
stepheight = 0.02
torsowx = 0.0
torsowy = 0.0

# 导航和避障参数
BASE_FORWARD_SPEED = 0.25
CENTERING_SPEED = 0.15
EMERGENCY_STOP_DISTANCE = 60  # 紧急停止距离（像素）
OBSTACLE_DETECTION_DISTANCE = 100  # 障碍物检测距离（像素）
MIN_OBSTACLE_SIZE = 40  # 最小障碍物大小
AVOIDANCE_LATERAL_DISTANCE = 0.25  # 避障横移距离
CENTERING_LATERAL_DISTANCE = 0.15  # 中心校正横移距离
SAFETY_PAUSE_TIME = 0.5  # 安全暂停时间


class IntelligentNavigationSystem:
    """
    智能导航系统 - 整合边界检测、障碍物检测和智能决策
    """

    def __init__(self):
        # 导入增强的边界检测器类（这里需要从前面的代码中导入）
        self.boundary_detector = None  # 在初始化时创建

        # 状态管理
        self.navigation_mode = "normal"  # normal, centering, avoiding, emergency_stop
        self.last_action_time = 0
        self.action_cooldown = 0.3

        # 障碍物跟踪
        self.tracked_obstacles = []
        self.obstacle_history = []
        self.history_size = 3

        # 性能统计
        self.total_corrections = 0
        self.successful_avoidances = 0

    def initialize_boundary_detector(self):
        """
        初始化边界检测器
        """

        # 这里需要导入前面创建的 EnhancedWhiteBoundaryDetector 类
        # 为了演示，我创建一个简化版本
        class SimpleBoundaryDetector:
            def __init__(self):
                self.calibrated = False
                self.field_center = None
                self.left_boundary = None
                self.right_boundary = None

            def calibrate_boundaries_enhanced(self, image):
                h, w = image.shape[:2]
                self.left_boundary = int(w * 0.2)
                self.right_boundary = int(w * 0.8)
                self.field_center = (self.left_boundary + self.right_boundary) // 2
                self.calibrated = True
                return True

            def calculate_center_deviation(self, robot_x):
                if not self.calibrated:
                    return 0, "unknown"
                deviation = robot_x - self.field_center
                return deviation, "left_of_center" if deviation < 0 else "right_of_center"

            def get_centering_correction(self, robot_x):
                deviation, _ = self.calculate_center_deviation(robot_x)
                max_dev = 50
                correction = np.clip(deviation / max_dev, -1, 1)
                urgency = "high" if abs(correction) > 0.6 else "medium" if abs(correction) > 0.3 else "low"
                return correction, urgency

            def check_boundary_safety(self, robot_x):
                if not self.calibrated:
                    return "unknown", 0
                left_dist = robot_x - self.left_boundary
                right_dist = self.right_boundary - robot_x
                min_dist = min(left_dist, right_dist)
                return "safe" if min_dist > 40 else "danger", min_dist

        self.boundary_detector = SimpleBoundaryDetector()

    def analyze_obstacles_for_avoidance(self, obstacles, image_width, robot_center_x):
        """
        分析障碍物情况，制定避障策略
        """
        if not obstacles:
            return "no_obstacles", None, None

        # 筛选前方重要障碍物
        critical_obstacles = []
        for obs in obstacles:
            obs_x, obs_y = obs['center']
            obs_width = obs.get('width', 0)

            # 检查障碍物是否在前方路径上
            if (obs_width >= MIN_OBSTACLE_SIZE and
                    obs_y > OBSTACLE_DETECTION_DISTANCE and
                    abs(obs_x - robot_center_x) < 100):  # 前方100像素范围内

                critical_obstacles.append(obs)

        if not critical_obstacles:
            return "path_clear", None, None

        # 找到最近的威胁障碍物
        closest_obstacle = max(critical_obstacles, key=lambda x: x['center'][1])

        # 检查是否需要紧急停止
        if closest_obstacle['center'][1] > (480 - EMERGENCY_STOP_DISTANCE):  # 假设图像高度480
            return "emergency_stop", closest_obstacle, None

        # 分析左右空间
        left_space = self._calculate_space_on_side(obstacles, robot_center_x, 'left', image_width)
        right_space = self._calculate_space_on_side(obstacles, robot_center_x, 'right', image_width)

        # 决定避障方向
        if left_space > right_space and left_space > 0.3:
            return "avoid_left", closest_obstacle, {"left_space": left_space, "right_space": right_space}
        elif right_space > left_space and right_space > 0.3:
            return "avoid_right", closest_obstacle, {"left_space": left_space, "right_space": right_space}
        else:
            return "no_safe_path", closest_obstacle, {"left_space": left_space, "right_space": right_space}

    def _calculate_space_on_side(self, obstacles, robot_x, side, image_width):
        """
        计算指定侧边的可用空间
        """
        if side == 'left':
            # 计算左侧空间：从左边界到机器人位置
            boundary = self.boundary_detector.left_boundary if self.boundary_detector.calibrated else 0
            available_width = robot_x - boundary

            # 检查左侧是否有障碍物阻挡
            for obs in obstacles:
                obs_x = obs['center'][0]
                if obs_x < robot_x and obs_x > boundary:
                    # 有障碍物在左侧，减少可用空间
                    available_width = min(available_width, robot_x - obs_x)
        else:
            # 计算右侧空间
            boundary = self.boundary_detector.right_boundary if self.boundary_detector.calibrated else image_width
            available_width = boundary - robot_x

            for obs in obstacles:
                obs_x = obs['center'][0]
                if obs_x > robot_x and obs_x < boundary:
                    available_width = min(available_width, obs_x - robot_x)

        # 归一化到0-1范围
        total_width = image_width
        return max(0, available_width / total_width)

    def make_navigation_decision(self, image, obstacles, robot_center_x):
        """
        综合决策函数 - 整合边界检测和障碍物避障
        """
        image_height, image_width = image.shape[:2]
        current_time = time.time()

        # 检查是否在冷却期
        if current_time - self.last_action_time < self.action_cooldown:
            return "wait", None

        # 1. 首先检查边界安全性
        boundary_status, boundary_distance = self.boundary_detector.check_boundary_safety(robot_center_x)

        if boundary_status == "danger":
            self.navigation_mode = "emergency_boundary_correction"
            # 紧急边界修正
            if robot_center_x < image_width // 2:
                return "emergency_right", {"reason": "boundary_violation", "distance": boundary_distance}
            else:
                return "emergency_left", {"reason": "boundary_violation", "distance": boundary_distance}

        # 2. 障碍物分析
        obstacle_status, critical_obstacle, space_info = self.analyze_obstacles_for_avoidance(
            obstacles, image_width, robot_center_x)

        if obstacle_status == "emergency_stop":
            self.navigation_mode = "emergency_stop"
            return "emergency_stop", {"obstacle": critical_obstacle}

        elif obstacle_status in ["avoid_left", "avoid_right"]:
            self.navigation_mode = "avoiding"
            direction = obstacle_status.split("_")[1]  # left 或 right

            # 验证避障方向的边界安全性
            future_x = robot_center_x + (-80 if direction == "left" else 80)
            future_safety, _ = self.boundary_detector.check_boundary_safety(future_x)

            if future_safety != "danger":
                self.successful_avoidances += 1
                return obstacle_status, {"obstacle": critical_obstacle, "space_info": space_info}
            else:
                # 避障方向不安全，采用保守策略
                return "slow_forward", {"reason": "unsafe_avoidance"}

        elif obstacle_status == "no_safe_path":
            self.navigation_mode = "emergency_stop"
            return "stop_and_wait", {"reason": "no_safe_path", "obstacle": critical_obstacle}

        # 3. 如果没有紧急情况，检查是否需要中心校正
        deviation, center_status = self.boundary_detector.calculate_center_deviation(robot_center_x)
        correction_strength, urgency = self.boundary_detector.get_centering_correction(robot_center_x)

        if urgency in ["high", "medium"] and abs(correction_strength) > 0.2:
            self.navigation_mode = "centering"
            self.total_corrections += 1

            if correction_strength < 0:  # 需要向左
                return "center_left", {"deviation": deviation, "strength": abs(correction_strength)}
            else:  # 需要向右
                return "center_right", {"deviation": deviation, "strength": abs(correction_strength)}

        # 4. 正常前进
        self.navigation_mode = "normal"
        return "forward", {"status": "normal_navigation"}

    def execute_action(self, action, params, motion_proxy):
        """
        执行导航动作
        """
        self.last_action_time = time.time()

        try:
            if action == "forward":
                # 正常前进
                motion_proxy.moveTo(BASE_FORWARD_SPEED, 0, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("正常前进")

            elif action == "center_left":
                # 向左校正到中心
                lateral_distance = min(CENTERING_LATERAL_DISTANCE, params['strength'] * 0.3)
                motion_proxy.moveTo(CENTERING_SPEED, lateral_distance, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("向左校正中心，偏差: {params['deviation']:.0f}px")

            elif action == "center_right":
                # 向右校正到中心
                lateral_distance = -min(CENTERING_LATERAL_DISTANCE, params['strength'] * 0.3)
                motion_proxy.moveTo(CENTERING_SPEED, lateral_distance, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("向右校正中心，偏差: {params['deviation']:.0f}px")

            elif action == "avoid_left":
                # 向左避障
                motion_proxy.moveTo(0, AVOIDANCE_LATERAL_DISTANCE, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("向左避障 - 左侧空间: {params['space_info']['left_space']:.2f}")
                time.sleep(0.8)  # 等待动作完成

            elif action == "avoid_right":
                # 向右避障
                motion_proxy.moveTo(0, -AVOIDANCE_LATERAL_DISTANCE, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("向右避障 - 右侧空间: {params['space_info']['right_space']:.2f}")
                time.sleep(0.8)

            elif action in ["emergency_left", "emergency_right"]:
                # 紧急边界修正
                direction = 1 if action == "emergency_left" else -1
                emergency_distance = direction * 0.2
                motion_proxy.moveTo(0, emergency_distance, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("紧急边界修正: {action}, 距离: {params['distance']:.1f}px")
                time.sleep(1.0)

            elif action == "emergency_stop":
                # 紧急停止
                motion_proxy.stopMove()
                print("紧急停止！前方障碍物过近")
                time.sleep(SAFETY_PAUSE_TIME)

            elif action == "stop_and_wait":
                # 停止等待
                motion_proxy.moveTo(0.05, 0, 0,  # 极慢前进
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("无安全路径，等待中 - 原因: {params['reason']}")

            elif action == "slow_forward":
                # 缓慢前进
                motion_proxy.moveTo(BASE_FORWARD_SPEED * 0.3, 0, 0,
                                    [["MaxStepX", maxstepx], ["MaxStepY", maxstepy],
                                     ["MaxStepTheta", maxsteptheta], ["MaxStepFrequency", maxstepfrequency],
                                     ["StepHeight", stepheight], ["TorsoWx", torsowx], ["TorsoWy", torsowy]])
                print("缓慢前进 - 原因: {params['reason']}")

            elif action == "wait":
                # 等待（冷却期）
                pass

        except Exception as e:
            print("执行动作 {action} 时出错: {e}")
            motion_proxy.stopMove()

    def visualize_navigation_state(self, image, obstacles, robot_center_x, action, params):
        """
        可视化导航状态
        """
        result = image.copy()

        # 使用边界检测器的可视化
        if self.boundary_detector and self.boundary_detector.calibrated:
            result = self.boundary_detector.visualize_enhanced(result, robot_center_x)

        # 标记障碍物
        for obs in obstacles:
            center = obs['center']
            color = (0, 0, 255) if obs.get('width', 0) >= MIN_OBSTACLE_SIZE else (128, 128, 128)
            cv2.circle(result, center, 8, color, -1)

            # 如果是关键障碍物，特殊标记
            if obs.get('width', 0) >= MIN_OBSTACLE_SIZE:
                cv2.putText(result, "OBS {obs.get('width', 0):.0f}",
                            (center[0] - 20, center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示当前状态
        status_color = {
            "forward": (0, 255, 0),
            "center_left": (0, 255, 255),
            "center_right": (0, 255, 255),
            "avoid_left": (255, 255, 0),
            "avoid_right": (255, 255, 0),
            "emergency_stop": (0, 0, 255),
            "emergency_left": (0, 0, 255),
            "emergency_right": (0, 0, 255),
            "stop_and_wait": (255, 0, 255),
            "slow_forward": (128, 128, 255),
            "wait": (128, 128, 128)
        }.get(action, (255, 255, 255))

        # 状态信息
        cv2.putText(result, "Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(result, "Mode: {self.navigation_mode}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # 统计信息
        cv2.putText(result, "Corrections: {self.total_corrections}", (10, result.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Avoidances: {self.successful_avoidances}", (10, result.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result


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


def main_navigation_loop(motion_proxy, posture_proxy):
    """
    主导航循环
    """
    # 初始化系统
    navigation_system = IntelligentNavigationSystem()
    navigation_system.initialize_boundary_detector()

    # 初始化视频
    video_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
    video_client = video_proxy.subscribeCamera(
        "intelligent_nav_" + str(random.random()),
        0, CONFIG["resolution"], CONFIG["colorSpace"], CONFIG["fps"]
    )

    frame_count = 0

    try:
        print("启动智能导航系统...")
        print("功能：白线中心导航 + 智能避障 + 安全保护")

        while True:
            # 获取图像
            frame = get_image_from_camera(0, video_proxy, video_client)
            if frame is None:
                continue

            image_height, image_width = frame.shape[:2]
            robot_center_x = image_width // 2

            # 前15帧校准边界
            if frame_count < 15:
                navigation_system.boundary_detector.calibrate_boundaries_enhanced(frame)
                frame_count += 1
                cv2.putText(frame, "Calibrating... {frame_count}/15",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Intelligent Navigation", frame)
                cv2.waitKey(100)
                continue

            # 检测障碍物
            processed_img, obstacles = detect_cylinder(frame, color_ranges)

            # 导航决策
            action, params = navigation_system.make_navigation_decision(
                frame, obstacles, robot_center_x)

            # 执行动作
            navigation_system.execute_action(action, params, motion_proxy)

            # 可视化
            result_img = navigation_system.visualize_navigation_state(
                processed_img, obstacles, robot_center_x, action, params)

            # 显示结果
            cv2.imshow("Intelligent Navigation", result_img)

            # 退出条件
            if cv2.waitKey(30) == 27:  # ESC
                break

            frame_count += 1

    except Exception as e:
        print("导航系统错误: {e}")
    finally:
        video_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()
        motion_proxy.stopMove()
        print("智能导航系统已停止")


# 主程序
if __name__ == "__main__":
    print("启动NAO智能导航系统...")
    print("特性：")
    print("1. 增强白线检测和中心导航")
    print("2. 智能障碍物避障")
    print("3. 多重安全保护机制")

    try:
        # 初始化机器人
        motionProxy, postureProxy = initialize_robot(CONFIG["ip"])
        motionProxy.setSmartStiffnessEnabled(1)

        # 调整头部角度
        change_the_postion(motionProxy, row, angle)

        # 启动导航系统
        main_navigation_loop(motionProxy, postureProxy)

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print("系统错误: {e}")
    finally:
        print("清理资源...")
        try:
            motionProxy.rest()
        except:
            pass