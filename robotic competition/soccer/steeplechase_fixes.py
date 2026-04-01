#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO机器人预编程路径障碍跑（优化版）
根据特定障碍物布局实现S形避障路径
不依赖传感器，通过精确运动控制完成比赛
"""

import sys
import time
import math
import math
from naoqi import ALProxy


class NAOPredefinedObstacleRun:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化NAO机器人及相关模块"""
        try:
            # 初始化基本代理模块
            self.motion = ALProxy("ALMotion", ip, port)
            self.posture = ALProxy("ALRobotPosture", ip, port)
            self.tts = ALProxy("ALTextToSpeech", ip, port)
            self.leds = ALProxy("ALLeds", ip, port)

            # 场地参数 (单位: 米)
            self.track_length = 6.0  # 跑道长度
            self.track_width = 2.0  # 跑道宽度
            self.obstacle_diameter = 0.3  # 障碍物直径

            # 初始化标志变量
            self.start_time = 0
            self.is_running = False

            # 行走参数 - 调整为更适合精确运动的参数
            self.max_step_x = 0.05  # 最大前进步长 (m) - 减小以提高精度
            self.max_step_y = 0.15  # 最大侧移步长 (m) - 减小以提高精度
            self.max_step_theta = 0.3  # 最大转向角度 (rad) - 减小以提高精度
            self.step_frequency = 0.6  # 步频 - 减小以提高稳定性

            # 根据图片设置实际障碍物位置 (x, y, 直径)
            # 以机器人初始位置为原点(0,0)
            # 根据图片，障碍物呈交错分布：红色在下方，蓝色在上方，黄色在下方
            self.obstacles = [
                [1.5, -0.5, self.obstacle_diameter],  # 红色障碍物：距起点1.5m，在下半部分
                [3.0, 0.5, self.obstacle_diameter],  # 蓝色障碍物：距起点3.0m，在上半部分
                [4.5, -0.5, self.obstacle_diameter]  # 黄色障碍物：距起点4.5m，在下半部分
            ]

            # 预定义S形路径点序列 (x, y, theta)
            self.path_points = []  # 将在initialize_path函数中生成

            # 设置语言
            self.tts.setLanguage("Chinese")

            print("NAO机器人预编程路径障碍跑初始化完成")

        except Exception as e:
            print("初始化失败: {}".format(e))
            sys.exit(1)

    def prepare_for_run(self):
        """准备竞赛前的设置"""
        self.motion.wakeUp()
        self.motion.setStiffnesses("Body", 1.0)
        self.posture.goToPosture("StandInit", 0.5)

        # 新版 NAOqi 使用 setMotionConfig 替代 moveConfig
        self.motion.setMotionConfig([["MAX_STEP_X", self.max_step_x],
                                     ["MAX_STEP_Y", self.max_step_y],
                                     ["MAX_STEP_THETA", self.max_step_theta],
                                     ["MAX_STEP_FREQUENCY", self.step_frequency],
                                     ["STEP_HEIGHT", 0.02]])

        # 其他设置保持不变
        self.motion.setExternalCollisionProtectionEnabled("All", True)
        self.leds.fadeRGB("FaceLeds", 0, 0, 1.0, 0.5)
    def initialize_path(self):
        """根据障碍物位置初始化S形路径点"""
        # 清空路径点
        self.path_points = []

        # 添加起点 (原点)
        self.path_points.append([0.0, 0.0, 0.0])  # [x, y, theta]

        # 添加第一个过渡点 - 帮助机器人开始向右移动以便绕过第一个障碍物
        self.path_points.append([0.5, -0.2, 0.0])

        # 为每个障碍物创建绕行路径点
        for i, obstacle in enumerate(self.obstacles):
            obs_x, obs_y, obs_diameter = obstacle

            # 安全绕行距离
            safe_distance = obs_diameter * 0.7  # 障碍物直径 * 安全系数

            # 对于每个障碍物，分别创建前/中/后三个路径点，使路径更平滑

            # 障碍物前方的路径点
            before_point = [obs_x - 0.3, obs_y - safe_distance if i == 1 else obs_y + safe_distance, 0.0]

            # 与障碍物平行的路径点
            parallel_point = [obs_x, obs_y - safe_distance if i == 1 else obs_y + safe_distance, 0.0]

            # 障碍物后方的路径点
            after_point = [obs_x + 0.3, obs_y - safe_distance if i == 1 else obs_y + safe_distance, 0.0]

            # 添加路径点
            self.path_points.append(before_point)
            self.path_points.append(parallel_point)
            self.path_points.append(after_point)

        # 添加最后的过渡点 - 帮助机器人返回中线朝向终点
        self.path_points.append([5.0, 0.0, 0.0])

        # 添加终点 (跑道尽头中心)
        self.path_points.append([self.track_length, 0.0, 0.0])

        print("已初始化S形路径，共{}个路径点".format(len(self.path_points)))
        self.visualize_path()  # 输出路径详情用于调试
        return True

    def visualize_path(self):
        """打印路径信息用于可视化和调试"""
        print("\n==== S形路径可视化 ====")
        print("以机器人初始位置为原点(0,0)")
        print("跑道尺寸: {}m x {}m".format(self.track_length, self.track_width))
        print("\n障碍物位置:")

        for i, obstacle in enumerate(self.obstacles):
            obs_x, obs_y, obs_diameter = obstacle
            color = ["红色", "蓝色", "黄色"][i]
            print("{}障碍物: 位置({:.2f}, {:.2f})，直径: {:.2f}m".format(
                color, obs_x, obs_y, obs_diameter
            ))

        print("\n路径点序列:")
        for i, point in enumerate(self.path_points):
            x, y, theta = point
            point_type = "起点" if i == 0 else "终点" if i == len(self.path_points) - 1 else "路径点"
            print("{} {}: ({:.2f}, {:.2f})".format(point_type, i, x, y))

        print("==== 可视化结束 ====\n")

    def navigate_path(self):
        """执行S形路径导航"""
        if not self.path_points:

            return False

        self.start_time = time.time()
        self.is_running = True

        # 依次导航到每个路径点
        for i, point in enumerate(self.path_points[1:]):  # 跳过第一个点(起点)
            if not self.is_running:
                break

            target_x, target_y, _ = point
            print("导航到第{}个路径点: ({:.2f}, {:.2f})".format(i + 1, target_x, target_y))

            # 导航到目标点
            success = self._navigate_to_point(target_x, target_y)

            if not success:
                print("导航到点 ({:.2f}, {:.2f}) 失败".format(target_x, target_y))
                continue

            # 短暂暂停，让机器人稳定
            time.sleep(0.1)

            # 检查是否到达终点
            if i == len(self.path_points) - 2:  # 最后一个目标点
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                print("比赛完成，用时: {:.1f}秒".format(elapsed_time))

                # 眼睛LED反馈 - 设置为绿色表示成功
                self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 1.0)



                self.is_running = False
                return True

        return self.is_running

    def _navigate_to_point(self, target_x, target_y):
        """导航到指定的目标点，使用更精细的分段导航策略"""
        try:
            # 获取当前位置
            current_pose = self.motion.getRobotPosition(True)
            current_x, current_y, current_theta = current_pose

            # 计算到目标点的距离和方向
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # 计算目标角度
            target_theta = math.atan2(dy, dx)

            # 计算需要旋转的角度
            delta_theta = target_theta - current_theta

            # 规范化角度到[-pi, pi]范围
            while delta_theta > math.pi:
                delta_theta -= 2.0 * math.pi
            while delta_theta < -math.pi:
                delta_theta += 2.0 * math.pi

            # 如果角度差较大，分多次小角度旋转以提高精度
            if abs(delta_theta) > 0.3:  # 约17度
                # 将大角度旋转分解为多个小角度旋转
                steps = int(abs(delta_theta) / 0.2) + 1
                step_theta = delta_theta / steps

                for s in range(steps):
                    print("旋转步骤 {}/{}: {:.2f} 弧度".format(s + 1, steps, step_theta))
                    self.motion.moveTo(0, 0, step_theta)
                    time.sleep(0.1)  # 短暂暂停以稳定
            elif abs(delta_theta) > 0.05:  # 角度差大于约3度才旋转
                print("旋转 {:.2f} 弧度".format(delta_theta))
                self.motion.moveTo(0, 0, delta_theta)
                time.sleep(0.1)  # 短暂暂停以稳定

            # 更新机器人位置和方向
            current_pose = self.motion.getRobotPosition(True)
            current_x, current_y, current_theta = current_pose

            # 重新计算距离（旋转后可能位置有变化）
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # 如果距离很小，认为已到达
            if distance < 0.05:  # 5cm以内认为已到达
                return True

            # 现在机器人已经面向目标点，开始移动
            # 如果距离较远，分段移动以提高精度

            # 确定最佳段数
            if distance > 0.5:
                # 分段移动策略 - 距离越远分越多段
                segments = int(distance / 0.3) + 1  # 每段最多30cm
                segment_distance = distance / segments

                print("距离 {:.2f}m 分为 {} 段移动，每段 {:.2f}m".format(
                    distance, segments, segment_distance))

                for s in range(segments):
                    # 执行一段移动
                    print("移动段 {}/{}: {:.2f}m".format(s + 1, segments, segment_distance))

                    # 精细控制移动 - 考虑到NAO行走的特点
                    # 对于小于10cm的移动使用更精细的控制
                    if segment_distance < 0.1:
                        # 对于极短距离，使用更精细的步态控制
                        self.motion.setMoveArmsEnabled(False, False)  # 禁用摆臂以提高稳定性
                        self.motion.moveTo(segment_distance, 0, 0)
                        self.motion.setMoveArmsEnabled(True, True)  # 恢复摆臂
                    else:
                        self.motion.moveTo(segment_distance, 0, 0)

                    # 短暂暂停以稳定
                    time.sleep(0.05)

                    # 检查进展 - 如果已经足够接近目标，可提前结束
                    updated_pose = self.motion.getRobotPosition(True)
                    updated_x, updated_y = updated_pose[0], updated_pose[1]
                    updated_distance = math.sqrt((target_x - updated_x) ** 2 + (target_y - updated_y) ** 2)

                    if updated_distance < 0.05:  # 5cm以内认为已到达
                        return True
            else:
                # 距离较短，一次移动完成
                print("直接移动 {:.2f}m".format(distance))
                self.motion.moveTo(distance, 0, 0)

            # 最终确认是否到达目标点附近
            final_pose = self.motion.getRobotPosition(True)
            final_x, final_y = final_pose[0], final_pose[1]
            final_distance = math.sqrt((target_x - final_x) ** 2 + (target_y - final_y) ** 2)

            print("最终距离误差: {:.2f}m".format(final_distance))
            return final_distance < 0.1  # 10cm误差内认为导航成功

        except Exception as e:
            print("导航过程中出错: {}".format(e))
            return False

    def run_competition(self):
        """执行完整的比赛流程"""
        try:
            # 准备比赛
            self.prepare_for_run()

            # 初始化S形路径
            self.initialize_path()

            # 等待开始信号

            time.sleep(0.5)  # 模拟等待开始信号

            # 开始导航

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
        finally:
            # 确保机器人停止
            self.motion.stopMove()


def main():
    """主函数"""
    print("NAO机器人预编程路径障碍跑程序启动")

    # 创建NAO障碍跑对象
    try:
        # 使用实际的NAO机器人IP地址
        nao_ip = "192.168.43.221"  # 请替换为您的NAO机器人IP地址
        nao_port = 9559

        obstacle_run = NAOPredefinedObstacleRun(nao_ip, nao_port)

        # 执行比赛
        result = obstacle_run.run_competition()

        if result:
            print("比赛成功完成")
        else:
            print("比赛未能成功完成")

    except Exception as e:
        print("程序执行错误: {}".format(e))

    print("NAO机器人预编程路径障碍跑程序结束")


if __name__ == "__main__":
    main()