# -*- coding: utf-8 -*-
"""
NAO机器人声呐实时监测程序
Python 2.7 版本
"""

import time
import sys
from naoqi import ALProxy


class NAOSonarMonitor:
    def __init__(self, robot_ip="192.168.1.100", robot_port=9559):
        """
        初始化NAO声呐监测器

        Args:
            robot_ip: NAO机器人的IP地址
            robot_port: NAO机器人的端口号（默认9559）
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port

        # 初始化代理
        try:
            self.memory = ALProxy("ALMemory", robot_ip, robot_port)
            self.sonar = ALProxy("ALSonar", robot_ip, robot_port)
            print("成功连接到NAO机器人: {0}:{1}".format(robot_ip, robot_port))
        except Exception as e:
            print("连接NAO机器人失败: {0}".format(e))
            sys.exit(1)

    def start_sonar_monitoring(self, update_interval=0.5):
        """
        开始声呐监测

        Args:
            update_interval: 更新间隔（秒）
        """
        print("开始声呐监测...")
        print("按 Ctrl+C 停止监测\n")

        # 订阅声呐服务
        self.sonar.subscribe("SonarMonitor")

        try:
            while True:
                # 获取声呐数据
                left_distance = self.get_sonar_distance("left")
                right_distance = self.get_sonar_distance("right")

                # 输出距离信息
                self.display_distance(left_distance, right_distance)

                # 检测障碍物
                self.check_obstacles(left_distance, right_distance)

                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n停止声呐监测...")
        finally:
            self.stop_sonar_monitoring()

    def get_sonar_distance(self, sensor="left"):
        """
        获取指定声呐传感器的距离

        Args:
            sensor: "left" 或 "right"

        Returns:
            距离值（米）
        """
        try:
            if sensor == "left":
                # 左声呐数据路径
                return self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            elif sensor == "right":
                # 右声呐数据路径
                return self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")
            else:
                print("错误: 传感器参数应为 'left' 或 'right'")
                return None
        except Exception as e:
            print("获取声呐数据失败: {0}".format(e))
            return None

    def display_distance(self, left_dist, right_dist):
        """
        显示声呐距离信息
        """
        print("-" * 50)
        print("时间: {0}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        print("左声呐距离: {0:.3f} 米".format(left_dist))
        print("右声呐距离: {0:.3f} 米".format(right_dist))

        # 计算平均距离
        if left_dist is not None and right_dist is not None:
            avg_distance = (left_dist + right_dist) / 2
            print("平均距离: {0:.3f} 米".format(avg_distance))

    def check_obstacles(self, left_dist, right_dist, warning_threshold=0.5, danger_threshold=0.2):
        """
        检查障碍物并发出警告

        Args:
            left_dist: 左声呐距离
            right_dist: 右声呐距离
            warning_threshold: 警告阈值（米）
            danger_threshold: 危险阈值（米）
        """
        if left_dist is None or right_dist is None:
            return

        # 检查最小距离
        min_distance = min(left_dist, right_dist)

        if min_distance < danger_threshold:
            print(" 检测到近距离障碍物! ({0:.3f}米)".format(min_distance))
        elif min_distance < warning_threshold:
            print("附近有障碍物 ({0:.3f}米)".format(min_distance))

    def get_sonar_details(self):
        """
        获取声呐详细信息
        """
        try:
            # 获取声呐配置信息
            sonar_info = self.memory.getDataListName("Device/SubDeviceList/US")
            print("声呐设备信息:")
            for info in sonar_info:
                if "Sensor" in info or "Value" in info:
                    value = self.memory.getData(info)
                    print("  {0}: {1}".format(info.split('/')[-1], value))
        except Exception as e:
            print("获取声呐详细信息失败: {0}".format(e))

    def stop_sonar_monitoring(self):
        """
        停止声呐监测
        """
        try:
            self.sonar.unsubscribe("SonarMonitor")
            print("已取消订阅声呐服务")
        except Exception as e:
            print("停止声呐监测时出错: {0}".format(e))

    def continuous_distance_reading(self, duration=10, interval=0.5):
        """
        连续读取距离数据

        Args:
            duration: 持续时间（秒）
            interval: 读取间隔（秒）
        """
        print("开始连续距离读取，持续时间: {0}秒".format(duration))

        start_time = time.time()
        measurements = []

        try:
            self.sonar.subscribe("ContinuousReading")

            while time.time() - start_time < duration:
                left_dist = self.get_sonar_distance("left")
                right_dist = self.get_sonar_distance("right")

                if left_dist is not None and right_dist is not None:
                    measurement = {
                        'timestamp': time.time(),
                        'left': left_dist,
                        'right': right_dist,
                        'average': (left_dist + right_dist) / 2
                    }
                    measurements.append(measurement)

                    # 显示当前读数
                    print("[{0:.1f}s] 左: {1:.3f}m, 右: {2:.3f}m, 平均: {3:.3f}m".format(
                        time.time() - start_time,
                        left_dist,
                        right_dist,
                        measurement['average']
                    ))

                time.sleep(interval)

            # 显示统计信息
            self.display_statistics(measurements)

        except Exception as e:
            print("连续读取时出错: {0}".format(e))
        finally:
            self.sonar.unsubscribe("ContinuousReading")

    def display_statistics(self, measurements):
        """
        显示测量统计数据
        """
        if not measurements:
            print("没有测量数据")
            return

        # 提取距离数据
        left_distances = [m['left'] for m in measurements]
        right_distances = [m['right'] for m in measurements]
        avg_distances = [m['average'] for m in measurements]

        print("\n" + "=" * 50)
        print("测量统计:")
        print("总测量次数: {0}".format(len(measurements)))
        print("左声呐 - 最小值: {0:.3f}m, 最大值: {1:.3f}m, 平均值: {2:.3f}m".format(
            min(left_distances), max(left_distances), sum(left_distances) / len(left_distances)
        ))
        print("右声呐 - 最小值: {0:.3f}m, 最大值: {1:.3f}m, 平均值: {2:.3f}m".format(
            min(right_distances), max(right_distances), sum(right_distances) / len(right_distances)
        ))
        print("平均距离 - 最小值: {0:.3f}m, 最大值: {1:.3f}m, 平均值: {2:.3f}m".format(
            min(avg_distances), max(avg_distances), sum(avg_distances) / len(avg_distances)
        ))


def main():
    """
    主函数
    """
    print("=" * 60)
    print("NAO机器人声呐监测系统")
    print("=" * 60)

    # 设置NAO机器人的IP地址
    # 请根据实际网络设置修改IP地址
    robot_ip = "192.168.43.247"  # 替换为您的NAO机器人的实际IP

    # 创建声呐监测器
    monitor = NAOSonarMonitor(robot_ip=robot_ip)

    # 显示声呐详细信息
    monitor.get_sonar_details()

    # 选择监测模式
    print("\n请选择监测模式:")
    print("1. 实时持续监测")
    print("2. 定时采集数据")

    choice = raw_input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        # 实时持续监测
        interval = float(raw_input("请输入更新间隔(秒，默认0.5): ") or "0.5")
        monitor.start_sonar_monitoring(update_interval=interval)
    elif choice == "2":
        # 定时采集
        duration = float(raw_input("请输入采集时长(秒，默认10): ") or "10")
        interval = float(raw_input("请输入采集间隔(秒，默认0.5): ") or "0.5")
        monitor.continuous_distance_reading(duration=duration, interval=interval)
    else:
        print("无效选择，使用默认模式")
        monitor.start_sonar_monitoring()


if __name__ == "__main__":
    main()