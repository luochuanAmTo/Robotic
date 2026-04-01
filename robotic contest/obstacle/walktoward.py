# -*- coding: utf-8 -*-
from naoqi import ALProxy, ALModule
import time
import math


class BHumanWalker(ALModule):
    def __init__(self, name, robot_ip, port=9559):
        ALModule.__init__(self, name)
        self.ip = robot_ip
        self.port = port
        self.motion = ALProxy("ALMotion", self.ip, self.port)
        self.posture = ALProxy("ALRobotPosture", self.ip, self.port)
        self.memory = ALProxy("ALMemory", self.ip, self.port)

        # B-Human风格运动参数
        self.walk_params = {
            "MaxStepX": 0.08,  # 最大前进步长(m)
            "MaxStepY": 0.04,  # 侧向步长
            "MaxStepTheta": 0.3,  # 旋转步幅(rad)
            "StepHeight": 0.02,  # 抬脚高度(m)
            "TorsoWx": 0.0,  # 躯干前后倾斜(rad)
            "TorsoWy": 0.07,  # 躯干侧向倾斜
            "MaxStepFrequency": 0.6,  # 步频(Hz)
            "ZmpFactorX": 0.4,  # ZMP稳定性参数
            "ZmpFactorY": 0.25
        }

        # PID控制参数
        self.pid_params = {
            "Kp_roll": 0.15,
            "Ki_roll": 0.001,
            "Kd_roll": 0.05,
            "Kp_pitch": 0.12,
            "Ki_pitch": 0.001,
            "Kd_pitch": 0.04
        }

        # 状态变量
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.last_error_roll = 0.0
        self.last_error_pitch = 0.0
        self.is_fallen = False

        # 初始化传感器订阅
        self.subscribe_sensors()

    def subscribe_sensors(self):
        """订阅IMU传感器数据"""
        self.memory.subscribeToMicroEvent(
            "IMUData", "BHWalker", "on_imu_update",
            {"sensors": [
                "Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value",
                "Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value",
                "Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value",
                "Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value"
            ]}
        )

    def on_imu_update(self, eventName, value, subscriberIdentifier):
        """IMU数据回调函数"""
        if self.is_fallen:
            return

        # 解析传感器数据
        gyro_x = value[0]
        gyro_y = value[1]
        accel_x = value[2]
        accel_y = value[3]

        # 计算姿态角（简化版互补滤波）
        dt = 0.1  # 采样周期
        pitch = math.atan2(accel_x, math.sqrt(accel_y ** 2 + 1)) * 180 / math.pi
        roll = math.atan2(accel_y, math.sqrt(accel_x ** 2 + 1)) * 180 / math.pi

        # PID控制
        self.dynamic_balance_adjust(roll, pitch, dt)

    def dynamic_balance_adjust(self, roll, pitch, dt):
        """动态平衡调整"""
        # 滚转轴PID
        error_roll = -roll  # 目标为0度
        self.integral_roll += error_roll * dt
        derivative_roll = (error_roll - self.last_error_roll) / dt
        adj_roll = (self.pid_params["Kp_roll"] * error_roll +
                    self.pid_params["Ki_roll"] * self.integral_roll +
                    self.pid_params["Kd_roll"] * derivative_roll)

        # 俯仰轴PID
        error_pitch = -pitch
        self.integral_pitch += error_pitch * dt
        derivative_pitch = (error_pitch - self.last_error_pitch) / dt
        adj_pitch = (self.pid_params["Kp_pitch"] * error_pitch +
                     self.pid_params["Ki_pitch"] * self.integral_pitch +
                     self.pid_params["Kd_pitch"] * derivative_pitch)

        # 更新躯干角度
        self.walk_params["TorsoWx"] = max(min(adj_pitch * math.pi / 180, 0.2), -0.2)
        self.walk_params["TorsoWy"] = max(min(adj_roll * math.pi / 180, 0.2), -0.2)
        self.update_walk_params()

        # 保存误差
        self.last_error_roll = error_roll
        self.last_error_pitch = error_pitch

    def update_walk_params(self):
        """更新运动参数"""
        config = []
        for key, value in self.walk_params.items():
            config.append([key, value])
        self.motion.setMotionConfig(config)

    def check_fall_status(self):
        """跌倒检测"""
        posture_name = self.posture.getPosture()
        if posture_name in ["Crouch", "Sit", "LyingBelly", "LyingBack"]:
            self.is_fallen = True
            return True
        return False

    def recover_from_fall(self):
        """跌倒恢复"""
        print("Initiating fall recovery...")
        self.motion.stopMove()
        self.posture.goToPosture("Crouch", 0.8)
        time.sleep(1)
        self.posture.goToPosture("Stand", 0.8)
        self.motion.wakeUp()
        self.is_fallen = False

    def execute_walk(self, duration=30):
        """执行直走任务"""
        try:
            # 初始化状态
            self.motion.wakeUp()
            self.posture.goToPosture("Stand", 0.8)
            self.update_walk_params()

            # 启动直走
            self.motion.moveToward(1.0, 0, 0)  # 全速前进

            # 主控制循环
            start_time = time.time()
            while time.time() - start_time < duration:
                if self.check_fall_status():
                    self.recover_from_fall()
                    self.motion.moveToward(1.0, 0, 0)

                time.sleep(0.1)  # 控制循环周期

        except KeyboardInterrupt:
            self.motion.stopMove()
        finally:
            self.motion.stopMove()
            self.posture.goToPosture("Crouch", 0.5)
            self.motion.rest()


if __name__ == "__main__":
    # 初始化连接
    robot_ip = "192.168.1.100"  # 替换为实际IP
    walker = BHumanWalker("BHWalker", robot_ip)

    # 执行直走30秒
    walker.execute_walk(30)