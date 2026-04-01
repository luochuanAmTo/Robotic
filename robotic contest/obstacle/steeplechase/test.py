# coding:utf-8
import math
import time
import random
import mrtest
from proxy_and_image import *
from recognized_cylinder import *
from control_nao import change_the_postion

# 摄像头参数
frameHeight = 0
frameWidth = 0
frameChannels = 0
frameArray = None
cameraPitchRange = 47.64 / 180 * math.pi
cameraYawRange = 60.97 / 180 * math.pi

# 初始角度及步态参数
row = "HeadPitch"
angle = 0.5235987755982988
maxstepx = 0.10
maxstepy = 0.05  # 减小 Y 方向侧移步长
maxsteptheta = 0.15  # 减少转角以避免偏航
maxstepfrequency = 0.6
stepheight = 0.02
torsowx = 0.0
torsowy = 0.0

# PID 控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# 初始化函数
def initialize_robot(IP):
    AutonomousLifeProxy = get_Proxy("ALAutonomousLife", IP)
    AutonomousLifeProxy.setState("disabled")

    motionProxy = get_Proxy("ALMotion", IP)
    motionProxy.stiffnessInterpolation("Body", 1, 1.5)
    motionProxy.angleInterpolation(["HeadPitch", "HeadYaw"], [0, 0], [0.3, 0.3], True)

    postureProxy = get_Proxy("ALRobotPosture", IP)
    postureProxy.goToPosture("StandInit", 1.5)

    return motionProxy, postureProxy

# 获取IMU姿态角度和角速度（roll/pitch/yaw）
# 获取IMU姿态角度和角速度（roll/pitch/yaw）
def get_imu_data(memoryProxy):
    try:
        roll = memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        pitch = memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        # 尝试新路径
        try:
            yaw_rate = memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeZ/Sensor/Value")
        except RuntimeError:
            yaw_rate = 0.0  # 如果读取不到，返回0，不中断程序
    except RuntimeError as e:
        print("读取IMU数据失败: ", e)
        roll, pitch, yaw_rate = 0.0, 0.0, 0.0
    return roll, pitch, yaw_rate


if __name__ == "__main__":
    IP = CONFIG["ip"]
    motionProxy, postureProxy = initialize_robot(IP)
    motionProxy.setSmartStiffnessEnabled(1)
    memoryProxy = get_Proxy("ALMemory", IP)

    # 调整头部姿态
    change_the_postion(motionProxy, row, angle)

    # 初始化 PID 控制器
    pid_roll = PIDController(Kp=2.0, Ki=0.01, Kd=0.05)
    pid_pitch = PIDController(Kp=2.0, Ki=0.01, Kd=0.05)
    pid_yaw = PIDController(Kp=0.8, Ki=0.01, Kd=0.02)  # 新增 Yaw 控制防偏移

    # 开始移动时实时纠偏（走2秒）
    start_time = time.time()
    duration = 1.0

    print("开始动态平衡移动...")
    while time.time() - start_time < duration:
        roll, pitch, yaw_rate = get_imu_data(memoryProxy)
        now = time.time()
        dt = 0.05

        # 姿态 PID 控制补偿
        torsowy = pid_roll.update(-roll, dt)
        torsowx = pid_pitch.update(-pitch, dt)
        theta = pid_yaw.update(-yaw_rate, dt)

        # 执行 moveToward，行走中不断调整
        motionProxy.moveToward(
            0.5,   # 向前速度 (m/s)
            0.0,   # 侧移速度
            theta,  # 动态修正转向角速度
            [
                ["MaxStepX", maxstepx],
                ["MaxStepY", maxstepy],
                ["MaxStepTheta", maxsteptheta],
                ["MaxStepFrequency", maxstepfrequency],
                ["StepHeight", stepheight],
                ["TorsoWx", torsowx],
                ["TorsoWy", torsowy],
            ],
        )


    # 停止运动
    motionProxy.stopMove()
    print("移动完成。")
