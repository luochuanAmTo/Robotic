# coding:utf-8
import cv2
from detect_goal_and_obstacle import process_goal_image
from control_nao import change_the_postion


class AlignmentStateMachine:
    def __init__(self, mt_proxy, low_yellow, high_yellow, head_min_angle, head_max_angle, head_step):
        self.mt_proxy = mt_proxy
        self.state = "ALIGN"
        self.low_yellow = low_yellow
        self.high_yellow = high_yellow
        self.head_min_angle = head_min_angle
        self.head_max_angle = head_max_angle
        self.head_step = head_step
        self.head_angle = 0
        self.direction = 1  # 1 表示向右，-1 表示向左
        self.row = "HeadPitch"
        self.angle = 0.5235987755982988  # 初始角度

    def update(self, cir_center, mid_x, goal_img):
        if mid_x is not None:
            if cir_center[0] - mid_x > 70:
                self.state = "TURN_RIGHT"
            elif mid_x - cir_center[0] > 70:
                self.state = "TURN_LEFT"
            else:
                self.state = "ALIGN"
        else:
            self.state = "FAIL"

        self.execute(cir_center, goal_img)

    def execute(self, cir_center, goal_img):
        while self.state != "ALIGN":
            if self.state == "FAIL":
                print("未能识别到球门或障碍物，调整角度")
                self.adjust_head_angle()  # 调用调整头部角度的方法

                # 继续尝试找到两个黄条
                goal_img, mid_x = process_goal_image(goal_img, self.low_yellow, self.high_yellow)

                if mid_x is not None:
                    # 根据头部转动方向调整机器人
                    if self.direction == 1:  # 如果头部在向右转
                        print("头部向右转时检测到两条黄线，机器人向左转")
                        self.state = "TURN_LEFT"
                    else:  # 如果头部在向左转
                        print("头部向左转时检测到两条黄线，机器人向右转")
                        self.state = "TURN_RIGHT"

                    # 调整机器人方向后，恢复头部初始角度
                    self.restore_head_position()

                    # 继续检测黄条是否保持可见
                    goal_img, mid_x = process_goal_image(goal_img, self.low_yellow, self.high_yellow)
                    if mid_x is None:
                        self.state = "FAIL"
                        continue  # 如果失去目标，继续调整头部寻找

            elif self.state == "TURN_RIGHT":
                print("向右转")
                self.mt_proxy.moveTo(0, 0, -0.1)
                cv2.imwrite("rgoal.jpg", goal_img)
                self.state = "ALIGN"
            elif self.state == "TURN_LEFT":
                print("向左转")
                self.mt_proxy.moveTo(0, 0, 0.1)
                cv2.imwrite("lgoal.jpg", goal_img)
                self.state = "ALIGN"

        print("球门角度正确")
        cv2.imwrite("goal.jpg", goal_img)

    def adjust_head_angle(self):
        self.head_angle += self.direction * self.head_step
        if self.head_angle > self.head_max_angle or self.head_angle < self.head_min_angle:
            self.direction *= -1  # 改变转动方向
            self.head_angle += self.direction * self.head_step  # 修正角度到有效范围内
        change_the_postion(self.mt_proxy, "HeadYaw", self.head_angle)
        print("转头寻找: ", self.head_angle)

    def restore_head_position(self):
        # 恢复头部到初始角度
        change_the_postion(self.mt_proxy, self.row, self.angle)
        print("恢复头部到初始角度: ", self.angle)
