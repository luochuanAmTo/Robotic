# coding=utf-8
# import cv2
# import numpy as np
#
# def detect_goal_and_obstacle(img, low_yellow, high_yellow):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # 识别黄色区域
#     yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
#     yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 按照面积从大到小排序轮廓
#     yellow_contours = sorted(yellow_contours, key=cv2.contourArea, reverse=True)
#
#     # 选取最大的两个轮廓
#     largest_contours = yellow_contours[:2]
#
#     return largest_contours
#
# def calculate_mid_point(contour):
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         return cx, cy  # 返回X和Y坐标
#     return None
#
# def calculate_centroid_midpoint(centroids):
#     if len(centroids) == 2:
#         mid_x = int((centroids[0][0] + centroids[1][0]) / 2)
#         mid_y = int((centroids[0][1] + centroids[1][1]) / 2)
#         return mid_x, mid_y  # 返回两个质心的中点坐标
#     return None
#
# # 测试代码
# if __name__ == "__main__":
#     img = cv2.imread('goal.jpg')  # 替换为实际测试图像路径
#     low_yellow = np.array([20, 100, 100])
#     high_yellow = np.array([30, 255, 255])
#     largest_contours = detect_goal_and_obstacle(img, low_yellow, high_yellow)
#
#     centroids = []
#     for contour in largest_contours:
#         # 计算每个轮廓的质心
#         mid_point = calculate_mid_point(contour)
#         if mid_point is not None:
#             centroids.append(mid_point)
#             cv2.circle(img, mid_point, 5, (0, 255, 255), -1)  # 标注质心
#
#     # 计算两个质心的中点
#     centroid_midpoint = calculate_centroid_midpoint(centroids)
#     if centroid_midpoint is not None:
#         cv2.circle(img, centroid_midpoint, 5, (0, 0, 255), -1)  # 标注质心的中点
#
#         left_point = (centroid_midpoint[0] - 80, centroid_midpoint[1])
#         right_point = (centroid_midpoint[0] + 80, centroid_midpoint[1])
#         cv2.circle(img, left_point, 5, (255, 0, 0), -1)  # 标注左边80个像素的位置
#         cv2.circle(img, right_point, 5, (0, 255, 0), -1)  # 标注右边80个像素的位置
#
#     cv2.imshow("Detected Goal and Obstacle", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# yellow_line_detection.py
import cv2
import numpy as np

def detect_goal_and_obstacle(img, low_yellow, high_yellow):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = sorted(yellow_contours, key=cv2.contourArea, reverse=True)
    # 添加最小面积检查
    min_area = 100  # 根据图像大小和黄条实际尺寸调整
    large_contours = [cnt for cnt in yellow_contours if cv2.contourArea(cnt) > min_area]

    largest_contours = yellow_contours[:2]
    return largest_contours

def calculate_mid_point(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None

def calculate_centroid_midpoint(centroids):
    if len(centroids) == 2:
        mid_x = int((centroids[0][0] + centroids[1][0]) / 2)
        mid_y = int((centroids[0][1] + centroids[1][1]) / 2)
        return mid_x, mid_y
    return None


import cv2
import numpy as np


def process_goal_image(goal_img, low_yellow, high_yellow):
    largest_contours = detect_goal_and_obstacle(goal_img, low_yellow, high_yellow)
    centroids = []

    # 检查是否检测到至少两个黄条
    if len(largest_contours) < 2:
        print("未能检测到足够的黄条")
        return goal_img, None  # 返回图像和 None，表示无法检测到完整目标

    for contour in largest_contours:
        mid_point = calculate_mid_point(contour)
        if mid_point is not None:
            centroids.append(mid_point)
            cv2.circle(goal_img, mid_point, 5, (0, 255, 255), -1)

    centroid_midpoint = calculate_centroid_midpoint(centroids)
    centroid_midpoint_x = None  # 初始化中点的 x 坐标

    if centroid_midpoint is not None:
        cv2.circle(goal_img, centroid_midpoint, 5, (0, 0, 255), -1)
        left_point = (centroid_midpoint[0] - 70, centroid_midpoint[1])
        right_point = (centroid_midpoint[0] + 70, centroid_midpoint[1])
        cv2.circle(goal_img, left_point, 5, (255, 0, 0), -1)
        cv2.circle(goal_img, right_point, 5, (0, 255, 0), -1)
        centroid_midpoint_x = centroid_midpoint[0]  # 获取中点的 x 坐标

    return goal_img, centroid_midpoint_x

