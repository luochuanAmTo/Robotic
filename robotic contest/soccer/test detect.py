# coding=utf-8
# import cv2
# import numpy as np
#
#
# def detect_goal_and_obstacle(img, low_yellow, high_yellow):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # 识别黄色区域
#     # yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
#     # yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # yellow_centers = [cv2.minEnclosingCircle(cnt)[0] for cnt in yellow_contours]
#     # 识别黄色区域
#     yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
#
#     # 对二值掩模进行形态学操作，去除噪声
#     kernel = np.ones((5, 5), np.uint8)
#     yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
#
#     # 查找轮廓
#     yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 计算每个符合面积要求的轮廓的中心点
#     yellow_centers = []
#     for cnt in yellow_contours:
#         area = cv2.contourArea(cnt)
#         if area > 100:  # 过滤掉小的轮廓
#             center = cv2.minEnclosingCircle(cnt)[0]
#             yellow_centers.append(center)
#
#     # 调试输出，查看识别到的黄色区域中心点
#     print("Yellow centers:", yellow_centers)
#
#     # 寻找最左边和最右边的黄色区域
#     if yellow_centers:
#         leftmost = min(yellow_centers, key=lambda x: x[0])
#         rightmost = max(yellow_centers, key=lambda x: x[0])
#         return [leftmost, rightmost], yellow_centers
#
#     return [], []
#
#
# def calculate_mid_point(yellow_centers):
#     if len(yellow_centers) == 2:
#         y_center = yellow_centers[0]
#         g_center = yellow_centers[1]
#         mid_point = (int((y_center[0] + g_center[0]) / 2), int((y_center[1] + g_center[1]) / 2))
#         return mid_point[0], mid_point  # 返回 X 坐标和中点坐标
#     return None, None
#
#
# # 测试代码
# if __name__ == "__main__":
#     img = cv2.imread('goal.jpg')  # 替换为实际测试图像路径
#     low_yellow = np.array([20, 100, 100])
#     high_yellow = np.array([30, 255, 255])
#     yellow_positions, yellow_centers = detect_goal_and_obstacle(img, low_yellow, high_yellow)
#     mid_x, mid_point = calculate_mid_point(yellow_positions)
#
#     # 标注所有识别到的黄色点
#     for center in yellow_centers:
#         cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
#
#     if mid_x is not None:
#         cv2.circle(img, mid_point, 5, (0, 0, 255), -1)  # 标注中点
#         left_point = (mid_point[0] - 80, mid_point[1])
#         right_point = (mid_point[0] + 80, mid_point[1])
#         cv2.circle(img, left_point, 5, (255, 0, 0), -1)  # 标注左边80个像素的位置
#         cv2.circle(img, right_point, 5, (0, 255, 0), -1)  # 标注右边80个像素的位置
#
#         cv2.imshow("Detected Goal and Obstacle", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("未能识别到两条黄色竖线")
import cv2
import numpy as np


def detect_goal_and_obstacle(img, low_yellow, high_yellow):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 识别黄色区域
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

    # 对二值掩模进行形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return yellow_contours


def calculate_mid_point(contour):
    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy  # 返回X和Y坐标
    return None


def calculate_centroid_midpoint(centroids):
    if len(centroids) == 2:
        mid_x = int((centroids[0][0] + centroids[1][0]) / 2)
        mid_y = int((centroids[0][1] + centroids[1][1]) / 2)
        return mid_x, mid_y  # 返回两个质心的中点坐标
    return None


if __name__ == "__main__":
    img = cv2.imread('failgoal.jpg')  # 替换为实际测试图像路径
    low_yellow = np.array([0, 139, 0])
    high_yellow = np.array([46, 255, 88])
    yellow_contours = detect_goal_and_obstacle(img, low_yellow, high_yellow)

    if yellow_contours:
        # 按照面积从大到小排序轮廓
        yellow_contours = sorted(yellow_contours, key=cv2.contourArea, reverse=True)

        # 选择面积最大的两个轮廓
        largest_contours = yellow_contours[:2]
        centroids = []

        for contour in largest_contours:
            # 绘制每个黄色区域的轮廓
            cv2.drawContours(img, [contour], -1, (0, 255, 255), 2)  # 黄色轮廓

            # 计算每个轮廓的质心
            mid_point = calculate_mid_point(contour)
            if mid_point is not None:
                centroids.append(mid_point)
                cv2.circle(img, mid_point, 5, (0, 0, 255), -1)  # 标注质心

        # 计算两个质心的中点
        centroid_midpoint = calculate_centroid_midpoint(centroids)
        if centroid_midpoint is not None:
            cv2.circle(img, centroid_midpoint, 5, (255, 0, 0), -1)  # 标注质心的中点

        cv2.imshow("Detected Goal and Obstacle", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能识别到黄色区域")

