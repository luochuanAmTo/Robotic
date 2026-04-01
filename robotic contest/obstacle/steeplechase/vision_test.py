# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 读取图像
image = cv2.imread('blue.jpg')

# 将BGR图像转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义蓝色的HSV阈值范围
lower_blue = np.array([80, 50,50])
upper_blue = np.array([130, 230, 255])

# 根据阈值创建掩码
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 使用开运算去除噪声
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 使用中值滤波进一步平滑图像
median = cv2.medianBlur(opening, 5)

# 找到最大的连通区域作为蓝色柱体
contours, _ = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    result = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)
else:
    result = image.copy()

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Mask', median)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()