# -*- coding: utf-8 -*-
import os
import cv2

# 定义图像路径
image_path = r'D:\123\python\soccer2\test.jpg'  # 替换为您的图像文件名

# 打印图像路径以进行调试
print("Checking image path:", image_path)

# 检查文件是否存在
if not os.path.isfile(image_path):
    print("Error: The file does not exist: {image_path}")  # 使用f-string格式化
    exit()

# 加载图像
img = cv2.imread(image_path)

# 检查图像是否加载成功
if img is None:
    print("Error: Image not loaded. Check the path and format.")
    exit()

# 进一步处理代码可以在这里添加
print("Image loaded successfully!")  # 可以添加一些处理逻辑
