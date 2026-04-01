# coding:utf-8
import random
from naoqi import ALProxy
import cv2
import numpy as np
from proxy_and_image import CONFIG, color_ranges, get_Proxy, get_image_from_camera


def enhanced_preprocess_image(bgr_img, color_ranges):
    """
    增强的图像预处理，提高多障碍检测的准确性
    """
    # 增强对比度和亮度
    image_float = np.float32(bgr_img)
    alpha = 1.8  # 对比度控制
    beta = 50  # 亮度控制
    adjusted = cv2.convertScaleAbs(image_float * alpha + beta)

    # 高斯模糊减噪
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)

    # 转换为HSV
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 初始化掩码
    masks = {}
    all_contours = {}

    # 遍历颜色范围，生成掩码
    for color_name, ranges in color_ranges.items():
        if color_name == 'red':
            # 红色需要两个范围
            lower1, upper1 = ranges[0]
            lower2, upper2 = ranges[1]
            mask1 = cv2.inRange(hsv_img, lower1, upper1)
            mask2 = cv2.inRange(hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv_img, lower, upper)

        # 形态学操作 - 去噪和填充
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)

        # 开运算去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # 闭运算填充内部空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

        masks[color_name] = mask

        # 查找每种颜色的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours[color_name] = contours

    return masks, all_contours


def estimate_obstacle_distance(obstacle_info, image_height):
    """
    根据障碍物在图像中的位置和大小估算距离
    """
    center_y = obstacle_info['center'][1]
    obstacle_height = obstacle_info['height']

    # 简化的距离估算（基于垂直位置）
    # 假设图像底部为近距离，顶部为远距离
    distance_factor = (image_height - center_y) / image_height

    # 结合障碍物大小进行修正
    size_factor = max(50, obstacle_info['width']) / 100.0

    estimated_distance = distance_factor * 3.0 + (1.0 / size_factor)
    return max(0.5, estimated_distance)  # 最小距离0.5米


def filter_valid_obstacles(contours, color_name, min_area=500, max_area=50000):
    """
    筛选有效的障碍物轮廓
    """
    valid_obstacles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 面积筛选
        if area < min_area or area > max_area:
            continue

        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # 长宽比筛选（柱体通常比较高）
        if aspect_ratio > 2.0 or aspect_ratio < 0.3:
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算中心点
        center = tuple(np.mean(box, axis=0).astype(int))

        # 计算轮廓的凸性（convexity）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # 筛选相对规整的形状
        if convexity < 0.6:
            continue

        obstacle_info = {
            'color': color_name,
            'center': center,
            'width': max(rect[1][0], rect[1][1]),  # 取长边作为宽度
            'height': min(rect[1][0], rect[1][1]),  # 取短边作为高度
            'angle': rect[2],
            'area': area,
            'contour': cnt,
            'box': box,
            'convexity': convexity,
            'aspect_ratio': aspect_ratio
        }

        valid_obstacles.append(obstacle_info)

    return valid_obstacles


def detect_cylinder_enhanced(bgr_img, color_ranges):
    """
    增强的柱体检测函数，支持多障碍检测和精确分析
    """
    # 预处理图像
    masks, all_contours = enhanced_preprocess_image(bgr_img, color_ranges)

    # 初始化检测结果
    all_detected_obstacles = []
    output_img = bgr_img.copy()
    image_height, image_width = bgr_img.shape[:2]

    # 为每种颜色处理轮廓
    for color_name, contours in all_contours.items():
        # 筛选有效障碍
        valid_obstacles = filter_valid_obstacles(contours, color_name)

        # 绘制检测结果
        for obstacle in valid_obstacles:
            # 估算距离
            obstacle['estimated_distance'] = estimate_obstacle_distance(obstacle, image_height)

            # 根据颜色选择绘制颜色
            if color_name == 'red':
                draw_color = (0, 0, 255)
            elif color_name == 'blue':
                draw_color = (255, 0, 0)
            elif color_name == 'yellow':
                draw_color = (0, 255, 255)
            else:
                draw_color = (128, 128, 128)

            # 绘制轮廓和边界框
            cv2.drawContours(output_img, [obstacle['box']], 0, draw_color, 2)
            cv2.drawContours(output_img, [obstacle['contour']], -1, draw_color, 1)

            # 绘制中心点
            center = obstacle['center']
            cv2.circle(output_img, center, 5, draw_color, -1)

            # 添加信息标签
            label_text = "{color_name} {obstacle['width']:.0f}x{obstacle['height']:.0f}"
            distance_text = "~{obstacle['estimated_distance']:.1f}m"

            cv2.putText(output_img, label_text,
                        (center[0] - 40, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
            cv2.putText(output_img, distance_text,
                        (center[0] - 30, center[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

            all_detected_obstacles.append(obstacle)

    # 按距离排序（最近的在前）
    all_detected_obstacles.sort(key=lambda x: x['estimated_distance'])

    # 在图像上显示总体信息
    info_text = "Obstacles: {len(all_detected_obstacles)}"
    cv2.putText(output_img, info_text, (10, image_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output_img, all_detected_obstacles


def analyze_obstacle_distribution(obstacles, image_width):
    """
    分析障碍物在图像中的分布
    """
    if not obstacles:
        return {"left_obstacles": [], "center_obstacles": [], "right_obstacles": []}

    # 将图像分为三个区域
    left_boundary = image_width * 0.33
    right_boundary = image_width * 0.67

    distribution = {
        "left_obstacles": [],
        "center_obstacles": [],
        "right_obstacles": []
    }

    for obstacle in obstacles:
        center_x = obstacle['center'][0]

        if center_x < left_boundary:
            distribution["left_obstacles"].append(obstacle)
        elif center_x > right_boundary:
            distribution["right_obstacles"].append(obstacle)
        else:
            distribution["center_obstacles"].append(obstacle)

    return distribution


# 为了兼容性，保留原函数名
def detect_cylinder(bgr_img, color_ranges):
    """
    保持与原代码的兼容性
    """
    return detect_cylinder_enhanced(bgr_img, color_ranges)


def main():
    """增强的测试主函数"""
    try:
        # 初始化NAO摄像头
        video = get_Proxy("ALVideoDevice", CONFIG["ip"])

        # 订阅摄像头
        video_sub = video.subscribeCamera(
            "enhanced_cylinder_client",
            0,  # 上部摄像头
            CONFIG["resolution"],
            CONFIG["colorSpace"],
            CONFIG["fps"],
        )

        print("已启动增强型障碍检测系统，按ESC键退出...")

        while True:
            # 获取图像
            frame = get_image_from_camera(0, video, video_sub)
            if frame is None:
                print("无法获取图像")
                break

            # 检测障碍物
            result_img, obstacles = detect_cylinder_enhanced(frame, color_ranges)

            # 分析障碍物分布
            distribution = analyze_obstacle_distribution(obstacles, frame.shape[1])

            # 显示结果
            cv2.imshow("Enhanced Obstacle Detection", result_img)

            # 打印检测信息
            if obstacles:
                print("\n检测到 {len(obstacles)} 个障碍物:")
                for i, obs in enumerate(obstacles):
                    print("  {i + 1}. {obs['color']} | 中心: {obs['center']} | "
                          "尺寸: {obs['width']:.1f}x{obs['height']:.1f} | "
                          "距离: ~{obs['estimated_distance']:.1f}m")

                print("分布 - 左: {len(distribution['left_obstacles'])}, "
                      "中: {len(distribution['center_obstacles'])}, "
                      "右: {len(distribution['right_obstacles'])}")

            # 退出条件
            if cv2.waitKey(30) == 27:  # ESC键
                break

    except Exception as e:
        print("错误:", e)
    finally:
        # 释放资源
        if 'video' in locals():
            video.unsubscribe(video_sub)
        cv2.destroyAllWindows()
        print("增强检测系统已终止")


if __name__ == "__main__":
    main()