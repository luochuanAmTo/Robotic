# coding:utf-8
import random
from naoqi import ALProxy
import cv2
import numpy as np
from proxy_and_image import CONFIG, color_ranges, get_Proxy, get_image_from_camera


def preprocess_image(bgr_img, color_ranges):
    """
    对图像进行预处理，包括HSV转换，颜色过滤，平滑处理，腐蚀和膨胀
    """
    # 增强对比度
    image_float = np.float32(bgr_img)
    alpha = 2  # 对比度控制 (1.0-3.0)
    beta = 80   # 亮度控制 (0-100)
    adjusted = cv2.convertScaleAbs(image_float * alpha + beta)

    # 转换为HSV
    hsv_img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    # 初始化掩码
    mask_combined = np.zeros_like(hsv_img[:, :, 0])

    # 遍历颜色范围，生成掩码
    masks = {}
    for color_name, ranges in color_ranges.items():
        if color_name == 'red':
            # 红色需要两个范围（HSV是环状的）
            lower1, upper1 = ranges[0]
            lower2, upper2 = ranges[1]
            mask1 = cv2.inRange(hsv_img, lower1, upper1)
            mask2 = cv2.inRange(hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv_img, lower, upper)

        # 腐蚀和膨胀（去噪）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        masks[color_name] = mask
        mask_combined = cv2.bitwise_or(mask_combined, mask)

    # 查找所有轮廓
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return masks, contours


def detect_cylinder(bgr_img, color_ranges):
    """
        检测红、蓝、黄三个柱体
        :param bgr_img: 输入的BGR图像
        :return: 检测结果（带标记的图像和检测信息）
    """

    # 预处理图像
    masks, contours = preprocess_image(bgr_img, color_ranges)

    # 初始化检测结果
    detected_rectangles = []
    output_img = bgr_img.copy()

    # 遍历所有轮廓
    for cnt in contours:
        # 过滤小轮廓（根据实际场景调整阈值）
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算矩形中心点
        center = tuple(np.mean(box, axis=0).astype(int))

        # 判断颜色
        for color, mask in masks.items():
            if mask[center[1], center[0]] == 255:
                # 根据颜色绘制矩形
                if color == 'red':
                    cv2.drawContours(output_img, [box], 0, (0, 0, 255), 2)
                elif color == 'blue':
                    cv2.drawContours(output_img, [box], 0, (255, 0, 0), 2)
                elif color == 'yellow':
                    cv2.drawContours(output_img, [box], 0, (0, 255, 255), 2)

                # 记录检测信息
                detected_rectangles.append({
                    'color': color,
                    'center': center,
                    'width': rect[1][0],
                    'height': rect[1][1],
                    'angle': rect[2]
                })
                break

    return output_img, detected_rectangles


def main():
    """主函数：订阅NAO摄像头并实时检测"""
    try:
        # 初始化NAO摄像头
        video = get_Proxy("ALVideoDevice", CONFIG["ip"])
        subscriber_id = "cylinder_client"
        camera_index = 0  # 0: 顶部摄像头, 1: 底部摄像头
        resolution = 2  # 2: VGA (640x480)
        color_space = 11  # 11: BGR格式

        # 订阅摄像头
        video_sub = video.subscribeCamera(
            subscriber_id,
            0,  # 上部摄像头
            CONFIG["resolution"],
            CONFIG["colorSpace"],
            CONFIG["fps"],
        )

        print
        "已启动NAO摄像头订阅，按ESC键退出..."

        while True:
            # 获取图像
            frame = get_image_from_camera(0, video, video_sub)
            if frame is None:
                print
                "无法获取图像"
                break

            # 转换为OpenCV格式
            image_np = np.frombuffer(frame[6], dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # 检测柱体
            result_img, rectangles = detect_cylinder(image)

            # 显示结果
            cv2.imshow("NAO Vision - Rectangle Detection", result_img)

            # 打印检测信息
            for rect in rectangles:
                print
                "检测到 {} 柱体 | 中心点: {} | 尺寸: {:.1f}x{:.1f} | 角度: {:.1f}°".format(
                    rect['color'], rect['center'], rect['size'][0], rect['size'][1], rect['angle']
                )

            # 退出条件
            if cv2.waitKey(30) == 27:  # ESC键
                break

    except Exception as e:
        print
        "错误:", e
    finally:
        # 释放资源
        if 'video' in locals():
            video.unsubscribe(video_sub)
        cv2.destroyAllWindows()
        print
        "程序已终止"


if __name__ == "__main__":
    main()  # 替换为你的NAO机器人IP
