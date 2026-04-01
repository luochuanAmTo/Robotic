# coding:utf-8

import cv2
import numpy as np
from proxy_and_image import CONFIG

def preprocess_image(bgr_img, low_black, high_black, low_white, high_white):
    """
    对图像进行预处理，包括HSV转换，颜色过滤，平滑处理，腐蚀和膨胀
    """
    # 增强对比度
    image_float = np.float32(bgr_img)
    alpha = 1.5  # 对比度控制 (1.0-3.0)
    beta = 50   # 亮度控制 (0-100)
    adjusted = cv2.convertScaleAbs(image_float * alpha + beta)

    # 转换为HSV
    hsv_img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    # 颜色分割：创建黑色和白色掩码
    black_mask = cv2.inRange(hsv_img, low_black, high_black)
    white_mask = cv2.inRange(hsv_img, low_white, high_white)

    # 组合黑色和白色掩码
    combined_mask = cv2.add(black_mask, white_mask)

    # 使用自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(combined_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 使用Canny边缘检测
    edges = cv2.Canny(adaptive_thresh, 50, 150)

    # 形态学操作，填补空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return combined_mask, closed


def detect_circle(bgr_img, low_black, high_black, low_white, high_white):
    """
    检测图像中的圆形物体
    """
    combined_mask, preprocessed_image = preprocess_image(bgr_img, low_black, high_black, low_white, high_white)

    # 霍夫圆检测
    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100,
    )

    center = None
    radius = None
    if circles is not None:
        # 仅取最大的一个圆
        x, y, radius = circles[0][0]
        center = (x, y)
        cv2.circle(bgr_img, center, int(radius), (0, 255, 0), 2)
        return center, int(radius), combined_mask
    else:
        return None, None, combined_mask


def main():
    # 从 CONFIG 读取阈值参数
    low_black = np.array(CONFIG["black_low"])
    high_black = np.array(CONFIG["black_high"])
    low_white = np.array(CONFIG["white_low"])
    high_white = np.array(CONFIG["white_high"])

    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            center, radius, combined_mask = detect_circle(
                img,
                low_black,
                high_black,
                low_white,
                high_white,
            )
            if center is not None:
                print("Detected circle at {} with radius {}".format(center, radius))

            # 显示原始图像和掩码
            result = cv2.bitwise_and(img, img, mask=combined_mask)

            cv2.imshow("res", img)
            cv2.imshow("combined_mask", combined_mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
