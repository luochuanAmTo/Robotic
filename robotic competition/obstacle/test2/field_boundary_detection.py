# coding:utf-8
import cv2
import numpy as np
from proxy_and_image import get_Proxy, get_image_from_camera, CONFIG


class FieldBoundaryDetector:
    """
    场地边界检测器 - 用于检测机器人可行走区域的边界
    """

    def __init__(self):
        self.field_width_pixels = None
        self.left_boundary = None
        self.right_boundary = None
        self.calibrated = False

    def detect_ground_edges(self, image):
        """
        检测地面边缘 - 通过颜色差异找到场地边界
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=100, maxLineGap=50)

        return edges, lines

    def find_vertical_boundaries(self, image):
        """
        寻找垂直方向的场地边界
        """
        height, width = image.shape[:2]

        # 只分析图像下半部分（地面区域）
        roi = image[height // 2:, :]

        # 转换为HSV进行颜色分析
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 检测地面颜色（假设为较暗的颜色）
        # 这里可以根据实际场地颜色调整
        lower_ground = np.array([0, 0, 0])
        upper_ground = np.array([180, 255, 100])

        ground_mask = cv2.inRange(hsv, lower_ground, upper_ground)

        # 形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)

        # 水平投影找到左右边界
        horizontal_projection = np.sum(ground_mask, axis=0)

        # 找到有效地面区域的左右边界
        valid_columns = np.where(horizontal_projection > height * 0.1)[0]

        if len(valid_columns) > 0:
            left_boundary = valid_columns[0]
            right_boundary = valid_columns[-1]
            return left_boundary, right_boundary, ground_mask

        return None, None, ground_mask

    def detect_field_lines(self, image):
        """
        检测场地线条（如果有的话）
        """
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测白色线条（场地标线）
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选线状轮廓
        line_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 过滤小噪点
                # 计算长宽比
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 3:  # 线状物体
                        line_contours.append(contour)

        return line_contours

    def calibrate_field_boundaries(self, image):
        """
        校准场地边界
        """
        height, width = image.shape[:2]

        # 方法1：基于颜色检测边界
        left_bound, right_bound, ground_mask = self.find_vertical_boundaries(image)

        if left_bound is not None and right_bound is not None:
            self.left_boundary = left_bound
            self.right_boundary = right_bound
            self.field_width_pixels = right_bound - left_bound
            self.calibrated = True

            print("场地边界校准完成: 左边界={left_bound}, 右边界={right_bound}, 宽度={self.field_width_pixels}像素")
            return True
        else:
            # 如果无法检测到边界，使用默认值
            self.left_boundary = int(width * 0.1)
            self.right_boundary = int(width * 0.9)
            self.field_width_pixels = self.right_boundary - self.left_boundary
            self.calibrated = True

            print("使用默认场地边界: 左={self.left_boundary}, 右={self.right_boundary}")
            return False

    def get_available_space(self, obstacle_center_x):
        """
        根据障碍物位置计算左右可用空间
        """
        if not self.calibrated:
            return None, None

        # 计算左侧空间（从左边界到障碍物）
        left_space_pixels = obstacle_center_x - self.left_boundary

        # 计算右侧空间（从障碍物到右边界）
        right_space_pixels = self.right_boundary - obstacle_center_x

        # 转换为相对比例（0-1）
        total_width = self.field_width_pixels
        left_space_ratio = max(0, left_space_pixels / total_width)
        right_space_ratio = max(0, right_space_pixels / total_width)

        return left_space_ratio, right_space_ratio

    def visualize_boundaries(self, image):
        """
        在图像上可视化检测到的边界
        """
        if not self.calibrated:
            return image

        result = image.copy()
        height = image.shape[0]

        # 绘制左右边界线
        cv2.line(result, (self.left_boundary, 0), (self.left_boundary, height), (0, 255, 0), 2)
        cv2.line(result, (self.right_boundary, 0), (self.right_boundary, height), (0, 255, 0), 2)

        # 添加标签
        cv2.putText(result, "LEFT", (self.left_boundary + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "RIGHT", (self.right_boundary - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return result


def test_boundary_detection():
    """
    测试边界检测功能
    """
    try:
        # 初始化摄像头
        video_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
        video_client = video_proxy.subscribeCamera(
            "boundary_test", 0, CONFIG["resolution"],
            CONFIG["colorSpace"], CONFIG["fps"]
        )

        # 初始化边界检测器
        boundary_detector = FieldBoundaryDetector()

        print("边界检测测试启动，按ESC退出...")

        frame_count = 0
        while True:
            # 获取图像
            frame = get_image_from_camera(0, video_proxy, video_client)
            if frame is None:
                continue

            # 前几帧用于校准
            if frame_count < 10:
                boundary_detector.calibrate_field_boundaries(frame)

            # 可视化边界
            result = boundary_detector.visualize_boundaries(frame)

            # 显示额外信息
            if boundary_detector.calibrated:
                info_text = "Field Width: {boundary_detector.field_width_pixels} pixels"
                cv2.putText(result, info_text, (10, result.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Field Boundary Detection", result)

            if cv2.waitKey(30) == 27:  # ESC
                break

            frame_count += 1

    except Exception as e:
        print("测试错误: {e}")
    finally:
        if 'video_proxy' in locals():
            video_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_boundary_detection()