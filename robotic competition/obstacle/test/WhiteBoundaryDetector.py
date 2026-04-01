# coding:utf-8
import cv2
import numpy as np
import time
from proxy_and_image import get_Proxy, get_image_from_camera, CONFIG


class WhiteBoundaryDetector:
    """
    专门针对白色边线的检测器
    """

    def __init__(self):
        self.left_boundary = None
        self.right_boundary = None
        self.field_width_pixels = None
        self.calibrated = False
        self.warning_distance = 50  # 距离边线50像素时开始警告
        self.danger_distance = 30  # 距离边线30像素时进入危险区域

    def detect_white_lines(self, image):
        """
        专门检测白色边线
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 检测白色线条 - 使用更严格的阈值
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        # 形态学操作 - 连接断开的线段
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_line)

        # 去除小的噪点
        kernel_clean = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_clean)

        return white_mask

    def find_vertical_boundaries_from_lines(self, white_mask):
        """
        从白色线条掩码中找到垂直边界
        """
        height, width = white_mask.shape

        # 只分析图像下半部分（地面区域）
        roi_mask = white_mask[height // 2:, :]

        # 垂直投影 - 计算每列的白色像素数量
        vertical_projection = np.sum(roi_mask, axis=0)

        # 平滑投影结果
        kernel_size = 5
        smoothed_projection = np.convolve(vertical_projection,
                                          np.ones(kernel_size) / kernel_size, mode='same')

        # 设定阈值检测边线
        threshold = np.max(smoothed_projection) * 0.3
        line_positions = np.where(smoothed_projection > threshold)[0]

        if len(line_positions) == 0:
            return None, None

        # 寻找左右边界
        # 假设左边界是最左侧的强烈白线，右边界是最右侧的强烈白线
        gaps = np.diff(line_positions)
        large_gaps = np.where(gaps > 50)[0]  # 寻找大间隔

        if len(large_gaps) >= 1:
            # 找到明显的左右分界
            left_boundary = line_positions[large_gaps[0]]
            right_boundary = line_positions[large_gaps[0] + 1]
        else:
            # 没有明显分界，使用最左和最右的线
            left_boundary = line_positions[0]
            right_boundary = line_positions[-1]

        return left_boundary, right_boundary

    def detect_boundary_lines_hough(self, image):
        """
        使用霍夫变换检测边界线
        """
        # 检测白色线条
        white_mask = self.detect_white_lines(image)

        # 边缘检测
        edges = cv2.Canny(white_mask, 50, 150)

        # 霍夫线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=100, maxLineGap=20)

        if lines is None:
            return None, None, white_mask

        # 筛选垂直线（场地边界通常是垂直的）
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算线的角度
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # 接近垂直的线（80-100度或-10到10度）
            if angle > 80 or angle < 10:
                vertical_lines.append(line[0])

        if len(vertical_lines) < 2:
            return None, None, white_mask

        # 按x坐标排序，选择最左和最右的线
        vertical_lines.sort(key=lambda line: min(line[0], line[2]))
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]

        # 计算线的x坐标中点作为边界
        left_boundary = (left_line[0] + left_line[2]) // 2
        right_boundary = (right_line[0] + right_line[2]) // 2

        return left_boundary, right_boundary, white_mask

    def calibrate_boundaries(self, image):
        """
        校准场地边界
        """
        height, width = image.shape[:2]

        # 方法1：直接从白色线条检测
        left_bound, right_bound = self.find_vertical_boundaries_from_lines(
            self.detect_white_lines(image))

        # 方法2：霍夫变换检测（作为备选）
        if left_bound is None or right_bound is None:
            left_bound, right_bound, _ = self.detect_boundary_lines_hough(image)

        if left_bound is not None and right_bound is not None:
            # 确保左右边界合理
            if left_bound < right_bound and (right_bound - left_bound) > width * 0.3:
                self.left_boundary = left_bound
                self.right_boundary = right_bound
                self.field_width_pixels = right_bound - left_bound
                self.calibrated = True

                print(
                    "白色边线检测成功: 左边界={left_bound}, 右边界={right_bound}, 场地宽度={self.field_width_pixels}像素")
                return True

        # 使用默认值
        print("无法检测到白色边线，使用默认边界")
        self.left_boundary = int(width * 0.15)
        self.right_boundary = int(width * 0.85)
        self.field_width_pixels = self.right_boundary - self.left_boundary
        self.calibrated = True
        return False

    def check_boundary_violation(self, robot_center_x):
        """
        检查是否即将越线或已经越线
        """
        if not self.calibrated:
            return "unknown", 0

        # 计算到左右边界的距离
        distance_to_left = robot_center_x - self.left_boundary
        distance_to_right = self.right_boundary - robot_center_x

        # 找到最近的边界距离
        min_distance = min(distance_to_left, distance_to_right)
        closest_side = "left" if distance_to_left < distance_to_right else "right"

        # 判断危险程度
        if min_distance < 0:
            return "violation", min_distance  # 已经越线
        elif min_distance < self.danger_distance:
            return "danger", min_distance  # 危险区域
        elif min_distance < self.warning_distance:
            return "warning", min_distance  # 警告区域
        else:
            return "safe", min_distance  # 安全区域

    def get_safe_avoidance_direction(self, obstacle_center_x, robot_center_x):
        """
        获取安全的避障方向（确保不越线）
        """
        if not self.calibrated:
            return None, "boundary_not_calibrated"

        # 计算避障后的位置（假设横移距离）
        avoidance_distance = 60  # 像素

        # 检查左侧避障是否安全
        left_position = robot_center_x - avoidance_distance
        left_safe = left_position > (self.left_boundary + self.warning_distance)

        # 检查右侧避障是否安全
        right_position = robot_center_x + avoidance_distance
        right_safe = right_position < (self.right_boundary - self.warning_distance)

        # 计算到障碍物的左右空间
        left_space = obstacle_center_x - self.left_boundary
        right_space = self.right_boundary - obstacle_center_x

        # 综合判断最佳方向
        if left_safe and right_safe:
            # 两边都安全，选择空间更大的一边
            return "left" if left_space > right_space else "right", "both_safe"
        elif left_safe and not right_safe:
            return "left", "right_unsafe"
        elif right_safe and not left_safe:
            return "right", "left_unsafe"
        else:
            return None, "no_safe_direction"

    def visualize_boundaries_and_warnings(self, image, robot_center_x=None):
        """
        可视化边界和警告信息
        """
        if not self.calibrated:
            return image

        result = image.copy()
        height, width = image.shape[:2]

        # 绘制边界线
        cv2.line(result, (self.left_boundary, 0), (self.left_boundary, height), (0, 255, 0), 3)
        cv2.line(result, (self.right_boundary, 0), (self.right_boundary, height), (0, 255, 0), 3)

        # 绘制警告区域
        warning_left = self.left_boundary + self.warning_distance
        warning_right = self.right_boundary - self.warning_distance
        cv2.line(result, (warning_left, 0), (warning_left, height), (0, 255, 255), 2)
        cv2.line(result, (warning_right, 0), (warning_right, height), (0, 255, 255), 2)

        # 绘制危险区域
        danger_left = self.left_boundary + self.danger_distance
        danger_right = self.right_boundary - self.danger_distance
        cv2.line(result, (danger_left, 0), (danger_left, height), (0, 0, 255), 2)
        cv2.line(result, (danger_right, 0), (danger_right, height), (0, 0, 255), 2)

        # 添加标签
        cv2.putText(result, "BOUNDARY", (self.left_boundary + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "BOUNDARY", (self.right_boundary - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 如果提供了机器人位置，显示状态
        if robot_center_x is not None:
            status, distance = self.check_boundary_violation(robot_center_x)

            # 选择状态颜色
            if status == "violation":
                color = (0, 0, 255)  # 红色
                text = "VIOLATION! {distance:.0f}px"
            elif status == "danger":
                color = (0, 100, 255)  # 橙色
                text ="DANGER {distance:.0f}px"
            elif status == "warning":
                color = (0, 255, 255)  # 黄色
                text = "WARNING {distance:.0f}px"
            else:
                color = (0, 255, 0)  # 绿色
                text = "SAFE {distance:.0f}px"

            # 显示机器人位置和状态
            cv2.circle(result, (robot_center_x, height // 2), 10, color, -1)
            cv2.putText(result, text, (10, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result


def test_white_boundary_detection():
    """
    测试白色边线检测功能
    """
    try:
        # 初始化摄像头
        video_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
        video_client = video_proxy.subscribeCamera(
            "white_boundary_test", 0, CONFIG["resolution"],
            CONFIG["colorSpace"], CONFIG["fps"]
        )

        # 初始化白色边线检测器
        boundary_detector = WhiteBoundaryDetector()

        print("白色边线检测测试启动，按ESC退出...")

        frame_count = 0
        robot_center_x = None

        while True:
            # 获取图像
            frame = get_image_from_camera(1, video_proxy, video_client)
            if frame is None:
                continue

            height, width = frame.shape[:2]

            # 前几帧用于校准边界
            if frame_count < 10:
                boundary_detector.calibrate_boundaries(frame)

            # 模拟机器人在图像中央（实际应用中从其他传感器获取）
            if robot_center_x is None:
                robot_center_x = width // 2

            # 检测白色线条并可视化
            white_mask = boundary_detector.detect_white_lines(frame)

            # 可视化边界和警告
            result = boundary_detector.visualize_boundaries_and_warnings(frame, robot_center_x)

            # 显示白色线条检测结果
            cv2.imshow("White Lines Detection", white_mask)
            cv2.imshow("Boundary Detection with Warnings", result)

            # 打印状态信息
            if boundary_detector.calibrated:
                status, distance = boundary_detector.check_boundary_violation(robot_center_x)
                print("Frame {frame_count}: Status={status}, Distance={distance:.1f}px")

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
    test_white_boundary_detection()