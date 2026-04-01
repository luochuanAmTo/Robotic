# coding:utf-8
import cv2
import numpy as np
import time
from proxy_and_image import get_Proxy, get_image_from_camera, CONFIG


class EnhancedWhiteBoundaryDetector:
    """
    增强的白色边线检测器 - 专注于精确的边线检测和中心导航
    """

    def __init__(self):
        self.left_boundary = None
        self.right_boundary = None
        self.field_center = None
        self.field_width_pixels = None
        self.calibrated = False

        # 检测参数
        self.white_threshold_low = 170  # 白色检测下限
        self.white_threshold_high = 255  # 白色检测上限
        self.min_line_length = 80  # 最小线段长度
        self.max_line_gap = 30  # 最大线段间隔

        # 安全距离参数
        self.boundary_safety_margin = 40  # 边界安全距离
        self.center_tolerance = 30  # 中心位置容忍度

        # 历史数据平滑
        self.boundary_history = []
        self.history_size = 5

    def detect_white_lines_enhanced(self, image):
        """
        增强的白色线条检测
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊减噪
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 自适应阈值检测白色区域
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 固定阈值检测强白色区域
        _, fixed_thresh = cv2.threshold(
            blurred, self.white_threshold_low, 255, cv2.THRESH_BINARY
        )

        # 结合两种方法
        combined_mask = cv2.bitwise_or(adaptive_thresh, fixed_thresh)

        # 形态学操作 - 加强线条连接
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

        # 去除噪点
        kernel_open = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

        return combined_mask

    def find_boundary_lines_robust(self, white_mask):
        """
        鲁棒的边界线检测方法
        """
        height, width = white_mask.shape

        # 只分析图像下2/3部分（地面区域）
        roi_start = height // 3
        roi_mask = white_mask[roi_start:, :]

        # 方法1: 垂直投影法
        vertical_projection = np.sum(roi_mask, axis=0)

        # 平滑投影
        kernel_size = 7
        smoothed = np.convolve(vertical_projection,
                               np.ones(kernel_size) / kernel_size, mode='same')

        # 寻找峰值（白线位置）
        threshold = np.max(smoothed) * 0.4
        peaks = []

        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > threshold and
                    smoothed[i] > smoothed[i - 1] and
                    smoothed[i] > smoothed[i + 1]):
                peaks.append(i)

        # 方法2: 霍夫线检测
        edges = cv2.Canny(roi_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=40,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)

        hough_peaks = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 检查是否为近似垂直线
                if abs(x2 - x1) < 20:  # 垂直线
                    avg_x = (x1 + x2) // 2
                    hough_peaks.append(avg_x)

        # 合并两种方法的结果
        all_candidates = peaks + hough_peaks
        all_candidates = sorted(set(all_candidates))

        if len(all_candidates) < 2:
            return None, None

        # 选择最左和最右的候选线作为边界
        left_boundary = all_candidates[0]
        right_boundary = all_candidates[-1]

        # 验证边界合理性
        field_width = right_boundary - left_boundary
        if field_width < width * 0.3 or field_width > width * 0.9:
            return None, None

        return left_boundary, right_boundary

    def smooth_boundaries(self, left_bound, right_bound):
        """
        使用历史数据平滑边界检测结果
        """
        if left_bound is None or right_bound is None:
            return self.left_boundary, self.right_boundary

        # 添加到历史记录
        self.boundary_history.append((left_bound, right_bound))

        # 保持历史记录大小
        if len(self.boundary_history) > self.history_size:
            self.boundary_history.pop(0)

        # 计算平均值
        if len(self.boundary_history) >= 3:
            lefts = [b[0] for b in self.boundary_history]
            rights = [b[1] for b in self.boundary_history]

            # 去除异常值后计算平均
            lefts_filtered = self._remove_outliers(lefts)
            rights_filtered = self._remove_outliers(rights)

            smooth_left = int(np.mean(lefts_filtered))
            smooth_right = int(np.mean(rights_filtered))

            return smooth_left, smooth_right

        return left_bound, right_bound

    def _remove_outliers(self, data):
        """
        移除异常值
        """
        if len(data) < 3:
            return data

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [x for x in data if lower_bound <= x <= upper_bound]

    def calibrate_boundaries_enhanced(self, image):
        """
        增强的边界校准
        """
        height, width = image.shape[:2]

        # 检测白色线条
        white_mask = self.detect_white_lines_enhanced(image)

        # 找到边界线
        left_bound, right_bound = self.find_boundary_lines_robust(white_mask)

        # 平滑处理
        left_bound, right_bound = self.smooth_boundaries(left_bound, right_bound)

        if left_bound is not None and right_bound is not None:
            self.left_boundary = left_bound
            self.right_boundary = right_bound
            self.field_width_pixels = right_bound - left_bound
            self.field_center = (left_bound + right_bound) // 2
            self.calibrated = True

            print(
                "边界校准成功: 左={left_bound}, 右={right_bound}, 中心={self.field_center}, 宽度={self.field_width_pixels}")
            return True
        else:
            # 使用默认值或保持之前的值
            if not self.calibrated:
                self.left_boundary = int(width * 0.2)
                self.right_boundary = int(width * 0.8)
                self.field_width_pixels = self.right_boundary - self.left_boundary
                self.field_center = width // 2
                self.calibrated = True
                print("使用默认边界设置")

            return False

    def calculate_center_deviation(self, robot_x):
        """
        计算机器人相对于场地中心的偏差
        """
        if not self.calibrated:
            return 0, "unknown"

        deviation = robot_x - self.field_center
        deviation_ratio = deviation / (self.field_width_pixels / 2)

        if abs(deviation) <= self.center_tolerance:
            status = "centered"
        elif deviation < 0:
            status = "left_of_center"
        else:
            status = "right_of_center"

        return deviation, status

    def get_centering_correction(self, robot_x):
        """
        获取回到中心的修正方向和强度
        """
        deviation, status = self.calculate_center_deviation(robot_x)

        if status == "centered":
            return 0, "no_correction_needed"

        # 计算修正强度（归一化到-1到1）
        max_deviation = self.field_width_pixels / 2
        correction_strength = np.clip(deviation / max_deviation, -1, 1)

        # 根据偏差大小确定修正策略
        if abs(correction_strength) > 0.6:
            urgency = "high"
        elif abs(correction_strength) > 0.3:
            urgency = "medium"
        else:
            urgency = "low"

        return correction_strength, urgency

    def check_boundary_safety(self, robot_x):
        """
        检查边界安全性
        """
        if not self.calibrated:
            return "unknown", 0

        left_distance = robot_x - self.left_boundary
        right_distance = self.right_boundary - robot_x
        min_distance = min(left_distance, right_distance)

        if min_distance < 0:
            return "violation", min_distance
        elif min_distance < self.boundary_safety_margin:
            return "danger", min_distance
        else:
            return "safe", min_distance

    def visualize_enhanced(self, image, robot_x=None):
        """
        增强的可视化
        """
        if not self.calibrated:
            return image

        result = image.copy()
        height = image.shape[0]

        # 绘制边界线
        cv2.line(result, (self.left_boundary, 0),
                 (self.left_boundary, height), (0, 255, 0), 3)
        cv2.line(result, (self.right_boundary, 0),
                 (self.right_boundary, height), (0, 255, 0), 3)

        # 绘制中心线
        cv2.line(result, (self.field_center, 0),
                 (self.field_center, height), (255, 0, 255), 2)

        # 绘制安全区域
        safe_left = self.left_boundary + self.boundary_safety_margin
        safe_right = self.right_boundary - self.boundary_safety_margin
        cv2.line(result, (safe_left, 0), (safe_left, height), (0, 255, 255), 1)
        cv2.line(result, (safe_right, 0), (safe_right, height), (0, 255, 255), 1)

        # 标注
        cv2.putText(result, "LEFT", (self.left_boundary + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "RIGHT", (self.right_boundary - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "CENTER", (self.field_center - 30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # 如果提供了机器人位置
        if robot_x is not None:
            # 机器人位置标记
            cv2.circle(result, (robot_x, height // 2), 8, (0, 0, 255), -1)

            # 计算偏差信息
            deviation, status = self.calculate_center_deviation(robot_x)
            safety, distance = self.check_boundary_safety(robot_x)

            # 状态颜色
            if safety == "violation":
                color = (0, 0, 255)
            elif safety == "danger":
                color = (0, 100, 255)
            elif status == "centered":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            # 显示状态信息
            info_text = "Dev: {deviation:+.0f}px | {status} | {safety}"
            cv2.putText(result, info_text, (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 绘制偏差箭头
            if abs(deviation) > self.center_tolerance:
                arrow_start = (robot_x, height // 2 + 30)
                arrow_end = (self.field_center, height // 2 + 30)
                cv2.arrowedLine(result, arrow_start, arrow_end, (255, 255, 0), 3)

        return result


def test_enhanced_white_boundary():
    """
    测试增强的白色边界检测
    """
    try:
        # 初始化摄像头
        video_proxy = get_Proxy("ALVideoDevice", CONFIG["ip"])
        video_client = video_proxy.subscribeCamera(
            "enhanced_boundary", 0, CONFIG["resolution"],
            CONFIG["colorSpace"], CONFIG["fps"]
        )

        # 初始化检测器
        detector = EnhancedWhiteBoundaryDetector()

        print("增强白色边界检测启动，按ESC退出...")

        frame_count = 0
        while True:
            frame = get_image_from_camera(0, video_proxy, video_client)
            if frame is None:
                continue

            # 校准（前10帧）
            if frame_count < 10:
                detector.calibrate_boundaries_enhanced(frame)

            # 模拟机器人位置（实际中从传感器获取）
            robot_x = frame.shape[1] // 2 + int(np.sin(frame_count * 0.1) * 50)

            # 检测白线
            white_mask = detector.detect_white_lines_enhanced(frame)

            # 可视化结果
            result = detector.visualize_enhanced(frame, robot_x)

            # 显示白线检测
            cv2.imshow("White Lines", white_mask)
            cv2.imshow("Enhanced Boundary Detection", result)

            # 打印状态
            if detector.calibrated:
                deviation, status = detector.calculate_center_deviation(robot_x)
                correction, urgency = detector.get_centering_correction(robot_x)
                print(
                    "Frame {frame_count}: 偏差={deviation:+.0f}, 状态={status}, 修正强度={correction:.2f}, 紧急度={urgency}")

            if cv2.waitKey(30) == 27:
                break

            frame_count += 1

    except Exception as e:
        print("测试错误: {e}")
    finally:
        if 'video_proxy' in locals():
            video_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_enhanced_white_boundary()