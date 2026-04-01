
# -*- coding: utf-8 -*-
"""
融合B-Human高级颜色检测的圆柱体识别系统（NAO使用）
"""

import qi
import cv2
import numpy as np
import time
import math
from detect_conf import BHumanAdvancedConfig  # 高级颜色识别模块


class CylinderDetector:
    def __init__(self, nao_ip="127.0.0.1", nao_port=9559):
        self.nao_ip = nao_ip
        self.nao_port = nao_port

        # NAOqi连接初始化
        self.session = qi.Session()
        self.session.connect("tcp://{}:{}".format(nao_ip, nao_port))

        # NAO服务初始化
        self.video_service = self.session.service("ALVideoDevice")
        self.motion_service = self.session.service("ALMotion")
        self.posture_service = self.session.service("ALRobotPosture")

        # 相机参数
        self.camera_id = 0  # 顶置摄像头
        self.resolution = 2  # VGA
        self.color_space = 11  # RGB
        self.fps = 30

        # 检测参数
        self.focal_length = 525.0  # 像素焦距（估值）
        self.camera_height = 0.48  # 相机高（单位：米）
        self.min_contour_area = 500
        self.max_contour_area = 50000
        self.min_circularity = 0.3

        # 使用高级颜色识别模块
        self.color_detector = BHumanAdvancedConfig()

        self.video_client = None
        self.is_running = False

    def initialize_camera(self):
        try:
            self.video_client = self.video_service.subscribe(
                "advanced_cylinder_detector", self.resolution, self.color_space, self.fps
            )
            print("[INFO] 相机初始化成功")
            return True
        except Exception as e:
            print("[ERROR] 相机初始化失败: {}".format(e))
            return False

    def get_image(self):
        try:
            img = self.video_service.getImageRemote(self.video_client)
            width, height, array = img[0], img[1], img[6]
            image = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
            return image
        except Exception as e:
            print("[ERROR] 图像获取失败: {}".format(e))
            return None

    def estimate_distance(self, contour):
        _, _, _, h = cv2.boundingRect(contour)
        real_height = 0.30  # 假设障碍物高度为30cm
        if h > 0:
            return (real_height * self.focal_length) / h
        return float('inf')

    def detect_cylinders(self, image):
        """主检测逻辑"""
        detections = []
        height, width = image.shape[:2]

        # 应用自适应光照增强
        image = self.color_detector.adaptive_lighting_correction(image)

        for color in ['red', 'yellow', 'blue']:
            mask = self.color_detector.multi_space_color_detection(image, color)

            # 轮廓提取
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area < area < self.max_contour_area:
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    if circularity < self.min_circularity:
                        continue

                    # 计算中心
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = self.estimate_distance(contour)

                    detections.append({
                        'color': color,
                        'center': (cx, cy),
                        'distance': dist,
                        'contour': contour,
                        'area': area,
                        'circularity': circularity
                    })

        return detections

    def draw_result(self, image, detection):
        """在图像上绘制检测结果"""
        if detection is None:
            return image

        annotated = image.copy()
        cx, cy = detection['center']
        color = detection['color']
        contour = detection['contour']

        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0)
        }
        bgr = color_map.get(color, (255, 255, 255))

        cv2.drawContours(annotated, [contour], -1, bgr, 2)
        cv2.circle(annotated, (cx, cy), 5, bgr, -1)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr, 2)

        text = "{} | {:.2f}m".format(color.upper(), detection['distance'])
        cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        return annotated

    def run_detection_loop(self):
        if not self.initialize_camera():
            return

        print("[INFO] 启动检测循环，按 'q' 退出")
        self.is_running = True

        try:
            while self.is_running:
                image = self.get_image()
                if image is None:
                    continue

                detections = self.detect_cylinders(image)
                nearest = min(detections, key=lambda d: d['distance'], default=None)

                annotated = self.draw_result(image, nearest)
                cv2.imshow("Cylinder Detection", annotated)

                if nearest:
                    print("识别圆柱体: {} - {:.2f}m".format(nearest['color'], nearest['distance']))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.033)

        except KeyboardInterrupt:
            print("[INFO] 用户中断检测")

        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        if self.video_client:
            try:
                self.video_service.unsubscribe(self.video_client)
            except:
                pass
        cv2.destroyAllWindows()
        print("[INFO] 资源清理完毕")


def main():
    nao_ip = "192.168.43.48"  # 修改为你的 NAO IP
    detector = CylinderDetector(nao_ip)
    detector.run_detection_loop()


if __name__ == "__main__":
    main()
