# -*- coding: utf-8 -*-
import sys
import time
import cv2
import numpy as np
from naoqi import ALProxy


class ColorRecognitionWithDisplay:
    def __init__(self, robot_ip, robot_port=9559):
        self.robot_ip = robot_ip
        self.robot_port = robot_port

        # 初始化代理
        try:
            self.camera_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
            self.tts_proxy = ALProxy("ALTextToSpeech", robot_ip, robot_port)
            self.motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
            self.posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)
            self.audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)
            print("成功连接到机器人: {}:{}".format(robot_ip, robot_port))

            # 设置语音参数
            self.setup_voice()

        except Exception as e:
            print("连接机器人失败: {}".format(e))
            raise

        # 摄像头参数
        self.camera_name = "color_recognition_display"
        self.resolution = 2  # VGA 640x480
        self.color_space = 13  # BGR色彩空间 (OpenCV使用)
        self.fps = 10

        # 颜色识别状态
        self.last_color = ""
        self.speech_enabled = True

    def setup_voice(self):
        """设置语音参数"""
        try:
            # 设置音量 (0-100)
            self.audio_device.setOutputVolume(20)
            # 设置语音语言为中文
            self.tts_proxy.setLanguage("Chinese")
            # 设置语音速度 (50-400，默认100)
            self.tts_proxy.setParameter("speed", 95)
            print("语音模块已初始化")
        except Exception as e:
            print("语音设置失败: {}".format(e))

    def setup_camera(self):
        """设置摄像头"""
        try:
            self.camera_subscriber = self.camera_proxy.subscribeCamera(
                self.camera_name, 0, self.resolution, self.color_space, self.fps
            )
            print("摄像头已启动")
            return True
        except Exception as e:
            print("摄像头设置失败: {}".format(e))
            return False

    def release_camera(self):
        """释放摄像头资源"""
        try:
            if hasattr(self, 'camera_subscriber'):
                self.camera_proxy.unsubscribe(self.camera_subscriber)
                print("摄像头资源已释放")
        except Exception as e:
            print("释放摄像头资源失败: {}".format(e))

    def get_image_cv2(self):
        """获取OpenCV格式的图像"""
        try:
            # 获取图像数据
            image_data = self.camera_proxy.getImageRemote(self.camera_subscriber)

            if image_data and len(image_data) >= 6:
                width = image_data[0]
                height = image_data[1]
                image_array = image_data[6]  # 图像数据

                # 将图像数据转换为numpy数组
                image = np.frombuffer(image_array, dtype=np.uint8)
                image = image.reshape((height, width, 3))

                return image
            return None
        except Exception as e:
            print("获取图像失败: {}".format(e))
            return None

    def recognize_color_from_image(self, image):
        """从图像中识别颜色"""
        if image is None:
            return "未知", (0, 0, 0)

        try:
            height, width = image.shape[:2]

            # 定义中心检测区域 (占画面的20%)
            region_size = min(width, height) // 5
            center_x = width // 2
            center_y = height // 2

            start_x = max(0, center_x - region_size // 2)
            end_x = min(width, center_x + region_size // 2)
            start_y = max(0, center_y - region_size // 2)
            end_y = min(height, center_y + region_size // 2)

            # 提取中心区域
            center_region = image[start_y:end_y, start_x:end_x]

            # 计算平均BGR值
            avg_b = np.mean(center_region[:, :, 0])
            avg_g = np.mean(center_region[:, :, 1])
            avg_r = np.mean(center_region[:, :, 2])

            avg_color = (avg_b, avg_g, avg_r)

            # 转换为颜色名称
            color_name = self.bgr_to_color_name(avg_b, avg_g, avg_r)

            return color_name, (int(avg_b), int(avg_g), int(avg_r))

        except Exception as e:
            print("颜色识别失败: {}".format(e))
            return "未知", (0, 0, 0)

    def bgr_to_color_name(self, b, g, r):
        """将BGR值转换为颜色名称"""
        # 计算HSV值用于更好的颜色识别
        bgr_pixel = np.uint8([[[b, g, r]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_pixel

        # 颜色识别逻辑
        if v < 30:  # 很暗
            return "黑色"
        elif v > 200 and s < 30:  # 很亮且饱和度低
            return "白色"
        elif s < 50 and v > 150:  # 低饱和度，高亮度
            return "灰色"
        elif s < 50:  # 低饱和度
            return "灰色"

        # 基于色相的颜色识别
        if h < 10 or h > 170:  # 红色区域
            if v > 150:
                return "红色"
            else:
                return "暗红色"
        elif 10 <= h < 25:  # 橙色
            return "橙色"
        elif 25 <= h < 35:  # 黄色
            return "黄色"
        elif 35 <= h < 85:  # 绿色
            return "绿色"
        elif 85 <= h < 110:  # 青色
            return "青色"
        elif 110 <= h < 130:  # 蓝色
            return "蓝色"
        elif 130 <= h < 170:  # 紫色
            return "紫色"
        else:
            return "混合色"

    def speak_color(self, color_name):
        """说出颜色名称"""
        if not self.speech_enabled:
            return

        try:
            message = "这是{}".format(color_name)
            print("机器人说: {}".format(message))

            # 说出颜色（不停止之前的语音，允许重叠）
            self.tts_proxy.say(message)

        except Exception as e:
            print("语音合成失败: {}".format(e))

    def test_voice(self):
        """测试语音功能"""
        try:
            print("测试语音功能...")
            self.tts_proxy.say("")
            print("语音测试完成")
            return True
        except Exception as e:
            print("语音测试失败: {}".format(e))
            return False

    def prepare_for_vision(self):
        """准备视觉识别姿势"""
        try:
            # 唤醒机器人
            self.motion_proxy.wakeUp()

            # 进入站立姿势
            self.posture_proxy.goToPosture("StandInit", 0.5)

            # 调整头部姿势以便更好地观察
            joint_names = ["HeadPitch"]
            angles = [0.1]  # 稍微低头
            fraction_max_speed = 0.1
            self.motion_proxy.setAngles(joint_names, angles, fraction_max_speed)

            time.sleep(1)
            print("机器人已准备好进行颜色识别")

        except Exception as e:
            print("准备姿势失败: {}".format(e))

    def draw_detection_info(self, image, color_name, avg_color):
        """在图像上绘制检测信息"""
        height, width = image.shape[:2]

        # 绘制中心检测区域
        region_size = min(width, height) // 5
        center_x = width // 2
        center_y = height // 2

        cv2.rectangle(image,
                      (center_x - region_size // 2, center_y - region_size // 2),
                      (center_x + region_size // 2, center_y + region_size // 2),
                      (255, 255, 255), 2)

        # 绘制十字准星
        cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)

        # 显示颜色信息
        info_text = "颜色: {}".format(color_name)
        rgb_text = "RGB: ({}, {}, {})".format(avg_color[2], avg_color[1], avg_color[0])
        speech_status = "语音: {}".format("开启" if self.speech_enabled else "关闭")

        # 根据颜色设置文本颜色
        if color_name == "黑色":
            text_color = (255, 255, 255)  # 白色
        else:
            text_color = (0, 0, 0)  # 黑色

        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(image, rgb_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(image, speech_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # 显示检测区域的平均颜色
        color_display_size = 50
        color_block = np.full((color_display_size, color_display_size, 3), avg_color, dtype=np.uint8)
        image[10:10 + color_display_size, width - color_display_size - 10:width - 10] = color_block

        # 显示操作提示
        cv2.putText(image, "按 'q' 退出", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(image, "按 's' 切换语音", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        return image

    def run_color_recognition_with_display(self):
        """运行带显示的实时颜色识别"""
        print("启动实时颜色识别...")
        print("请将彩色物体放在机器人正前方的十字准星位置")
        print("按 'q' 键退出程序")

        # 语音测试
        print("进行语音测试...")
        if not self.test_voice():
            print("警告: 语音功能可能有问题")
            self.speech_enabled = False
        else:
            print("语音功能正常")

        try:
            # 设置摄像头
            if not self.setup_camera():
                return

            # 准备姿势
            self.prepare_for_vision()

            # 创建显示窗口
            cv2.namedWindow('NAO机器人颜色识别', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('NAO机器人颜色识别', 800, 600)

            print("开始颜色识别...")

            while True:
                # 获取图像
                image = self.get_image_cv2()

                if image is not None:
                    # 识别颜色
                    color_name, avg_color = self.recognize_color_from_image(image)

                    # 在图像上绘制信息
                    display_image = self.draw_detection_info(image.copy(), color_name, avg_color)

                    # 显示图像
                    cv2.imshow('NAO机器人颜色识别', display_image)

                    # 如果颜色发生变化且不是未知颜色，立即说出来
                    if (color_name != self.last_color and
                            color_name != "未知" and
                            color_name != "混合色" and
                            color_name != "灰色"):

                        if self.speech_enabled:
                            self.speak_color(color_name)

                        self.last_color = color_name

                        print("识别到颜色: {} - RGB: ({}, {}, {})".format(
                            color_name, avg_color[2], avg_color[1], avg_color[0]))

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # 空格键手动识别
                    if color_name != "未知" and color_name != "混合色":
                        self.speak_color(color_name)
                elif key == ord('s'):  # s键切换语音开关
                    self.speech_enabled = not self.speech_enabled
                    status = "开启" if self.speech_enabled else "关闭"
                    print("语音功能已{}".format(status))
                    if self.speech_enabled:
                        self.tts_proxy.say("")

            print("颜色识别结束")

        except KeyboardInterrupt:
            print("\n用户中断颜色识别")
        except Exception as e:
            print("颜色识别过程出错: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            # 释放资源
            cv2.destroyAllWindows()
            self.release_camera()
            # 让机器人休息
            try:
                self.motion_proxy.rest()
            except:
                pass


def main():
    # 机器人配置
    ROBOT_IP = "192.168.43.58"  # 替换为你的机器人IP
    ROBOT_PORT = 9559

    if len(sys.argv) > 1:
        ROBOT_IP = sys.argv[1]

    print("=" * 60)
    print("          NAO机器人实时颜色识别系统")
    print("=" * 60)
    print("机器人IP: {}".format(ROBOT_IP))
    print("使用说明:")
    print("1. 将彩色物体放在十字准星位置")
    print("2. 确保环境光线充足")
    print("3. 机器人识别到颜色后会立即说出来")
    print("4. 按 'q' 键退出程序")
    print("5. 按空格键手动让机器人说出当前颜色")
    print("6. 按 's' 键切换语音开关")
    print("=" * 60)

    try:
        # 创建颜色识别实例
        color_detector = ColorRecognitionWithDisplay(ROBOT_IP, ROBOT_PORT)

        # 运行带显示的实时颜色识别
        color_detector.run_color_recognition_with_display()

    except Exception as e:
        print("程序运行失败: {}".format(e))

    print("程序结束")


if __name__ == "__main__":
    main()