# -*- coding: utf-8 -*-
import sys
import time
import cv2
import numpy as np
from naoqi import ALProxy


class FaceRecognition:
    def __init__(self, robot_ip, robot_port=9559):
        self.robot_ip = robot_ip
        self.robot_port = robot_port

        # 初始化代理
        try:
            self.face_proxy = ALProxy("ALFaceDetection", robot_ip, robot_port)
            self.video_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
            self.tts_proxy = ALProxy("ALTextToSpeech", robot_ip, robot_port)
            self.motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
            self.posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)
            self.memory_proxy = ALProxy("ALMemory", robot_ip, robot_port)
            print("成功连接到机器人: {}:{}".format(robot_ip, robot_port))
        except Exception as e:
            print("连接机器人失败: {}".format(e))
            raise

        # 摄像头参数
        self.camera_name = "face_recognition"
        self.resolution = 2  # VGA 640x480
        self.color_space = 13  # BGR
        self.fps = 10

        # 程序状态
        self.current_mode = "IDLE"  # IDLE, LEARNING, DETECTING
        self.learning_name = ""
        self.is_waiting_for_name = False

    def setup_camera(self):
        """设置摄像头"""
        try:
            self.camera_subscriber = self.video_proxy.subscribeCamera(
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
                self.video_proxy.unsubscribe(self.camera_subscriber)
                print("摄像头资源已释放")
        except Exception as e:
            print("释放摄像头资源失败: {}".format(e))

    def get_image_cv2(self):
        """获取OpenCV格式的图像"""
        try:
            image_data = self.video_proxy.getImageRemote(self.camera_subscriber)

            if image_data and len(image_data) >= 6:
                width = image_data[0]
                height = image_data[1]
                image_array = image_data[6]

                image = np.frombuffer(image_array, dtype=np.uint8)
                image = image.reshape((height, width, 3))

                return image
            return None
        except Exception as e:
            print("获取图像失败: {}".format(e))
            return None

    def start_face_detection(self):
        """开始人脸检测"""
        try:
            # 启动人脸检测
            self.face_proxy.subscribe("FaceRecognition")
            print("人脸检测已启动")
            return True
        except Exception as e:
            print("启动人脸检测失败: {}".format(e))
            return False

    def stop_face_detection(self):
        """停止人脸检测"""
        try:
            self.face_proxy.unsubscribe("FaceRecognition")
            print("人脸检测已停止")
        except Exception as e:
            print("停止人脸检测失败: {}".format(e))

    def get_face_data(self):
        """获取人脸检测数据"""
        try:
            # 从内存中获取人脸数据
            face_data = self.memory_proxy.getData("FaceDetected")

            if face_data and isinstance(face_data, list) and len(face_data) >= 2:
                return face_data
            return None
        except Exception as e:
            # print("获取人脸数据失败: {}".format(e))
            return None

    def learn_face(self, face_name):
        """学习新的人脸"""
        try:
            print("开始学习人脸: {}".format(face_name))
            self.tts_proxy.say("请面对镜头，保持不动，我将学习你的面容")
            time.sleep(2)

            # 先清除可能存在的同名数据
            try:
                self.face_proxy.forgetPerson(face_name)
            except:
                pass

            # 使用ALFaceDetection的学习功能
            result = self.face_proxy.learnFace(face_name)

            if result:
                print("成功学习人脸: {}".format(face_name))
                self.tts_proxy.say("学习完成，已记住{}".format(face_name))
                return True
            else:
                print("学习人脸失败")
                self.tts_proxy.say("学习失败，请重试")
                return False

        except Exception as e:
            print("学习人脸失败: {}".format(e))
            self.tts_proxy.say("学习过程中出现错误")
            return False

    def recognize_faces(self):
        """识别人脸并返回结果"""
        try:
            # 获取人脸数据
            face_data = self.get_face_data()

            if not face_data:
                return []

            recognized_faces = []

            # 解析人脸数据
            # face_data结构: [timestamp, face_info_array]
            if len(face_data) >= 2 and face_data[1]:
                for face_info in face_data[1]:
                    # 每个人脸的信息
                    face_details = {}

                    # 提取人脸位置信息
                    if len(face_info) >= 2:
                        # 人脸在图像中的位置 [x, y, width, height]
                        face_details['position'] = face_info[0]

                        # 人脸特征信息
                        extra_info = face_info[1]
                        if len(extra_info) >= 3:
                            # 人脸标签（如果已学习）
                            face_label = extra_info[2]
                            if face_label:
                                face_details['name'] = face_label
                            else:
                                face_details['name'] = "未知"

                            # 置信度
                            face_details['confidence'] = extra_info[1]

                    recognized_faces.append(face_details)

            return recognized_faces

        except Exception as e:
            print("识别人脸失败: {}".format(e))
            return []

    def draw_face_info(self, image, faces):
        """在图像上绘制人脸信息和当前模式"""
        height, width = image.shape[:2]

        # 绘制当前模式信息
        mode_colors = {
            "IDLE": (255, 255, 0),  # 黄色
            "LEARNING": (0, 255, 255),  # 青色
            "DETECTING": (0, 255, 0)  # 绿色
        }
        color = mode_colors.get(self.current_mode, (255, 255, 255))

        # 显示当前模式
        mode_text = "当前模式: {}".format(self.current_mode)
        if self.current_mode == "LEARNING" and self.learning_name:
            mode_text = "当前模式: LEARNING - {}".format(self.learning_name)

        cv2.putText(image, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 绘制检测到的人脸
        for face in faces:
            if 'position' in face:
                pos = face['position']

                # 计算人脸矩形坐标
                x = int(pos[0] * width)
                y = int(pos[1] * height)
                w = int(pos[2] * width)
                h = int(pos[3] * height)

                # 确保坐标在图像范围内
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))

                # 绘制人脸矩形
                rect_color = (0, 255, 0) if face.get('name') != '未知' else (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 2)

                # 显示人脸信息
                name = face.get('name', '未知')
                confidence = face.get('confidence', 0)

                info_text = "{} ({:.1f}%)".format(name, confidence * 100)
                cv2.putText(image, info_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 1)

        # 显示操作指南
        y_offset = 60
        guides = []

        if self.current_mode == "IDLE":
            guides = [
                "l - 开始录入人脸",
                "d - 开始检测模式",
                "q - 退出程序"
            ]
        elif self.current_mode == "LEARNING":
            if self.is_waiting_for_name:
                guides = [
                    "请在终端输入人脸名称",
                    "然后按Enter键确认"
                ]
            else:
                guides = [
                    "按ESC取消录入",
                    "请保持面对镜头"
                ]
        elif self.current_mode == "DETECTING":
            guides = [
                "d - 停止检测",
                "q - 退出程序"
            ]

        for i, guide in enumerate(guides):
            cv2.putText(image, guide, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示检测统计
        if faces:
            known_count = sum(1 for face in faces if face.get('name') != '未知')
            unknown_count = len(faces) - known_count
            stats_text = "已识别: {} 未知: {}".format(known_count, unknown_count)
            cv2.putText(image, stats_text, (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def speak_recognition_result(self, faces):
        """语音报告识别结果"""
        if not faces:
            return

        try:
            known_faces = [face for face in faces if face.get('name') != '未知']
            unknown_faces = [face for face in faces if face.get('name') == '未知']

            if known_faces:
                names = [face['name'] for face in known_faces]
                if len(names) == 1:
                    message = "我看到了 {}".format(names[0])
                else:
                    message = "我看到了 {} 和 {}".format(", ".join(names[:-1]), names[-1])
                self.tts_proxy.say(message)
                print("机器人说: {}".format(message))

            if unknown_faces:
                message = "发现了 {} 个陌生人".format(len(unknown_faces))
                self.tts_proxy.say(message)
                print("机器人说: {}".format(message))

        except Exception as e:
            print("语音报告失败: {}".format(e))

    def prepare_for_vision(self):
        """准备视觉识别姿势"""
        try:
            # 唤醒机器人
            self.motion_proxy.wakeUp()

            # 进入站立姿势
            self.posture_proxy.goToPosture("StandInit", 0.5)

            # 调整头部姿势以便更好地观察
            joint_names = ["HeadPitch", "HeadYaw"]
            angles = [0.0, 0.0]  # 平视前方
            fraction_max_speed = 0.1
            self.motion_proxy.setAngles(joint_names, angles, fraction_max_speed)

            time.sleep(1)
            print("机器人已准备好进行视觉识别")

        except Exception as e:
            print("准备姿势失败: {}".format(e))

    def start_learning_mode(self):
        """开始学习模式"""
        print("进入学习模式...")
        self.current_mode = "LEARNING"
        self.is_waiting_for_name = True

        # 获取人脸名称
        print("请输入人脸名称: ")
        sys.stdout.flush()

        # 这里需要从标准输入获取名称
        # 注意：在OpenCV窗口中按键不会影响这里的输入
        # 我们需要使用单独的输入机制
        return True

    def handle_learning_mode(self):
        """处理学习模式逻辑"""
        if self.is_waiting_for_name:
            # 检查是否有可用的输入（需要使用非阻塞方式）
            # 由于OpenCV的窗口会占用主循环，这里简化处理
            # 在实际使用中，你可能需要多线程来处理输入
            return False

        # 正在进行学习
        return self.perform_learning()

    def perform_learning(self):
        """执行学习过程"""
        try:
            if not self.learning_name:
                return False

            print("正在学习人脸: {}".format(self.learning_name))
            self.tts_proxy.say("请面对镜头，我将学习你的面容")

            # 等待3秒让人准备好
            for i in range(3, 0, -1):
                print("倒计时: {} 秒".format(i))
                time.sleep(1)

            # 学习人脸
            result = self.face_proxy.learnFace(self.learning_name)

            if result:
                print("学习成功: {}".format(self.learning_name))
                self.tts_proxy.say("学习完成，已记住{}".format(self.learning_name))
                return True
            else:
                print("学习失败")
                self.tts_proxy.say("学习失败，请重试")
                return False

        except Exception as e:
            print("学习过程中出错: {}".format(e))
            return False

    def start_detection_mode(self):
        """开始检测模式"""
        print("进入检测模式...")
        self.current_mode = "DETECTING"
        self.tts_proxy.say("开始人脸检测模式")
        return True

    def run_face_recognition(self):
        """运行人脸识别主程序"""
        print("启动人脸识别系统...")
        print("=" * 60)
        print("初始模式: IDLE")
        print("操作指南:")
        print("  按 'l' 键 - 进入人脸录入模式")
        print("  按 'd' 键 - 进入人脸检测模式")
        print("  按 'q' 键 - 退出程序")
        print("=" * 60)

        try:
            # 设置摄像头
            if not self.setup_camera():
                return

            # 启动人脸检测
            if not self.start_face_detection():
                return

            # 准备机器人姿势
            self.prepare_for_vision()

            # 创建显示窗口
            cv2.namedWindow('NAO机器人人脸识别', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('NAO机器人人脸识别', 800, 600)

            print("系统就绪，等待按键输入...")

            last_face_count = 0
            last_speak_time = 0
            learning_start_time = None

            while True:
                # 获取图像
                image = self.get_image_cv2()

                if image is not None:
                    # 识别人脸
                    faces = []
                    if self.current_mode in ["DETECTING", "LEARNING"]:
                        faces = self.recognize_faces()

                    # 在图像上绘制信息
                    display_image = self.draw_face_info(image.copy(), faces)

                    # 显示图像
                    cv2.imshow('NAO机器人人脸识别', display_image)

                    # 在检测模式下，定期报告结果
                    if self.current_mode == "DETECTING":
                        current_time = time.time()
                        if (len(faces) != last_face_count and
                                current_time - last_speak_time > 5):
                            self.speak_recognition_result(faces)
                            last_face_count = len(faces)
                            last_speak_time = current_time

                    # 在学习模式下显示倒计时
                    if self.current_mode == "LEARNING" and not self.is_waiting_for_name:
                        if learning_start_time is None:
                            learning_start_time = time.time()

                        elapsed = time.time() - learning_start_time
                        if elapsed < 3:  # 3秒准备时间
                            countdown = 3 - int(elapsed)
                            print("准备学习，请面对镜头... {}秒".format(countdown))

                # 检查按键
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC键
                    if self.current_mode == "LEARNING":
                        print("取消学习模式")
                        self.current_mode = "IDLE"
                        self.is_waiting_for_name = False
                        self.learning_name = ""
                        self.tts_proxy.say("已取消学习")

                elif key == ord('q'):
                    print("用户请求退出")
                    break

                elif key == ord('l'):
                    if self.current_mode == "IDLE":
                        # 切换到学习模式
                        self.current_mode = "LEARNING"
                        self.is_waiting_for_name = True
                        print("进入学习模式")
                        print("请在终端输入人脸名称，然后按Enter键: ", )
                        sys.stdout.flush()

                        # 这里需要特殊处理输入
                        # 由于OpenCV窗口占用，我们创建一个临时的输入窗口
                        cv2.destroyAllWindows()
                        self.stop_face_detection()
                        self.release_camera()

                        # 获取名称输入
                        face_name = raw_input("\n请输入人脸名称: ").strip()

                        if face_name:
                            # 重新设置摄像头
                            self.setup_camera()
                            self.start_face_detection()
                            cv2.namedWindow('NAO机器人人脸识别', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('NAO机器人人脸识别', 800, 600)

                            self.learning_name = face_name
                            self.is_waiting_for_name = False
                            print("开始学习人脸: {}".format(face_name))

                            # 执行学习
                            learning_success = self.learn_face(face_name)

                            # 学习完成后返回空闲模式
                            self.current_mode = "IDLE"
                            self.learning_name = ""

                            if learning_success:
                                print("学习完成，返回空闲模式")
                                self.tts_proxy.say("学习完成，返回空闲模式")
                            else:
                                print("学习失败，返回空闲模式")
                                self.tts_proxy.say("学习失败，返回空闲模式")
                        else:
                            # 重新设置摄像头
                            self.setup_camera()
                            self.start_face_detection()
                            cv2.namedWindow('NAO机器人人脸识别', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('NAO机器人人脸识别', 800, 600)
                            self.current_mode = "IDLE"
                            print("取消学习，返回空闲模式")

                elif key == ord('d'):
                    if self.current_mode == "IDLE":
                        # 开始检测模式
                        self.start_detection_mode()
                    elif self.current_mode == "DETECTING":
                        # 停止检测模式
                        print("停止检测模式")
                        self.current_mode = "IDLE"
                        self.tts_proxy.say("退出检测模式")

                elif key == ord('c'):  # 清除数据
                    if self.current_mode == "IDLE":
                        try:
                            self.face_proxy.clearDatabase()
                            print("已清除所有人脸数据")
                            self.tts_proxy.say("已清除所有人脸数据")
                        except Exception as e:
                            print("清除数据失败: {}".format(e))

        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print("程序运行出错: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            # 释放资源
            cv2.destroyAllWindows()
            self.stop_face_detection()
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
    print("          NAO机器人人脸识别系统")
    print("=" * 60)
    print("机器人IP: {}".format(ROBOT_IP))
    print("系统说明:")
    print("1. 初始为IDLE模式")
    print("2. 按'l'进入学习模式，输入名称后自动学习")
    print("3. 按'd'进入检测模式，实时识别人脸")
    print("4. 在检测模式下按'd'返回IDLE模式")
    print("5. 按'q'退出程序")
    print("=" * 60)

    try:
        # 创建人脸识别实例
        face_detector = FaceRecognition(ROBOT_IP, ROBOT_PORT)

        # 运行人脸识别
        face_detector.run_face_recognition()

    except Exception as e:
        print("程序运行失败: {}".format(e))

    print("程序结束")


if __name__ == "__main__":
    main()