# -*- coding: utf-8 -*-
"""
NAO + OpenCV 人脸学习与检测（Python 2.7 - 简化版）
功能：
 - 按 l: 学习新的人脸（从 NAO 摄像头），学习后输入名字并保存到pkl文件
 - 按 d: 检测已保存的人脸（20s 超时），检测到后说"你好"和对应名字
 - 按 q: 退出
依赖: numpy, cv2, naoqi
"""

from __future__ import print_function
import cv2
import os
import sys
import time
import pickle
import numpy as np

# NAO 配置
USE_NAO = True
NAO_IP = "192.168.43.58"  # 替换为你的NAO机器人IP
NAO_PORT = 9559

# 文件配置
DATA_DIR = "faces_data"
FACES_PKL = "faces_database.pkl"  # 保存人脸数据库

# 人脸检测器
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ---- NAO 摄像头连接 ----
use_nao = False
nao_proxy = None
try:
    if USE_NAO:
        from naoqi import ALProxy

        nao_proxy = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
        use_nao = True
        print("成功连接到 NAO 机器人")
except Exception as e:
    print("NAOqi 连接失败:", e)
    use_nao = False


# ---- 人脸数据库操作 ----
class FaceDatabase:
    def __init__(self):
        self.faces = {}  # {name: [face_image1, face_image2, ...]}
        self.load()

    def add_face(self, name, face_image):
        """添加人脸到数据库"""
        if name not in self.faces:
            self.faces[name] = []

        # 调整图像大小以便统一
        face_resized = cv2.resize(face_image, (100, 100))
        self.faces[name].append(face_resized)
        print("已添加人脸样本，{} 现在有 {} 个样本".format(name, len(self.faces[name])))

        # 立即保存
        self.save()
        return True

    def find_best_match(self, query_face, threshold=0.7):
        """查找最佳匹配的人脸"""
        if not self.faces:
            return None, 0.0

        query_resized = cv2.resize(query_face, (100, 100))
        best_name = None
        best_score = 0.0

        for name, face_images in self.faces.items():
            for stored_face in face_images:
                # 使用简单的模板匹配
                score = self.compare_faces(query_resized, stored_face)
                if score > best_score:
                    best_score = score
                    best_name = name

        # 如果相似度高于阈值，返回匹配结果
        if best_score >= threshold:
            return best_name, best_score
        else:
            return None, best_score

    def compare_faces(self, face1, face2):
        """比较两个人脸的相似度（0.0-1.0）"""
        # 方法1：直方图比较
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])

        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        # 使用相关系数
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, score)  # 确保非负

    def save(self):
        """保存人脸数据库到文件"""
        try:
            with open(FACES_PKL, 'wb') as f:
                pickle.dump(self.faces, f)
            print("人脸数据库已保存到:", FACES_PKL)
            return True
        except Exception as e:
            print("保存人脸数据库失败:", e)
            return False

    def load(self):
        """从文件加载人脸数据库"""
        try:
            if os.path.exists(FACES_PKL):
                with open(FACES_PKL, 'rb') as f:
                    self.faces = pickle.load(f)
                print("已加载人脸数据库，共 {} 个人".format(len(self.faces)))
                return True
        except Exception as e:
            print("加载人脸数据库失败:", e)
            self.faces = {}
        return False

    def list_faces(self):
        """列出所有已学习的人脸"""
        print("\n=== 已学习的人脸 ===")
        if not self.faces:
            print("暂无已学习的人脸")
            return

        for i, (name, images) in enumerate(self.faces.items()):
            print("{}. {} ({}个样本)".format(i + 1, name, len(images)))


# 初始化人脸数据库
face_db = FaceDatabase()


# ---- NAO 摄像头操作 ----
def nao_get_frame(subscriber):
    """从NAO摄像头获取一帧图像"""
    try:
        img = nao_proxy.getImageRemote(subscriber)
        if img is None:
            return None

        width = img[0]
        height = img[1]
        array = img[6]

        # 转换为numpy数组
        np_arr = np.frombuffer(array, dtype=np.uint8)
        frame = np_arr.reshape((height, width, 3))

        # 转换为BGR格式
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("获取NAO图像失败:", e)
        return None


def nao_subscribe(camera_index=0):
    """订阅NAO摄像头"""
    try:
        name = "face_recognition"
        # 参数说明：名字, 摄像头索引, 分辨率(2=640x480), 颜色空间(13=BGR), 帧率(15)
        subscriber = nao_proxy.subscribeCamera(name, camera_index, 2, 13, 15)
        return subscriber
    except Exception as e:
        print("订阅NAO摄像头失败:", e)
        return None


def nao_unsubscribe(subscriber):
    """取消订阅NAO摄像头"""
    try:
        nao_proxy.unsubscribe(subscriber)
    except:
        pass


# ---- NAO 语音功能 ----
def nao_say(text):
    """让NAO机器人说话"""
    try:
        tts = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)
        tts.setLanguage("Chinese")
        tts.say(text)
        print("NAO说: {}".format(text))
        return True
    except Exception as e:
        print("NAO语音失败:", e)
        return False


# ---- 人脸检测 ----
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def detect_faces(gray_img):
    """检测灰度图像中的人脸"""
    try:
        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces
    except Exception as e:
        print("人脸检测失败:", e)
        return []


# ---- 人脸捕获（学习模式） ----
def capture_face_for_learning():
    """捕获人脸用于学习"""
    print("\n=== 开始学习新人脸 ===")

    if not use_nao:
        print("无法连接NAO摄像头")
        return None

    subscriber = nao_subscribe()
    if subscriber is None:
        print("无法订阅NAO摄像头")
        return None

    print("请面对NAO摄像头，保持正脸...")
    nao_say("请面对我，保持正脸")

    captured_face = None
    start_time = time.time()
    timeout = 20

    while time.time() - start_time < timeout:
        frame = nao_get_frame(subscriber)
        if frame is None:
            time.sleep(0.1)
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detect_faces(gray)

        # 显示图像
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, "Learning: Face the camera",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("NAO Camera - Learning", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("用户取消")
            break

        if len(faces) > 0:
            # 选择最大的人脸
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = faces[0]

            # 裁剪人脸
            face_img = gray[y:y + h, x:x + w]

            # 显示捕获的人脸
            face_display = cv2.resize(face_img, (200, 200))
            cv2.imshow("Captured Face", face_display)

            # 等待用户确认
            print("检测到人脸，按空格键确认捕获，ESC取消")
            cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("NAO Camera - Learning", display_frame)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 32:  # 空格键
                    captured_face = face_img
                    print("人脸已捕获")
                    break
                elif key == 27:  # ESC
                    print("取消捕获")
                    break

            if captured_face is not None:
                break

    # 清理
    nao_unsubscribe(subscriber)
    cv2.destroyAllWindows()

    return captured_face


# ---- 学习流程 ----
def learn_face():
    """学习新人脸"""
    print("\n" + "=" * 50)
    print("开始学习新人脸")
    print("=" * 50)

    # 捕获人脸
    face_img = capture_face_for_learning()
    if face_img is None:
        print("未能捕获人脸")
        return

    # 询问姓名
    print("\n请输入这个人的名字:")
    sys.stdout.write("姓名: ")
    sys.stdout.flush()

    try:
        name = raw_input().strip()
    except:
        name = sys.stdin.readline().strip()

    if not name:
        print("姓名为空，取消保存")
        return

    # 添加到数据库
    if face_db.add_face(name, face_img):
        print("成功学习人脸: {}".format(name))
        nao_say("好的，我已经记住了 {}".format(name))

        # 询问是否添加更多样本
        print("\n是否添加更多样本以提高识别准确率？(y/n)")
        sys.stdout.write("选择: ")
        sys.stdout.flush()

        try:
            choice = raw_input().strip().lower()
        except:
            choice = sys.stdin.readline().strip().lower()

        if choice == 'y' or choice == 'yes':
            for i in range(2):  # 再添加2个样本
                print("\n请稍微改变姿势，准备捕获第{}个样本...".format(i + 2))
                time.sleep(2)

                extra_face = capture_face_for_learning()
                if extra_face is not None:
                    face_db.add_face(name, extra_face)
                    print("已添加第{}个样本".format(i + 2))
                else:
                    print("未能捕获额外样本")
                    break

    print("学习完成！")


# ---- 检测流程 ----
def detect_face():
    """检测人脸"""
    print("\n" + "=" * 50)
    print("开始人脸检测（20秒超时）")
    print("=" * 50)

    if not face_db.faces:
        print("尚未学习任何人脸，请先学习")
        nao_say("我还没有学习任何人脸")
        return

    print("已学习的人脸:")
    face_db.list_faces()

    if not use_nao:
        print("无法连接NAO摄像头")
        return

    subscriber = nao_subscribe()
    if subscriber is None:
        print("无法订阅NAO摄像头")
        return

    nao_say("开始人脸检测")

    start_time = time.time()
    timeout = 20
    last_detected_name = None
    last_detection_time = 0

    while time.time() - start_time < timeout:
        frame = nao_get_frame(subscriber)
        if frame is None:
            time.sleep(0.1)
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detect_faces(gray)

        # 处理每个人脸
        detected_in_frame = False
        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            # 裁剪人脸
            face_img = gray[y:y + h, x:x + w]

            # 查找匹配
            name, score = face_db.find_best_match(face_img, threshold=0.6)

            # 绘制结果
            if name:
                color = (0, 255, 0)  # 绿色：识别成功
                text = "{} ({:.1%})".format(name, score)
                detected_in_frame = True

                # 如果是新检测到的人脸，且距离上次说话超过3秒
                current_time = time.time()
                if name != last_detected_name or (current_time - last_detection_time > 3):
                    print("识别到: {} (相似度: {:.1%})".format(name, score))
                    nao_say("你好，{}".format(name))
                    last_detected_name = name
                    last_detection_time = current_time
            else:
                color = (0, 0, 255)  # 红色：未识别
                text = "Unknown"
                if score > 0:
                    text = "Unknown ({:.1%})".format(score)

            # 绘制边框和文字
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 显示剩余时间
        remaining = timeout - (time.time() - start_time)
        cv2.putText(display_frame, "Time: {:.1f}s".format(remaining),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow("NAO Camera - Detection", display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("用户取消检测")
            break

        # 如果检测到人脸，可以提前退出
        if detected_in_frame and (time.time() - last_detection_time > 2):
            # 可以选择继续检测或退出
            pass

    # 清理
    nao_unsubscribe(subscriber)
    cv2.destroyAllWindows()

    if last_detected_name:
        print("检测完成，识别到: {}".format(last_detected_name))
    else:
        print("检测完成，未识别到已知人脸")
        nao_say("没有识别到认识的人")


# ---- 主程序 ----
def main():
    """主程序"""
    print("\n" + "=" * 50)
    print("NAO 人脸识别系统")
    print("=" * 50)
    print("IP地址:", NAO_IP)
    print("NAO连接:", "成功" if use_nao else "失败")
    print("已学习人脸数:", len(face_db.faces))
    print("\n指令说明:")
    print("  l - 学习新人脸")
    print("  d - 检测人脸")
    print("  s - 显示已学习的人脸")
    print("  q - 退出程序")
    print("=" * 50)

    # 初始问候
    if use_nao:
        nao_say("人脸识别系统已启动")

    while True:
        print("\n请输入指令 (l/d/s/q):")
        sys.stdout.write("> ")
        sys.stdout.flush()

        try:
            command = raw_input().strip().lower()
        except:
            command = sys.stdin.readline().strip().lower()

        if command == 'l':
            learn_face()
        elif command == 'd':
            detect_face()
        elif command == 's':
            face_db.list_faces()
        elif command == 'q':
            print("退出程序...")
            if use_nao:
                nao_say("程序退出，再见")
            break
        else:
            print("无效指令，请输入 l, d, s 或 q")


# ---- 程序入口 ----
if __name__ == "__main__":
    print("NAO 人脸识别系统启动...")
    print("Python版本:", sys.version)

    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print("程序错误:", e)
        import traceback

        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("程序结束")