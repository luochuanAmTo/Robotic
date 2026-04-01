# -*- coding: utf-8 -*-
"""
NAO + OpenCV 人脸学习与检测（基于特征匹配的简化版）
功能：
 - 按 l: 学习新的人脸
 - 按 d: 检测已保存的人脸（20s 超时）
 - 按 q: 退出
"""

from __future__ import print_function
import cv2
import os
import sys
import time
import pickle
import numpy as np


USE_NAO = True
NAO_IP = "192.168.43.247"
NAO_PORT = 9559

# 文件/目录
DATA_DIR = "faces"
MODEL_FILE = "face_features.pkl"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 全局变量
face_features = {}  # name -> list of feature vectors
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# ---- NAO 摄像头支持 ----
use_nao = False
nao_proxy = None
try:
    if USE_NAO:
        from naoqi import ALProxy

        nao_proxy = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
        use_nao = True
except Exception as e:
    print("NAOqi import/connection failed (will use local webcam). Error:", e)
    use_nao = False


# 加载保存的特征
def load_features():
    global face_features
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            face_features = pickle.load(f)
        print("加载了 %d 个人的特征" % len(face_features))


def save_features():
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(face_features, f)
    print("特征已保存到", MODEL_FILE)


# 提取人脸特征（简化：使用HOG特征）
def extract_face_features(face_img):
    """从人脸图像中提取特征"""
    # 转换为灰度（如果还不是）
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # 调整大小
    gray = cv2.resize(gray, (100, 100))

    # 提取HOG特征（简化版）
    # 这里使用简单的直方图特征
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


# 计算特征相似度
def compare_features(feat1, feat2):
    """计算两个特征的相似度（距离越小越相似）"""
    # 使用欧氏距离
    return np.linalg.norm(feat1 - feat2)


# ---- 图像获取函数 ----
def nao_subscribe(camera_index=0):
    try:
        subscriber = nao_proxy.subscribeCamera("python_cv", camera_index, 2, 13, 15)
        return subscriber
    except Exception as e:
        print("NAO订阅失败:", e)
        return None


def nao_unsubscribe(subscriber):
    try:
        nao_proxy.unsubscribe(subscriber)
    except:
        pass


def nao_get_frame(subscriber):
    try:
        img = nao_proxy.getImageRemote(subscriber)
        if img is None:
            return None
        width = img[0]
        height = img[1]
        array = img[6]
        np_arr = np.frombuffer(array, dtype=np.uint8)
        frame = np_arr.reshape((height, width, 3))
        return frame
    except Exception as e:
        print("获取NAO图像失败:", e)
        return None


# ---- 人脸捕获 ----
def capture_face_from_camera(timeout=30):
    """捕获人脸图像"""
    start = time.time()
    consecutive = 0
    captured_face = None

    cap = None
    subscriber = None

    if use_nao:
        subscriber = nao_subscribe(0)
        if not subscriber:
            print("无法使用NAO摄像头，尝试本地摄像头")
            use_nao_local = False

    if not use_nao or not subscriber:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开本地摄像头")
            return None

    print("开始捕获人脸 - 超时 %ds。请将人脸对准摄像头..." % timeout)

    while True:
        if use_nao and subscriber:
            frame = nao_get_frame(subscriber)
        else:
            ret, frame = cap.read()
            if not ret:
                frame = None

        if frame is None:
            time.sleep(0.1)
            if time.time() - start > timeout:
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        # 显示检测框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture - Press ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("用户取消捕获")
            break

        if len(faces) > 0:
            x, y, w, h = faces[0]  # 取最大的脸
            face_img = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (200, 200))
            consecutive += 1

            cv2.imshow("Captured Face", face_resized)
            if consecutive >= 5:
                captured_face = face_resized.copy()
                print("人脸捕获成功")
                break
        else:
            consecutive = 0

        if time.time() - start > timeout:
            print("捕获超时")
            break

    # 清理
    if use_nao and subscriber:
        nao_unsubscribe(subscriber)
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    return captured_face


# ---- 学习流程 ----
def learn_flow():
    print("进入学习流程...")
    face_img = capture_face_from_camera(timeout=30)
    if face_img is None:
        print("未获取到人脸样本")
        return

    # 输入姓名
    sys.stdout.write("请输入该人脸对应的姓名（回车确认）：")
    sys.stdout.flush()
    try:
        name = raw_input().strip()
    except:
        name = sys.stdin.readline().strip()

    if not name:
        print("姓名为空，取消保存")
        return

    # 保存人脸图片
    person_dir = os.path.join(DATA_DIR, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    ts = int(time.time())
    fname = os.path.join(person_dir, "%d.png" % ts)
    cv2.imwrite(fname, face_img)
    print("保存人脸样本到", fname)

    # 提取特征并保存
    features = extract_face_features(face_img)

    if name not in face_features:
        face_features[name] = []

    face_features[name].append(features)
    save_features()
    print("特征已保存，当前有 %d 个人的特征" % len(face_features))


# ---- 检测流程 ----
def detect_flow(timeout=20):
    print("进入检测流程（%d 秒超时）。按 ESC 可取消。" % timeout)
    start = time.time()

    load_features()  # 加载特征

    if not face_features:
        print("没有已保存的特征，请先学习一些人脸")
        return

    # 获取摄像头
    cap = None
    subscriber = None

    if use_nao:
        subscriber = nao_subscribe(0)

    if not use_nao or not subscriber:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

    print("开始检测...")

    while time.time() - start < timeout:
        # 获取帧
        if use_nao and subscriber:
            frame = nao_get_frame(subscriber)
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (100, 100))

            # 提取当前人脸特征
            current_feat = extract_face_features(face_resized)

            # 与保存的特征比较
            best_match = None
            best_distance = float('inf')

            for name, feat_list in face_features.items():
                for saved_feat in feat_list:
                    distance = compare_features(current_feat, saved_feat)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name

            # 显示结果
            threshold = 0.8  # 阈值，根据实际情况调整
            if best_match and best_distance < threshold:
                text = "%s (%.2f)" % (best_match, best_distance)
                color = (0, 255, 0)
                print("检测到: %s, 距离: %.2f" % (best_match, best_distance))
            else:
                text = "Unknown (%.2f)" % best_distance if best_distance < float('inf') else "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("用户取消检测")
            break

    # 清理
    if use_nao and subscriber:
        nao_unsubscribe(subscriber)
    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ---- 主循环 ----
def main():
    load_features()
    print("\n===== 人脸学习与检测程序 =====")
    print("按键说明：")
    print("  l : 学习新的人脸")
    print("  d : 检测已保存的人脸")
    print("  q : 退出程序\n")

    while True:
        sys.stdout.write("等待按键 (l/d/q): ")
        sys.stdout.flush()
        try:
            key = raw_input().strip().lower()
        except:
            key = sys.stdin.readline().strip().lower()

        if key == 'l':
            learn_flow()
        elif key == 'd':
            detect_flow()
        elif key == 'q':
            print("退出程序")
            break
        else:
            print("未知按键，请按 l/d/q")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        cv2.destroyAllWindows()