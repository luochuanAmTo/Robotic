# -*- coding: utf-8 -*-
"""
NAO + OpenCV 人脸学习与检测（Python 2.7）
功能：
 - 按 l: 学习新的人脸（从 NAO 摄像头或本地摄像头），学习后输入名字并保存
 - 按 d: 检测已保存的人脸（20s 超时）
 - 按 q: 退出
依赖: numpy, cv2, naoqi (可选)
"""

from __future__ import print_function
import cv2
import os
import sys
import time
import pickle
import numpy as np

# 如果要使用 NAO 摄像头，确保 naoqi 已安装并可 import
USE_NAO = True             # 默认尝试使用 NAO 摄像头；若没有 NAO 请设为 False
NAO_IP = "192.168.43.58"   # <-- 把这里换成你的 NAO 机器人 IP
NAO_PORT = 9559

# 文件/目录
DATA_DIR = "faces"         # 每个姓名为子目录，保存采集的灰度人脸图像
MODEL_FILE = "trainer.yml" # LBPH 模型保存
LABELS_FILE = "labels.pickle"

# 人脸检测 Haar 模型（OpenCV 自带 xml 文件路径）
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 识别阈值（较低值表示更严格；LBPH 的距离越小越相似）
RECOG_CONF_THRESHOLD = 70  # 根据实际调整

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ---- NAO 摄像头支持 ----
use_nao = False
nao_proxy = None
video_client = None
try:
    if USE_NAO:
        from naoqi import ALProxy
        nao_proxy = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
        use_nao = True
except Exception as e:
    print("NAOqi import/connection failed (will use local webcam). Error:", e)
    use_nao = False

# Helper to create LBPH recognizer in a way compatible with multiple OpenCV versions
# def create_lbph_recognizer():
#     # Try different factories depending on OpenCV version
#     try:
#         # newer style: cv2.face
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         return recognizer
#     except Exception:
#         try:
#             recognizer = cv2.createLBPHFaceRecognizer()  # older OpenCV 2/3 style (may not exist)
#             return recognizer
#         except Exception:
#             try:
#                 recognizer = cv2.face.createLBPHFaceRecognizer()
#                 return recognizer
#             except Exception as e:
#                 print("无法创建 LBPH 识别器，请确认 OpenCV 安装包含 face 模块。错误:", e)
#                 return None


# 替换原来的 create_lbph_recognizer 函数
def create_lbph_recognizer():
    try:
        # 尝试各种可能的导入方式
        import cv2

        # 方法1: OpenCV 3/4 的 face 模块
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("使用 cv2.face.LBPHFaceRecognizer_create()")
            return recognizer
        except AttributeError:
            pass

        # 方法2: OpenCV 2/3 的旧方法
        try:
            recognizer = cv2.createLBPHFaceRecognizer()
            print("使用 cv2.createLBPHFaceRecognizer()")
            return recognizer
        except AttributeError:
            pass

        # 方法3: 直接使用opencv-contrib中的face模块
        try:
            import cv2.face
            recognizer = cv2.face.createLBPHFaceRecognizer()
            print("使用 cv2.face.createLBPHFaceRecognizer()")
            return recognizer
        except Exception:
            pass

        print("警告：无法创建LBPH识别器，将使用简单的基于特征匹配的方法")
        return None
    except Exception as e:
        print("无法创建LBPH识别器，错误:", e)
        return None

# 加载/初始化识别器与标签映射
recognizer = create_lbph_recognizer()
labels = {}  # name -> id
labels_inv = {}  # id -> name

def load_labels():
    global labels, labels_inv
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'rb') as f:
            labels = pickle.load(f)
        labels_inv = {v: k for k, v in labels.items()}
        print("加载标签：", labels)
    else:
        labels = {}
        labels_inv = {}

def save_labels():
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(labels, f)
    print("已保存标签到", LABELS_FILE)

def train_recognizer():
    """
    从 DATA_DIR 中读取已保存的灰度人脸图像（每个人一个目录），训练 LBPH 并保存模型
    """
    if recognizer is None:
        print("识别器不可用，跳过训练。")
        return

    image_paths = []
    labels_list = []
    current_id = 0
    # ensure labels mapping consistent with labels dict
    load_labels()
    label_id_map = dict(labels)  # copy

    for person_name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        # ensure label id exists
        if person_name not in label_id_map:
            # assign new id
            new_id = max(label_id_map.values()) + 1 if label_id_map else 0
            label_id_map[person_name] = new_id
        this_id = label_id_map[person_name]
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith(".png") and not fname.lower().endswith(".jpg"):
                continue
            path = os.path.join(person_dir, fname)
            # read grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            image_paths.append(img)
            labels_list.append(this_id)

    if not image_paths:
        print("没有人脸图像用于训练。")
        return

    # train
    print("开始训练识别器，样本数:", len(image_paths))
    try:
        recognizer.train(image_paths, np.array(labels_list))
    except Exception as e:
        # Some older versions expect list of numpy arrays only
        recognizer.train(image_paths, np.asarray(labels_list))
    # save
    try:
        recognizer.save(MODEL_FILE)
        print("训练完成并保存到", MODEL_FILE)
    except Exception as e:
        print("训练完成但保存模型失败:", e)

    # update global labels
    global labels, labels_inv
    labels = label_id_map
    labels_inv = {v: k for k, v in labels.items()}
    save_labels()

def load_trained_model():
    if recognizer is None:
        return
    if os.path.exists(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            print("已加载训练模型", MODEL_FILE)
            load_labels()
        except Exception as e:
            print("加载模型失败:", e)

# ---- 图像获取：从 NAO 或 本地摄像头 ----

def nao_get_frame(subscriber):
    """
    使用 NAO ALVideoDevice 获取一帧图像并转换为 OpenCV BGR 格式
    subscriber: subscription id returned by subscribeCamera/subscribe
    """
    try:
        img = nao_proxy.getImageRemote(subscriber)
        if img is None:
            return None
        # img[6] contains raw image byte array
        width = img[0]
        height = img[1]
        array = img[6]
        # convert to numpy array
        np_arr = np.frombuffer(array, dtype=np.uint8)
        # NAO colorSpace is typically 13 (kBGRColorSpace) or 11 (kRGBColorSpace) depending on request
        # Here we asked for kBGRColorSpace to get BGR raw bytes
        try:
            frame = np_arr.reshape((height, width, 3))
        except Exception:
            # fallback if shape mismatch
            frame = np.copy(np_arr)
            frame = frame.reshape((height, width, 3))
        return frame
    except Exception as e:
        print("从 NAO 获取图像失败:", e)
        return None

def nao_subscribe(camera_index=0, resolution=2, color_space=13, fps=15):
    """
    camera_index: 0 front, 1 bottom (NAO convention may vary)
    resolution: 2 -> 640x480 typically
    color_space: 13 -> kBGRColorSpace (so we can directly use OpenCV)
    """
    try:
        name = "python_cv_sub"
        # subscribeCamera(cameraName, cameraIndex, resolution, colorSpace, fps)
        subscriber = nao_proxy.subscribeCamera(name, camera_index, resolution, color_space, fps)
        return subscriber
    except Exception as e:
        print("NAO subscribeCamera 失败:", e)
        return None

def nao_unsubscribe(subscriber):
    try:
        nao_proxy.unsubscribe(subscriber)
    except Exception as e:
        # ignore
        pass

# ---- 人脸捕获（用于学习） ----

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def detect_faces_gray(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    return faces

def capture_face_from_camera(timeout=30, use_nao_cam=True):
    """
    打开（NAO或本地）摄像头流，显示画面并尝试检测一个稳定的人脸。
    当检测到人脸并稳定（例如连续N帧）时，捕获并返回灰度人脸图像（裁切并缩放到统一大小）。
    返回 (face_image_gray, frame_for_preview) 或 (None, None) 若超时/失败
    """
    start = time.time()
    consecutive_required = 5
    consecutive = 0
    captured_face = None

    cap = None
    subscriber = None
    use_nao_cam = use_nao_cam and use_nao

    if use_nao_cam:
        subscriber = nao_subscribe(camera_index=0)
        if subscriber is None:
            print("无法订阅 NAO 摄像头，改用本地摄像头。")
            use_nao_cam = False

    if not use_nao_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开本地摄像头。")
            return None, None

    print("开始捕获人脸 - 超时 %ds。请将人脸对准摄像头..." % timeout)
    while True:
        if use_nao_cam:
            frame = nao_get_frame(subscriber)
            if frame is None:
                time.sleep(0.1)
                if time.time() - start > timeout:
                    break
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                if time.time() - start > timeout:
                    break
                continue

        # flip if needed (NAO bottom or orientation issues) - 忽略
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_gray(gray)
        # draw rectangles for preview
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Capture - Press ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF
        # If ESC pressed cancel
        if key == 27:
            print("用户取消捕获。")
            break

        if len(faces) > 0:
            # choose biggest face
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            # crop and resize to consistent size 200x200
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200,200))
            consecutive += 1
            # show small preview of captured face
            cv2.imshow("CapturedFace", face_resized)
            if consecutive >= consecutive_required:
                captured_face = face_resized.copy()
                print("检测到稳定人脸，已捕获。")
                break
        else:
            consecutive = 0

        if time.time() - start > timeout:
            print("捕获超时。")
            break

    # cleanup
    if use_nao_cam and subscriber:
        try:
            nao_unsubscribe(subscriber)
        except:
            pass
    if cap:
        cap.release()
    cv2.destroyWindow("Capture - Press ESC to cancel")
    try:
        cv2.destroyWindow("CapturedFace")
    except:
        pass

    return captured_face, None

# ---- 学习流程 ----
def learn_flow():
    """
    按 l 键后调用：打开摄像头学习一张人脸，等待输入姓名并保存样本与训练模型
    """
    print("进入学习流程...")
    # 尝试从 NAO 摄像头捕获；若失败自动回退到本地摄像头
    face_img, _ = capture_face_from_camera(timeout=30, use_nao_cam=True)
    if face_img is None:
        print("未获取到人脸样本，学习中止。")
        return

    # Ask user for name (raw_input for py2)
    sys.stdout.write("请输入该人脸对应的姓名（回车确认）：")
    sys.stdout.flush()
    try:
        name = raw_input().strip()
    except Exception:
        name = sys.stdin.readline().strip()
    if not name:
        print("姓名为空，取消保存。")
        return

    # create person dir and save face image (use timestamp to avoid覆盖)
    person_dir = os.path.join(DATA_DIR, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    ts = int(time.time())
    fname = os.path.join(person_dir, "%d.png" % ts)
    cv2.imwrite(fname, face_img)
    print("保存人脸样本到", fname)

    # 可以多次采集并保存（此处只保存一次；你可以扩展为多次按键采集）
    # 训练识别器
    train_recognizer()

# ---- 检测流程 ----
def detect_flow():
    """
    打开摄像头进行检测（20s 超时），若检测到已保存的人脸则输出对应名称并返回
    """
    print("进入检测流程（20 秒超时）。按 ESC 可取消。")
    timeout = 20
    start = time.time()

    use_nao_cam = use_nao
    subscriber = None
    cap = None
    if use_nao_cam:
        subscriber = nao_subscribe(camera_index=0)
        if subscriber is None:
            print("无法订阅 NAO 摄像头，回退本地摄像头。")
            use_nao_cam = False

    if not use_nao_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开本地摄像头进行检测。")
            return

    load_trained_model()

    detected_name = None

    while True:
        if use_nao_cam:
            frame = nao_get_frame(subscriber)
            if frame is None:
                time.sleep(0.05)
                if time.time() - start > timeout:
                    break
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                if time.time() - start > timeout:
                    break
                continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_gray(gray)
        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200,200))
            # predict
            if recognizer is not None:
                try:
                    id_pred, conf = recognizer.predict(face_resized)
                except Exception:
                    # older API returns tuple
                    try:
                        pred = recognizer.predict(face_resized)
                        id_pred, conf = pred[0], pred[1]
                    except Exception:
                        id_pred, conf = None, None
                name = None
                if id_pred is not None and id_pred in labels_inv:
                    name = labels_inv.get(id_pred, None)
                # draw & display
                text = "Unknown"
                if name is not None and conf is not None and conf < RECOG_CONF_THRESHOLD:
                    text = "%s (%.1f)" % (name, conf)
                    detected_name = name
                else:
                    if conf is not None:
                        text = "Unknown (%.1f)" % conf
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, "No recognizer", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), 2)

        cv2.imshow("Detect - Press ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("用户取消检测。")
            break

        if detected_name:
            print("检测到已保存的人脸：", detected_name)
            break

        if time.time() - start > timeout:
            print("检测超时 (%ds 未检测到已保存的人脸)。" % timeout)
            break

    # cleanup
    if use_nao_cam and subscriber:
        try:
            nao_unsubscribe(subscriber)
        except:
            pass
    if cap:
        cap.release()
    try:
        cv2.destroyWindow("Detect - Press ESC to cancel")
    except:
        pass

# ---- 主循环 ----

def print_instructions():
    print("\n===== NAO 人脸学习与检测 程序 =====")
    print("按键说明：")
    print("  l : 学习（Learn） => 打开摄像头并学习一张人脸，随后输入姓名保存")
    print("  d : 检测（Detect） => 打开摄像头检测已保存的人脸（20s 超时）")
    print("  q : 退出程序")
    print("运行时窗口中可按 ESC 取消当前捕获/检测。\n")

def main_loop():
    load_trained_model()
    print_instructions()
    while True:
        sys.stdout.write("等待按键 (l/d/q): ")
        sys.stdout.flush()
        try:
            key = raw_input().strip().lower()
        except Exception:
            key = sys.stdin.readline().strip().lower()
        if not key:
            continue
        if key == 'l':
            learn_flow()
            print("返回主菜单。")
        elif key == 'd':
            detect_flow()
            print("返回主菜单。")
        elif key == 'q':
            print("退出程序。")
            break
        else:
            print("未知按键：", key)
            print("请按 l/d/q。")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出。")
    finally:
        cv2.destroyAllWindows()
