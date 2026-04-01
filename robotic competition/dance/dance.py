# # -*- coding: utf-8 -*-
# import sys
# import time
# import threading
# import os
# import subprocess
# from naoqi import ALProxy
# from movent import Taiji
#
#
# def play_music_system(mp3_file):
#     """使用系统默认播放器播放音乐"""
#     try:
#         if sys.platform == "win32":
#             # Windows
#             os.startfile(mp3_file)
#         elif sys.platform == "darwin":
#             # macOS
#             subprocess.call(["open", mp3_file])
#         else:
#             # Linux
#             subprocess.call(["xdg-open", mp3_file])
#         print("音乐开始播放")
#     except Exception as e:
#         print("系统播放器错误: {}".format(e))
#
#
# def play_music_ffplay(mp3_file):
#     """使用ffplay播放音乐（需要安装FFmpeg）"""
#     try:
#         subprocess.Popen(["ffplay", "-nodisp", "-autoexit", mp3_file])
#         print("音乐开始播放 (ffplay)")
#     except Exception as e:
#         print("ffplay播放错误: {}".format(e))
#         # 回退到系统播放器
#         play_music_system(mp3_file)
#
#
# def play_music_mpg123(mp3_file):
#     """使用mpg123播放音乐（需要安装mpg123）"""
#     try:
#         subprocess.Popen(["mpg123", mp3_file])
#         print("音乐开始播放 (mpg123)")
#     except Exception as e:
#         print("mpg123播放错误: {}".format(e))
#         # 回退到系统播放器
#         play_music_system(mp3_file)
#
#
# def stop_music():
#     """停止音乐播放"""
#     try:
#         if sys.platform == "win32":
#             os.system("taskkill /f /im ffplay.exe 2>nul")
#             os.system("taskkill /f /im mpg123.exe 2>nul")
#         else:
#             os.system("pkill -f ffplay 2>/dev/null")
#             os.system("pkill -f mpg123 2>/dev/null")
#     except:
#         pass
#
#
# def main(robot_ip, robot_port=9559):
#     try:
#         # 初始化运动代理
#         motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
#         posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)
#
#         # 唤醒机器人
#         motion_proxy.wakeUp()
#         posture_proxy.goToPosture("StandInit", 0.5)
#
#         # 创建太极动作实例
#         taiji_movement = Taiji()
#
#         print("准备开始太极表演...")
#         time.sleep(1)
#
#         # 播放音乐（选择一种方式）
#         mp3_file = "Taiji.mp3"
#
#         if not os.path.exists(mp3_file):
#             print("警告: 未找到 '{}' 文件，将只执行动作不播放音乐".format(mp3_file))
#             music_thread = None
#         else:
#             # 选择播放方式（取消注释你想要使用的方式）
#             music_thread = threading.Thread(target=play_music_system, args=(mp3_file,))
#             # music_thread = threading.Thread(target=play_music_ffplay, args=(mp3_file,))
#             # music_thread = threading.Thread(target=play_music_mpg123, args=(mp3_file,))
#
#             music_thread.daemon = True
#             music_thread.start()
#
#         print("开始表演太极...")
#
#         # 执行太极动作
#         motion_proxy.angleInterpolationBezier(taiji_movement.names, taiji_movement.times, taiji_movement.keys)
#
#         # 计算总时间
#         total_time = 0
#         for joint_times in taiji_movement.times:
#             if joint_times and joint_times[-1] > total_time:
#                 total_time = joint_times[-1]
#
#         print("动作执行时间: {} 秒".format(total_time))
#         time.sleep(total_time + 1)
#
#         # 停止音乐
#         stop_music()
#
#         # 回到初始姿势
#         print("表演结束，回到初始姿势...")
#         posture_proxy.goToPosture("StandInit", 0.5)
#         motion_proxy.rest()
#
#         print("太极表演完成！")
#
#     except Exception as e:
#         print("错误: {}".format(e))
#         stop_music()
#         try:
#             motion_proxy.rest()
#         except:
#             pass
#
#
# if __name__ == "__main__":
#     ROBOT_IP = "192.168.43.73"  # 替换为你的NAO机器人的实际IP地址
#     ROBOT_PORT = 9559
#
#     if len(sys.argv) > 1:
#         ROBOT_IP = sys.argv[1]
#
#     print("连接机器人: {}".format(ROBOT_IP))
#     main(ROBOT_IP, ROBOT_PORT)
#


# -*- coding: utf-8 -*-
import sys
import time
import threading
import os
from naoqi import ALProxy
from moement3 import Loveyou


def play_music_on_robot(robot_ip, robot_port, mp3_file_path):
    """在NAO机器人上播放音乐"""
    try:
        # 初始化音频播放代理
        audio_player = ALProxy("ALAudioPlayer", robot_ip, robot_port)

        # 停止所有正在播放的音乐
        audio_player.stopAll()

        # 加载并播放音乐文件
        print("尝试播放音乐: {}".format(mp3_file_path))
        file_id = audio_player.loadFile(mp3_file_path)
        audio_player.play(file_id)
        print("音乐开始播放")
        return True

    except Exception as e:
        print("在机器人上播放音乐错误: {}".format(e))
        return False


def stop_music_on_robot(robot_ip, robot_port):
    """停止NAO机器人上的音乐播放"""
    try:
        audio_player = ALProxy("ALAudioPlayer", robot_ip, robot_port)
        audio_player.stopAll()
        print("音乐已停止")
    except Exception as e:
        print("停止音乐时出错: {}".format(e))


def main(robot_ip, robot_port=9559):
    try:
        # 初始化运动代理
        motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
        posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)

        # 唤醒机器人
        motion_proxy.wakeUp()
        posture_proxy.goToPosture("StandInit", 0.5)

        #机器人自我介绍写在这里

        # 创建太极动作实例
        taiji_movement = Loveyou()

        print("准备开始太极表演...")
        time.sleep(1)

        # 在机器人上播放音乐
        mp3_file_path = "/home/nao/welcome/music/Loveyou.mp3"

        # 在后台线程中播放音乐
        music_thread = threading.Thread(
            target=play_music_on_robot,
            args=(robot_ip, robot_port, mp3_file_path)
        )
        music_thread.daemon = True
        music_thread.start()

        # 等待音乐开始播放
        time.sleep(2)

        print("开始表演太极...")

        # 执行太极动作
        motion_proxy.angleInterpolationBezier(taiji_movement.names, taiji_movement.times, taiji_movement.keys)

        # 计算总时间
        total_time = 0
        for joint_times in taiji_movement.times:
            if joint_times and joint_times[-1] > total_time:
                total_time = joint_times[-1]

        print("动作执行时间: {} 秒".format(total_time))

        # 等待动作完成
        time.sleep(total_time + 1)

        # 停止音乐
        stop_music_on_robot(robot_ip, robot_port)

        # 回到初始姿势
        print("表演结束，回到初始姿势...")
        posture_proxy.goToPosture("StandInit", 0.5)
        motion_proxy.rest()

        print("太极表演完成！")

    except Exception as e:
        print("错误: {}".format(e))
        # 发生错误时也尝试停止音乐
        try:
            stop_music_on_robot(robot_ip, robot_port)
        except:
            pass
        try:
            motion_proxy.rest()
        except:
            pass


if __name__ == "__main__":
    ROBOT_IP = "192.168.43.58"  # 替换为你的NAO机器人的实际IP地址
    ROBOT_PORT = 9559

    if len(sys.argv) > 1:
        ROBOT_IP = sys.argv[1]

    print("连接机器人: {}".format(ROBOT_IP))
    main(ROBOT_IP, ROBOT_PORT)
