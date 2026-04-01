# -*- coding: utf-8 -*-
import sys
import time
import threading
import os
from naoqi import ALProxy
from Taiji import Taiji


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


def robot_self_introduction(robot_ip, robot_port):
    """机器人自我介绍"""
    try:
        # 初始化语音代理
        tts_proxy = ALProxy("ALTextToSpeech", robot_ip, robot_port)
        audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)

        # 设置语音参数
        audio_device.setOutputVolume(60)  # 设置音量
        tts_proxy.setLanguage("Chinese")  # 设置中文
        tts_proxy.setParameter("speed", 90)  # 设置语速

        # 自我介绍内容
        introduction = """
        
        """
#今天很高兴为大家表演，希望我的舞蹈能够给大家带来欢乐！
#接下来请欣赏舞蹈！


        print("机器人开始自我介绍...")

        # 分段说出自我介绍，避免一次性太长
        sentences = introduction.strip().split('。')
        for sentence in sentences:
            if sentence.strip():  # 跳过空句子
                tts_proxy.say(sentence.strip() + "。")
                time.sleep(0.5)  # 句子间短暂停顿

        print("自我介绍完成")
        return True

    except Exception as e:
        print("自我介绍失败: {}".format(e))
        return False


def main(robot_ip, robot_port=9559):
    try:
        # 初始化运动代理
        motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
        posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)

        # 唤醒机器人
        motion_proxy.wakeUp()
        posture_proxy.goToPosture("StandInit", 0.5)

        # 机器人自我介绍
        print("=== 机器人自我介绍环节 ===")
        if not robot_self_introduction(robot_ip, robot_port):
            print("警告: 自我介绍失败，继续执行舞蹈")

        time.sleep(9)  # 自我介绍后稍作停顿

        # 创建舞蹈动作实例
        dance_movement = Taiji()

        print("准备开始舞蹈表演...")


        # 在机器人上播放音乐
        mp3_file_path = "/home/nao/welcome/music/Taiji.mp3"

        # 在后台线程中播放音乐
        music_thread = threading.Thread(
            target=play_music_on_robot,
            args=(robot_ip, robot_port, mp3_file_path)
        )
        music_thread.daemon = True
        music_thread.start()

        # 等待音乐开始播放


        print("开始表演舞蹈...")

        # 执行舞蹈动作
        motion_proxy.angleInterpolationBezier(dance_movement.names, dance_movement.times, dance_movement.keys)

        # 计算总时间
        # total_time = 0
        # for joint_times in dance_movement.times:
        #     if joint_times and joint_times[-1] > total_time:
        #         total_time = joint_times[-1]
        #
        # print("动作执行时间: {} 秒".format(total_time))
        #
        # # 等待动作完成
        # time.sleep(total_time + 1)
        #
        # # 停止音乐
        # stop_music_on_robot(robot_ip, robot_port)

        # 回到初始姿势
        print("表演结束，回到初始姿势...")
        posture_proxy.goToPosture("StandInit", 0.5)

        # 表演结束语
        try:
            tts_proxy = ALProxy("ALTextToSpeech", robot_ip, robot_port)

            print("机器人说: 表演结束，谢谢大家观看！")
        except:
            pass

        motion_proxy.rest()

        print("舞蹈表演完成！")

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

    print("=" * 50)
    print("      NAO机器人舞蹈表演系统")
    print("=" * 50)
    print("机器人IP: {}".format(ROBOT_IP))
    print("表演流程:")
    print("1. 机器人自我介绍")
    print("2. 播放音乐")
    print("3. 执行舞蹈动作")
    print("4. 表演结束语")
    print("=" * 50)

    print("连接机器人: {}".format(ROBOT_IP))
    main(ROBOT_IP, ROBOT_PORT)