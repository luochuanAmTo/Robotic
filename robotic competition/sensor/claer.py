# reset_asr.py
# -*- coding: utf-8 -*-
"""
重置NAO机器人语音识别引擎
"""
import sys
import time
from naoqi import ALProxy


def reset_speech_recognition(ip):
    """完全重置语音识别引擎"""
    try:
        print("正在连接到NAO机器人: %s" % ip)

        # 连接到所有必要的服务
        asr = ALProxy("ALSpeechRecognition", ip, 9559)
        audio = ALProxy("ALAudioDevice", ip, 9559)
        print("连接成功")

        print("\n1. 获取当前语音识别状态...")
        try:
            # 获取所有订阅者
            subscribers = asr.getSubscribersList()
            print("当前订阅者: %s" % subscribers)

            # 取消所有订阅
            for sub in subscribers:
                print("取消订阅: %s" % sub)
                try:
                    asr.unsubscribe(sub)
                except:
                    pass
        except Exception as e:
            print("获取订阅者列表失败: %s" % e)

        print("\n2. 完全停止语音识别引擎...")
        try:
            # 先暂停
            asr.pause(True)
            print("语音识别已暂停")

            # 停止处理
            asr.stopProcessing()
            print("语音处理已停止")

            # 重置引擎
            asr.setAudioExpression(False)
            print("音频表达式已禁用")

        except Exception as e:
            print("停止引擎时出错: %s" % e)

        print("\n3. 重新启动音频系统...")
        try:
            # 重启音频设备
            audio.disableEnergyComputation()
            time.sleep(0.5)
            audio.enableEnergyComputation()
            print("音频系统已重启")
        except Exception as e:
            print("重启音频系统失败: %s" % e)

        print("\n4. 设置基本参数...")
        try:
            # 先设置语言
            asr.setLanguage("Chinese")
            print("语言已设置为: Chinese")

            # 设置词汇表
            vocabulary = ["测试", "你好"]
            asr.setVocabulary(vocabulary, False)
            print("词汇表已设置")

        except Exception as e:
            print("设置参数失败: %s" % e)

        print("\n5. 测试语音识别...")
        try:
            asr.subscribe("ResetTest")
            print("语音识别已订阅")
            time.sleep(2)
            asr.unsubscribe("ResetTest")
            print("测试完成")
        except Exception as e:
            print("测试失败: %s" % e)

        print("\n" + "=" * 50)
        print("语音识别引擎重置完成!")
        print("现在可以运行主程序了")
        print("=" * 50)

        return True

    except Exception as e:
        print("重置失败: %s" % e)
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = raw_input("请输入NAO机器人IP地址: ").strip()

    reset_speech_recognition(ip)