# coding=UTF-8
import time
import threading
from naoqi import ALProxy
import dance_movements as Angle


class DanceController:
    def __init__(self, robot_ip, robot_port=9559):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.is_dancing = False
        self.current_dance_thread = None

        # 舞蹈映射
        self.dance_map = {
            0: "NoNoNo",
            1: "Gee",
            2: "Loveyou",
            3: "Apple",
            4: "Shanghai",
            5: "Style",
            6: "Taiji",
            7: "Thriller",
            8: "Hello"
        }

        # 初始化代理
        try:
            self.motion_proxy = ALProxy("ALMotion", self.robot_ip, self.robot_port)
            self.posture_proxy = ALProxy("ALRobotPosture", self.robot_ip, self.robot_port)
            self.audio_player = ALProxy("ALAudioPlayer", self.robot_ip, self.robot_port)
            print("成功连接到机器人: {}:{}".format(self.robot_ip, self.robot_port))
        except Exception as e:
            print("连接机器人失败: {}".format(e))
            raise

    def show_dance_menu(self):
        """显示舞蹈菜单"""
        print("\n" + "=" * 50)
        print("           NAO机器人舞蹈控制系统")
        print("=" * 50)
        print("可用舞蹈指令:")
        for num, name in self.dance_map.items():
            print("  [{}] - {}".format(num, name))
        print("  [s] - 停止当前舞蹈")
        print("  [q] - 退出程序")
        print("=" * 50)

    def perform_dance(self, dance_number):
        """执行指定编号的舞蹈"""
        if dance_number not in self.dance_map:
            print("无效的舞蹈编号: {}".format(dance_number))
            return False

        dance_name = self.dance_map[dance_number]
        print("\n>>> 开始舞蹈: {}".format(dance_name))

        # 在新线程中执行舞蹈
        self.current_dance_thread = threading.Thread(
            target=self._dance_routine,
            args=(dance_name,)
        )
        self.current_dance_thread.daemon = True
        self.current_dance_thread.start()

        return True

    def _dance_routine(self, dance_name):
        """舞蹈执行例程"""
        try:
            self.is_dancing = True

            # 停止当前动作
            self.motion_proxy.stopMove()

            # 播放音乐
            self.play_music(dance_name)

            # 准备舞蹈姿势
            self.posture_proxy.goToPosture("StandInit", 1.0)
            time.sleep(0.5)

            # 获取舞蹈动作
            dance_class = getattr(Angle, dance_name, None)
            if dance_class is None:
                print("未找到舞蹈类: {}".format(dance_name))
                self.is_dancing = False
                return

            dance_instance = dance_class()

            # 执行舞蹈动作
            print("执行舞蹈动作...")
            self.motion_proxy.angleInterpolationBezier(
                dance_instance.names,
                dance_instance.times,
                dance_instance.keys
            )

            # 等待舞蹈完成
            max_duration = max([max(times) for times in dance_instance.times])
            time.sleep(max_duration + 1)

            print(">>> 舞蹈 {} 完成!".format(dance_name))

        except Exception as e:
            print("执行舞蹈时出错: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            self.is_dancing = False

    def play_music(self, dance_name):
        """播放对应舞蹈的音乐"""
        try:
            file_path = "/home/nao/welcome/music/{}.mp3".format(dance_name)
            print("尝试播放音乐: {}".format(file_path))
            file_id = self.audio_player.loadFile(file_path)
            self.audio_player.play(file_id)
            print("播放音乐: {}".format(dance_name))
        except Exception as e:
            print("播放音乐失败: {}".format(e))

    def stop_dance(self):
        """停止舞蹈和音乐"""
        try:
            if self.is_dancing:
                print("\n>>> 停止当前舞蹈...")
                self.motion_proxy.stopMove()
                self.audio_player.stopAll()
                self.is_dancing = False

                if self.current_dance_thread and self.current_dance_thread.is_alive():
                    self.current_dance_thread.join(timeout=1.0)

                print(">>> 已停止舞蹈和音乐")
            else:
                print(">>> 当前没有正在进行的舞蹈")
        except Exception as e:
            print("停止舞蹈时出错: {}".format(e))

    def get_dance_status(self):
        """获取舞蹈状态"""
        return self.is_dancing

    def cleanup(self):
        """清理资源"""
        self.stop_dance()
        print(">>> 资源清理完成")

