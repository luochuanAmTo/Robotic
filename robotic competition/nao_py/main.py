#!/usr/bin/env python
# coding=UTF-8
import sys
import os
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dance_controller import DanceController

# 机器人配置 - 请修改为你的机器人实际IP
ROBOT_IP = "192.168.43.73"  # 使用您实际连接的IP
ROBOT_PORT = 9559


def get_user_input(prompt):
    """兼容Python 2和3的输入函数"""
    try:
        # Python 2
        return raw_input(prompt)
    except NameError:
        # Python 3
        return input(prompt)


def main():
    print("NAO机器人舞蹈控制系统启动中...")

    # 初始化舞蹈控制器
    try:
        controller = DanceController(ROBOT_IP, ROBOT_PORT)
    except Exception as e:
        print("初始化失败: {}".format(e))
        print("请检查:")
        print("  1. 机器人IP地址是否正确")
        print("  2. 机器人是否开机并连接到网络")
        print("  3. Python环境是否安装了naoqi包")
        get_user_input("按Enter键退出...")
        return

    print("系统初始化完成!")

    # 主控制循环
    while True:
        try:
            controller.show_dance_menu()

            if controller.get_dance_status():
                print("\n当前状态: 正在跳舞")
            else:
                print("\n当前状态: 待机")

            # 获取用户输入
            user_input = get_user_input("\n请输入指令编号: ").strip().lower()

            if user_input == 'q':
                # 退出程序
                print("\n>>> 感谢使用NAO机器人舞蹈控制系统!")
                break

            elif user_input == 's':
                # 停止舞蹈
                controller.stop_dance()

            elif user_input.isdigit():
                # 数字指令 - 执行舞蹈
                dance_number = int(user_input)
                if dance_number in controller.dance_map:
                    if controller.get_dance_status():
                        print(">>> 请先停止当前舞蹈再开始新的舞蹈!")
                    else:
                        controller.perform_dance(dance_number)
                else:
                    print(">>> 无效的舞蹈编号，请重新输入!")

            else:
                print(">>> 无效的指令，请重新输入!")

            # 短暂暂停，让用户看到反馈
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n>>> 检测到中断信号，正在退出...")
            break
        except Exception as e:
            print("发生错误: {}".format(e))
            import traceback
            traceback.print_exc()

    # 清理资源
    controller.cleanup()
    print("程序已退出")


if __name__ == "__main__":
    main()


