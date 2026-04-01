import time
import math
from naoqi import ALProxy


def get_robot_position(robot_ip="192.168.43.247", robot_port=9559):
    try:
        motion = ALProxy("ALMotion", robot_ip, robot_port)

        # 获取里程计位置（相对起始点）
        odom_pos = motion.getRobotPosition(use_sensors=True)
        # Python 2.7兼容的字符串格式化
        print("里程计位置: X={:.2f}m, Y={:.2f}m, θ={:.1f}°".format(
            odom_pos[0], odom_pos[1], math.degrees(odom_pos[2])))

        # 尝试获取地图中的位置（如果有SLAM）
        try:
            localization = ALProxy("ALLocalization", robot_ip, robot_port)
            map_pos = localization.getRobotPosition()
            print("地图位置: X={:.2f}m, Y={:.2f}m, θ={:.1f}°".format(
                map_pos[0], map_pos[1], math.degrees(map_pos[2])))
        except:
            print("未启用SLAM定位")

        return odom_pos

    except Exception as e:
        print("Error: {}".format(e))
        return None


if __name__ == "__main__":
    current_pos = get_robot_position()