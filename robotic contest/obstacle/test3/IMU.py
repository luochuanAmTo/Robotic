import time
from naoqi import ALProxy


def get_robot_position(robot_ip="192.168.1.3", robot_port=9559):
    try:
        motion = ALProxy("ALMotion", robot_ip, robot_port)

        # 获取里程计位置（相对起始点）
        odom_pos = motion.getRobotPosition(use_sensors=True)
        print("里程计位置: X={odom_pos[0]:.2f}m, Y={odom_pos[1]:.2f}m, θ={math.degrees(odom_pos[2]):.1f}°")

        # 尝试获取地图中的位置（如果有SLAM）
        try:
            localization = ALProxy("ALLocalization", robot_ip, robot_port)
            map_pos = localization.getRobotPosition()
            print("地图位置: X={map_pos[0]:.2f}m, Y={map_pos[1]:.2f}m, θ={math.degrees(map_pos[2]):.1f}°")
        except:
            print("未启用SLAM定位")

        return odom_pos

    except Exception as e:
        print("Error:", e)
        return None


if __name__ == "__main__":
    import math

    current_pos = get_robot_position()