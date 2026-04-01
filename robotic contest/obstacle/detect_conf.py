# -*- coding: utf-8 -*-

from builtins import FileNotFoundError
import numpy as np
import cv2
import json


class BHumanAdvancedConfig:


    def __init__(self):
        # 颜色识别的高级参数
        self.advanced_color_params = {
            # 红色配置
            'red': {
                'hsv_ranges': [
                    {'h_min': 0, 'h_max': 10, 's_min': 120, 'v_min': 70},
                    {'h_min': 170, 'h_max': 180, 's_min': 120, 'v_min': 70}
                ],
                'lab_ranges': [
                    {'l_min': 30, 'a_min': 20, 'b_min': 15}
                ],
                'rgb_ratios': {
                    'r_threshold': 0.4,  # R通道占比
                    'rg_ratio_min': 1.2  # R/G比值
                }
            },

            # 黄色配置 (特别优化浅黄色)
            'yellow': {
                'hsv_ranges': [
                    {'h_min': 15, 'h_max': 35, 's_min': 80, 'v_min': 100},
                    {'h_min': 20, 'h_max': 30, 's_min': 50, 'v_min': 150}  # 浅黄色
                ],
                'lab_ranges': [
                    {'l_min': 60, 'a_min': -10, 'b_min': 20}
                ],
                'rgb_ratios': {
                    'rg_ratio_min': 0.8,
                    'rb_ratio_min': 1.5
                }
            },

            # 蓝色配置 (特别优化浅蓝色)
            'blue': {
                'hsv_ranges': [
                    {'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50},
                    {'h_min': 90, 'h_max': 120, 's_min': 30, 'v_min': 120},  # 浅蓝色
                    {'h_min': 110, 'h_max': 140, 's_min': 80, 'v_min': 80}  # 深蓝色
                ],
                'lab_ranges': [
                    {'l_min': 30, 'a_min': -20, 'b_min': -30}
                ],
                'rgb_ratios': {
                    'b_threshold': 0.35,
                    'bg_ratio_min': 1.1
                }
            }
        }

        # 光照适应性参数
        self.lighting_adaptation = {
            'auto_white_balance': True,
            'gamma_correction': 1.2,
            'contrast_enhancement': True,
            'shadow_compensation': True,
            'highlight_suppression': True
        }

        # 形态学参数
        self.morphology_params = {
            'erosion_kernel_size': 3,
            'dilation_kernel_size': 5,
            'closing_iterations': 2,
            'opening_iterations': 1
        }

    def multi_space_color_detection(self, image, color_name):
        """多颜色空间融合检测 - B-Human核心算法"""
        height, width = image.shape[:2]

        # 1. HSV检测
        hsv_mask = self._hsv_detection(image, color_name)

        # 2. LAB检测
        lab_mask = self._lab_detection(image, color_name)

        # 3. RGB比值检测
        rgb_mask = self._rgb_ratio_detection(image, color_name)

        # 4. YUV检测 (补充)
        yuv_mask = self._yuv_detection(image, color_name)

        # 融合多个颜色空间的结果
        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        combined_mask = cv2.bitwise_and(combined_mask, rgb_mask)

        # 添加YUV作为补充验证
        yuv_weight = 0.3
        combined_mask = cv2.addWeighted(combined_mask, 0.7, yuv_mask, yuv_weight, 0)

        return combined_mask

    def _hsv_detection(self, image, color_name):
        """HSV颜色空间检测"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        ranges = self.advanced_color_params[color_name]['hsv_ranges']
        for range_param in ranges:
            lower = np.array([range_param['h_min'], range_param['s_min'], range_param['v_min']])
            upper = np.array([range_param['h_max'], 255, 255])
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, range_mask)

        return mask

    def _lab_detection(self, image, color_name):
        """LAB颜色空间检测"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        ranges = self.advanced_color_params[color_name]['lab_ranges']
        mask = np.zeros(lab.shape[:2], dtype=np.uint8)

        for range_param in ranges:
            if color_name == 'red':
                # 红色在a通道为正值
                condition = (l >= range_param['l_min']) & (a >= range_param['a_min']) & (b >= range_param['b_min'])
            elif color_name == 'yellow':
                # 黄色在b通道为正值
                condition = (l >= range_param['l_min']) & (a >= range_param['a_min']) & (b >= range_param['b_min'])
            elif color_name == 'blue':
                # 蓝色在b通道为负值
                condition = (l >= range_param['l_min']) & (a <= range_param['a_min']) & (b <= range_param['b_min'])

            range_mask = condition.astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, range_mask)

        return mask

    def _rgb_ratio_detection(self, image, color_name):
        """RGB比值检测 - 对光照变化更鲁棒"""
        b, g, r = cv2.split(image.astype(np.float32))
        total = r + g + b + 1e-6  # 避免除零

        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        ratios = self.advanced_color_params[color_name]['rgb_ratios']

        if color_name == 'red':
            condition = (r_ratio >= ratios['r_threshold']) & (r / (g + 1e-6) >= ratios['rg_ratio_min'])
        elif color_name == 'yellow':
            condition = (r / (g + 1e-6) >= ratios['rg_ratio_min']) & (r / (b + 1e-6) >= ratios['rb_ratio_min'])
        elif color_name == 'blue':
            condition = (b_ratio >= ratios['b_threshold']) & (b / (g + 1e-6) >= ratios['bg_ratio_min'])

        mask = (condition * 255).astype(np.uint8)
        return mask

    def _yuv_detection(self, image, color_name):
        """YUV颜色空间补充检测"""
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if color_name == 'red':
            # 红色在V通道有高值
            condition = (v > 130) & (u < 120)
        elif color_name == 'yellow':
            # 黄色在V通道中等值，U通道低值
            condition = (v > 120) & (u < 110) & (y > 100)
        elif color_name == 'blue':
            # 蓝色在U通道有高值
            condition = (u > 130) & (v < 120)

        mask = (condition * 255).astype(np.uint8)
        return mask

    def adaptive_lighting_correction(self, image):
        """自适应光照校正 - B-Human核心技术"""
        # 1. 白平衡校正
        if self.lighting_adaptation['auto_white_balance']:
            image = self._auto_white_balance(image)

        # 2. Gamma校正
        gamma = self.lighting_adaptation['gamma_correction']
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, look_up_table)

        # 3. 对比度增强
        if self.lighting_adaptation['contrast_enhancement']:
            image = self._enhance_contrast(image)

        # 4. 阴影补偿
        if self.lighting_adaptation['shadow_compensation']:
            image = self._compensate_shadows(image)

        return image

    def _auto_white_balance(self, image):
        """自动白平衡"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def _enhance_contrast(self, image):
        """增强对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _compensate_shadows(self, image):
        """阴影补偿"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 创建阴影掩码
        shadow_mask = gray < np.percentile(gray, 30)

        # 对阴影区域进行亮度提升
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.where(shadow_mask,
                                np.clip(hsv[:, :, 2] * 1.3, 0, 255),
                                hsv[:, :, 2])

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# 使用示例和配置
BHUMAN_DETECTION_CONFIG = {
    "camera_settings": {
        "auto_exposure": False,
        "exposure_value": 50,
        "auto_white_balance": False,
        "white_balance_temperature": 4000,
        "brightness": 55,
        "contrast": 60,
        "saturation": 128,
        "hue": 0,
        "gain": 32
    },

    "detection_thresholds": {
        "min_cylinder_area": 500,
        "max_cylinder_area": 50000,
        "min_circularity": 0.3,
        "max_distance": 3.0,  # 3米
        "confidence_threshold": 0.7
    },

    "bhuman_specific": {
        "use_multi_space_detection": True,
        "enable_adaptive_lighting": True,
        "use_temporal_consistency": True,
        "kalman_filter_enabled": True,
        "detection_history_length": 5
    }
}


def save_config_to_file(config, filename="bhuman_cylinder_config.json"):
    """保存配置到文件"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print("配置已保存到: {}".format(filename))


def load_config_from_file(filename="bhuman_cylinder_config.json"):
    """从文件加载配置"""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        print("配置已从文件加载: {}".format(filename))
        return config
    except FileNotFoundError:
        print("配置文件不存在，使用默认配置")
        return BHUMAN_DETECTION_CONFIG




if __name__ == "__main__":

    # 保存默认配置
    save_config_to_file(BHUMAN_DETECTION_CONFIG)

    # 创建高级配置实例
    advanced_config = BHumanAdvancedConfig()
    print("高级配置已初始化")