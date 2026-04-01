# coding:utf-8

from proxy_and_image import *


def preprocess_image(img, low_range, high_range):
    """
    预处理图像：将图像转换为HSV格式，并应用颜色阈值。
    """
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(low_range)
    upper = np.array(high_range)
    mask = cv2.inRange(imgHSV, lower, upper)
    return mask


def combine_masks(mask1, mask2):
    """
    结合两个掩膜
    """
    res = mask1 + mask2
    res[res > 1] = 1
    return res


def main():
    img = cv2.imread(CONFIG["path"])
    # 使用配置的颜色阈值进行图像预处理
    res1 = preprocess_image(img, CONFIG["white_low"], CONFIG["white_high"])
    res2 = preprocess_image(img, CONFIG["black_low"], CONFIG["black_high"])

    # 结合两个掩膜
    res = combine_masks(res1, res2)

    # 显示结果
    cv2.imshow("Combined Mask", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 启动颜色选择器
    # color_picker()


if __name__ == 'main':
    main()
