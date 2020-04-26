import cv2
import numpy as np
from PIL import Image
import colorsys
import time
from itertools import product
# 30976+ 29584+32400=101
fitz = [
    (249, 229, 217),  # 62001+52441+47089= 133
    (242, 213, 195),  # 58564+45369+38025= 125
    (239, 194, 167),  # 57121+37636+27889= 116
    (193, 155, 136),  # 37249+24025+18496= 94
    (153, 113, 95),  # 23409+12769+9025= 70
    (104, 77, 66)  # 10816+5929+4356= 48
]


# def transparent_back(img):
#     L, H = img.size
#     color_0 = img.getpixel((0, 0)) + (104, 80, 70)
#     i = 0
#     for h, l in product(range(H), range(L)):
#         dot = (l, h)
#         color_1 = img.getpixel(dot)
#         if color_1 > color_0:
#             color_1 = color_1[:-1] + (255,)
#             img.putpixel(dot, color_1)
#             i += 1
#     return img, i


def diffColor(a, b):
    c = []
    for a1 in a:
        c.append(np.sqrt(a1[0] ** 2 + a1[1] ** 2 + a1[2] ** 2) / 3)
    d = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2) / 3
    return c, d


def minIndex(c, d):
    x = np.abs(c-d).tolist()
    x = x.index(min(x))
    return x


def get_dominant_color(image):
    # 生成缩略图，减少计算量，减小cpu压力
    image = image.convert("RGBA")

    max_score = 0  # 原来的代码此处为None
    dominant_color = 0  # 原来的代码此处为None，但运行出错，改为0以后运行成功，原因在于在下面的score > max_score的比较中，max_score的初始格式不定

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        # if r < 110 or g < 90 or b < 90:
        # if r < 104 or g < 80 or b < 70:
        #     continue
        if (r, g, b) < (104, 80, 70):
            continue
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235.0 - 16.0)

        # 忽略高亮色
        if y > 0.9:
            continue
        saturation = colorsys.rgb_to_hsv(r, g, b)[2]
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    return dominant_color


def SkinColor(i):
    img = cv2.imread(i)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # rgb转hsv格式
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (10, 15, 10), (17, 170, 255))  # 设置阈值，去除背景部分
    # HSV_mask = cv2.inRange(img_HSV, (20, 20, 10), (20,190,255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 被用来填充前景物体中的小洞，或者前景物体上的小黑点。
    HSV_mask = 255 - HSV_mask

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # rgb转YCrCb格式
    # skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (10, 135, 85), (255, 180, 135))  # 设置阈值，去掉背景部分
    # YCrCb_mask = cv2.inRange(img_YCrCb, (50, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 被用来填充前景物体中的小洞，或者前景物体上的小黑点。
    YCrCb_mask = 255 - YCrCb_mask

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)  # 掩膜与在图像上的区域融合
    global_mask = cv2.medianBlur(global_mask, 3)  # 中值滤波
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    global_mask = 255 - global_mask
    global_result_ = cv2.bitwise_not(global_mask)

    # global_index = global_result > 0.85
    # global_noindex = global_result < 0.85
    global_result = cv2.cvtColor(global_result_, cv2.COLOR_RGB2BGRA)
    # indexs = np.array(np.where(global_index == True))
    # indexs = np.stack(indexs, axis=1)
    img[0][:, [0, 2]] = img[0][:, [2, 0]]
    img = img[:, :, [2, 1, 0]]
    global_result[global_result_ >= 0.85] = [0, 0, 0, 255]
    global_result[global_result_ < 0.85] = [255, 255, 255, 0]
    mask = Image.fromarray(np.uint8(global_result))
    img = Image.fromarray(np.uint8(img))
    b, g, r, a = mask.split()
    img.paste(mask, mask=a)

    color = get_dominant_color(img)
    print("color", color)
    c, d = diffColor(fitz, color)
    index = minIndex(c, d)
    print("{}号肤色".format(index+1))
    if index < 3:
        print("冷色调")
    else:
        print("暖色调")
    # cv2.imshow("3_global_result.jpg", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    img.show()


if __name__ == '__main__':
    path = r"A:\testdata\15.jpg"
    stime = time.time()
    SkinColor(path)
    etime = time.time()
    print(etime-stime)


