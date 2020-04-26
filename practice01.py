import cv2
import numpy as np
from PIL import Image
import colorsys
import time
from sklearn import cluster as cl
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 30976+ 29584+32400=101
fitz = [
    (249, 229, 217),  # 62001+52441+47089= 133
    (242, 213, 195),  # 58564+45369+38025= 125
    (239, 194, 167),  # 57121+37636+27889= 116
    (193, 155, 136),  # 37249+24025+18496= 94
    (153, 113, 95),  # 23409+12769+9025= 70
    (104, 77, 66)  # 10816+5929+4356= 48
]


# 计算欧式距离
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 初始化簇中心点 一开始随机从样本中选择k个 当做各类簇的中心
def initCentroid(data, k):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)
    for i in range(k):
        index = int(l[i])
        centpoint[i] = data[index]
    return centpoint


# 进行KMeans分类
def KMeans(data, k):
    # 样本个数
    num = np.shape(data)[0]

    # 记录各样本 簇信息 0:属于哪个簇 1:距离该簇中心点距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1

    # 记录是否有样本改变簇分类
    change = True
    # 初始化各簇中心点
    cp = initCentroid(data, k)

    while change:
        change = False

        # 遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1

            # 计算该样本距离每一个簇中心点的距离 找到距离最近的中心点
            for j in range(k):
                dis = distEuclid(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j

            # 如果找到的簇中心点非当前簇 则改变该样本的簇分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist

        # 根据样本重新分类  计算新的簇中心点
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)

    # print("finish!")
    return cp, cluster


# 展示结果  各类簇使用不同的颜色  中心点使用X表示
def Show(data, k, cp, cluster):
    num, dim = data.shape
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    # 二维图
    if dim == 2:
        for i in range(num):
            mark = int(cluster[i, 0])
            plt.plot(data[i, 0], data[i, 1], color[mark] + 'o')

        for i in range(k):
            plt.plot(cp[i, 0], cp[i, 1], color[i] + 'x')
    # 三维图
    elif dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(num):
            mark = int(cluster[i, 0])
            ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=color[mark])

        for i in range(k):
            ax.scatter(cp[i, 0], cp[i, 1], cp[i, 2], c=color[i], marker='x')
    plt.show()


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
    colors = []
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        if (r, g, b) < (104, 80, 70):
            continue
        colors.append([r, g, b])
    data = np.array(colors)
    cp, cluster = KMeans(data, 1)  # 聚类
    # Show(data, 1, cp, cluster)
    print(cp)
    return cp[0]


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

    global_result = cv2.cvtColor(global_result_, cv2.COLOR_RGB2BGRA)
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
    path = r"D:\E\document\datas\whitelight\0.jpg"
    stime = time.time()
    SkinColor(path)
    etime = time.time()
    print(etime-stime)


