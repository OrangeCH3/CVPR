import cv2
import numpy as np


def test():
    """
    调用系统库函数进行测试
    """

    image1 = cv2.imread('./image/cali.png')
    image1 = cv2.resize(image1, dsize=(600, 400))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)
    # 角点检测 第三个参数为角点检测的敏感度，其值必须介于3~31之间的奇数
    aim1 = cv2.cornerHarris(gray1, 3, 3, 0.04)
    print(aim1.shape)  # (400, 600)
    image1[aim1 > 0.01 * aim1.max()] = [0, 0, 255]
    cv2.imshow('', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def harris_detect(img, ksize=3):
    """
    自己实现角点检测

    params:
        img:灰度图片
        ksize：Sobel算子窗口大小

    return：
        corner：与源图像一样大小，角点处像素值设置为255
    """
    k = 0.04  # 响应函数k
    threshold = 0.01  # 设定阈值
    WITH_NMS = False  # 是否非极大值抑制

    # 1、使用Sobel计算像素点x,y方向的梯度
    h, w = img.shape[:2]
    # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 2、计算Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3、利用高斯函数对Ix^2,Iy^2,Ix*Iy进行滤波
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5、将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i, j] > R_max * threshold and R[i, j] == np.max(
                        R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                    corner[i, j] = 255
            else:
                # 只进行阈值检测
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255
    return corner


if __name__ == '__main__':
    image = cv2.imread('./image/cali.png')
    image = cv2.resize(image, dsize=(600, 400))
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = harris_detect(gray)
    print(dst.shape)  # (400, 600)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
