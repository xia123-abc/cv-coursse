import cv2
import numpy as np

# 创建一个彩色图片
img = np.zeros((400,600,3), dtype=np.uint8)
img[:,:,0] = 255  # 红色通道全满
img[:,:,1] = 0    # 绿色通道
img[:,:,2] = 0    # 蓝色通道

cv2.imwrite('red.jpg', img)
print("生成了红色图片 red.jpg")

# 读取并检查
test = cv2.imread('red.jpg')
print(f"图片尺寸: {test.shape}")
print(f"红色通道平均值: {test[:,:,2].mean()}")  # OpenCV 是 BGR，所以红色在第三个通道