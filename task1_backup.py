import cv2
import numpy as np
from matplotlib import pyplot as plt

# 任务1：读取测试图片
img = cv2.imread('test.jpg')
if img is None:
    print("图片读取失败，请检查文件路径")
    exit()

# 任务2：输出图像基本信息
print("="*50)
print("图像基本信息：")
print(f"图像尺寸 (高, 宽, 通道): {img.shape}")
print(f"图像通道数: {img.shape[2]}")
print(f"图像高度: {img.shape[0]} 像素")
print(f"图像宽度: {img.shape[1]} 像素")
print(f"像素数据类型: {img.dtype}")
print(f"图像总像素数: {img.size}")
print("="*50)

# 任务3：显示原图（使用OpenCV）
cv2.imshow('Original Image', img)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()

# 任务4：转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 用matplotlib显示原图和灰度图对比
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()

# 任务5：保存灰度图
cv2.imwrite('gray_test.jpg', gray)
print("灰度图已保存为 gray_test.jpg")

# 任务6：NumPy简单操作 - 输出左上角(10,10)位置的像素值
pixel_value = img[10, 10]
print(f"左上角(10,10)位置的BGR像素值: {pixel_value}")

# 任务6扩展：裁剪左上角100x100区域
crop = img[0:100, 0:100]
cv2.imwrite('crop_test.jpg', crop)
print("裁剪区域已保存为 crop_test.jpg")

