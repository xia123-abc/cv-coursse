import cv2
import matplotlib.pyplot as plt

# 1. 读入图像
img = cv2.imread('test.jpg')
if img is None:
    print("错误：无法读取图片，请检查文件路径")
    exit()

# 2. 得到它的 RGB 数值
# OpenCV 默认读取的是 BGR 顺序，转换为 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("=" * 50)
print("图像基本信息：")
print(f"图像尺寸（高，宽，通道）：{img_rgb.shape}")
print(f"RGB 数值范围：{img_rgb.min()} ~ {img_rgb.max()}")
print(f"像素数据类型：{img_rgb.dtype}")
print("=" * 50)

# 3. 转换到 YCbCr 色彩空间
# OpenCV 使用 YCrCb，顺序是 Y、Cr、Cb
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# 分离三个通道
Y, Cr, Cb = cv2.split(img_ycrcb)

# 输出各通道统计信息
print("\nYCbCr 通道统计：")
print(f"Y 通道 (亮度):   min={Y.min()}, max={Y.max()}, mean={Y.mean():.2f}")
print(f"Cb 通道 (色度):  min={Cb.min()}, max={Cb.max()}, mean={Cb.mean():.2f}")
print(f"Cr 通道 (色度):  min={Cr.min()}, max={Cr.max()}, mean={Cr.mean():.2f}")
print("=" * 50)

# 4. 分别显示 Y、Cb、Cr 三个通道
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(Y, cmap='gray')
plt.title('Y Channel (Luminance)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(Cb, cmap='gray')
plt.title('Cb Channel')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(Cr, cmap='gray')
plt.title('Cr Channel')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ 所有任务完成！")