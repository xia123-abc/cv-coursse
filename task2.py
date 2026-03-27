import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入彩色图像
img = cv2.imread('color.jpg')
if img is None:
    print("错误：找不到图片，请确保 color.jpg 存在")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
print(f"原始图像尺寸: {h} x {w}")

# 2. 转换到 YCbCr 色彩空间
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)

# 3. 对 Cb、Cr 通道进行下采样（缩小到 1/2）
scale = 2  # 下采样比例
Cb_down = cv2.resize(Cb, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
Cr_down = cv2.resize(Cr, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
print(f"下采样后 Cb/Cr 尺寸: {Cb_down.shape}")

# 4. 用插值方法恢复原尺寸
# 使用双线性插值放大回原尺寸
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 5. 重建图像：用原 Y 通道 + 恢复后的 Cb、Cr
img_reconstructed_ycrcb = cv2.merge([Y, Cr_up, Cb_up])
img_reconstructed_bgr = cv2.cvtColor(img_reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
img_reconstructed_rgb = cv2.cvtColor(img_reconstructed_bgr, cv2.COLOR_BGR2RGB)

# 6. 计算 PSNR
def psnr(img1, img2):
    """计算峰值信噪比"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

psnr_value = psnr(img_rgb, img_reconstructed_rgb)
print(f"\nPSNR (峰值信噪比): {psnr_value:.2f} dB")

# 7. 显示对比结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title(f'原图\n尺寸: {h}x{w}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_reconstructed_rgb)
plt.title(f'重建图像 (Cb/Cr 下采样 {scale}x)\nPSNR = {psnr_value:.2f} dB')
plt.axis('off')

# 显示差异图
diff = np.abs(img_rgb.astype(np.float64) - img_reconstructed_rgb.astype(np.float64))
plt.subplot(1, 3, 3)
plt.imshow(diff.astype(np.uint8))
plt.title(f'差异图 (放大显示误差)\n平均差异 = {np.mean(diff):.2f}')
plt.axis('off')

plt.tight_layout()
plt.show()

# 8. 分析结果
print("\n" + "="*50)
print("结果分析：")
print("="*50)
print(f"1. 下采样比例: {scale}x (Cb/Cr 分辨率降为原来的 1/{scale})")
print(f"2. PSNR 值: {psnr_value:.2f} dB")
if psnr_value > 30:
    print("   → PSNR > 30 dB，重建质量较好，人眼难以察觉明显差异")
elif psnr_value > 20:
    print("   → 20 dB < PSNR < 30 dB，重建质量一般，有一定失真")
else:
    print("   → PSNR < 20 dB，重建质量较差，失真明显")
print(f"3. 文件大小对比:")
print(f"   - 原图尺寸: {w}x{h}")
print(f"   - Cb/Cr 下采样后数据量: {Cb_down.size * 2} 像素")
print(f"   - 原 Cb/Cr 数据量: {Cb.size * 2} 像素")
print(f"   - 压缩率: {(Cb_down.size * 2) / (Cb.size * 2) * 100:.1f}%")
print("\n分析：人眼对亮度信息(Y)敏感，对色度信息(Cb/Cr)不敏感。")
print("对 Cb/Cr 下采样可大幅减少数据量，同时保持较好的视觉质量。")
print("这就是 JPEG 压缩的基本原理！")