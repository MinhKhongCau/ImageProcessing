
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("TkAgg")

# ==== 1. Đọc ảnh nhị phân từ file ====
# Đổi đường dẫn thành ảnh bạn có trên máy, ví dụ "circles.png"
img = cv2.imread("circles.png", cv2.IMREAD_GRAYSCALE)

# Nếu ảnh không phải nhị phân, ta threshold lại
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# ==== 2. Structuring elements hình đĩa ====
se15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
se35 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
se48 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48,48))

erosion15 = cv2.erode(img_bin, se15)
erosion35 = cv2.erode(img_bin, se35)
erosion48 = cv2.erode(img_bin, se48)

# ==== 3. Duality: erosion = dilation(background) ====
background = cv2.bitwise_not(img_bin)
dilate_bg = cv2.dilate(background, se15)
duality_erosion = cv2.bitwise_not(dilate_bg)

# ==== 4. Erosion không phải inverse của dilation ====
dilate_then_erode = cv2.erode(cv2.dilate(img_bin, se15), se15)
erode_then_dilate = cv2.dilate(cv2.erode(img_bin, se15), se15)

# ==== 5. Hiển thị kết quả ====
plt.figure(figsize=(12,8))

plt.subplot(2,3,1); plt.imshow(img_bin, cmap='gray'); plt.title("Original Binary")
plt.subplot(2,3,2); plt.imshow(erosion15, cmap='gray'); plt.title("Erosion SE=15")
plt.subplot(2,3,3); plt.imshow(erosion35, cmap='gray'); plt.title("Erosion SE=35")
plt.subplot(2,3,4); plt.imshow(erosion48, cmap='gray'); plt.title("Erosion SE=48")

plt.subplot(2,3,5); plt.imshow(duality_erosion, cmap='gray')
plt.title("Duality: erosion = dilation(bg)")

plt.subplot(2,3,6)
plt.imshow(np.hstack([dilate_then_erode, erode_then_dilate]), cmap='gray')
plt.title("Not inverse: Dilate→Erode vs Erode→Dilate")

plt.tight_layout()
plt.show()
