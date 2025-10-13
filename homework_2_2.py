#   Obtain the image “lady.bin” from the course web site. This is a 256 256 gray scale
#   image with 8-bit pixels. Plot a histogram for the image. Write a program to perform
#   a full-scale contrast stretch on the image and plot a histogram for the result


import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# (1) Read the raw binary image
# -----------------------------
filename = "lady.bin"
img = np.fromfile(filename, dtype=np.uint8)  # 8-bit grayscale
img = img.reshape((256, 256))

# Save original image to visualize
cv2.imwrite("LadyOriginal.png", img)

# -----------------------------
# (2) Plot original histogram
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(img.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Original Histogram")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# (3) Full-scale contrast stretch
# -----------------------------
# Full-scale contrast stretch: map min->0, max->255
min_val = np.min(img)
max_val = np.max(img)
contrast_stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Save contrast-stretched image
cv2.imwrite("LadyContrastStretched.png", contrast_stretched)

# -----------------------------
# (4) Plot histogram after stretching
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(contrast_stretched.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Histogram After Full-Scale Contrast Stretch")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()