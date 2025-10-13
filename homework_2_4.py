#   Obtain the image “johnny.bin” from the course web site. This image has 256 256
#   pixels. Each pixel has 8 bits. Plot the histogram of the original image. Write a program
#   to perform histogram equalization on this image. Show the equalized image and plot
#   its histogram.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# (1) Load the binary image
# -----------------------------
filename = "johnny.bin"
img = np.fromfile(filename, dtype=np.uint8).reshape((256, 256))

# Save original image to visualize
cv2.imwrite("JohnnyOriginal.png", img)

# -----------------------------
# (2) Plot original histogram
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(img.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Original Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig("JohnnyOriginalHistogram.png")  # save to file
plt.close()

# -----------------------------
# (3) Histogram equalization
# -----------------------------
equalized = cv2.equalizeHist(img)
cv2.imshow("JohnnyEqualized.png", equalized)
cv2.waitKey(0)

# -----------------------------
# (4) Plot histogram after equalization
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(equalized.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Histogram After Equalization")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig("JohnnyEqualizedHistogram.png")
plt.close()

cv2.destroyAllWindows()
