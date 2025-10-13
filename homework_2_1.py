# Obtain the image “Mammogram.bin” from the course web site. This image has 256
# 256 pixels. Each pixel has 8 bits. Note: the server is Unix; the filename is case
# sensitive. Do not make the mistake of getting the incorrect 512 512 file “mam-
# mogram.bin.”
# (a) There are two main regions in the input image: the imaged tissue and the dark
# background region on the left side of the image. Write a program to convert this
# gray scale image into a binary image by simple thresholding. In the binary image,
# use a value of 255 = 0xff for logical one and a value of 0 = 0x00 for logical zero.
# Select the threshold so that the binary image is equal to logical zero over the
# background region and logical one over the tissue.
# (b) Write a program to implement the Approximate Contour Image Generation al-
# gorithm given on page 2.104 of the notes. Your program should input the binary
# image and output a binary contour image. Run your program to generate an
# appoximate contour image from the binary image you obtained by thresholding
# Mammogram.bin.
# (c) Could a chain code be used to represent the main contour in your contour image?
# Why
import numpy as np
import cv2

# (a) Read the raw binary file
filename = "mammogram.bin"
img = np.fromfile(filename, dtype=np.uint32)  # read as 8-bit
img = img.reshape((256, 256))

# Choose a threshold
# You might want to inspect histogram to pick threshold
if img.dtype != np.uint8:
    img = ((img - img.min()) / (img.max() - img.min())*255).astype(np.uint8)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
# (b) Approximate contour
# Example: threshold around 50–80 works for many mammograms
threshold_value = 70
_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

# Show result
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(binary_img, kernel, iterations=1)
contour_img = cv2.subtract(binary_img, eroded)

cv2.imshow("Approximate Contour", contour_img)
cv2.waitKey(0)
cv2.imwrite("ContourMammogram.png", contour_img)
contour_img.tofile("ContourMammogram.bin")

cv2.destroyAllWindows()
