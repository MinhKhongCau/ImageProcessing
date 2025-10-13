#   Obtain the image “actontBin.bin” from the course web site. This image has 256 256
#   pixels with 8 bits each. It is a true binary image; the pixel value 255 represents logical
#   one and the pixel value 0 represents logical zero.
#   Write a program to find instances of the letter “T” in the image using the Binary
#   Template Matching algorithm given on pages 2.92 - 2.97 of the notes. You will have
#   to design the template yourself based on an analysis of the image. Apply the match
#   measure M2 at every pixel in the input image where a sufficiently large neighborhood
#   exists. Construct an output image J1 where each pixel is equal to the match measure
#   M2 (set J1 equal to zero at pixels where a sufficiently large neighborhood does not
#   exist in the input image).
#   Threshold the image J1 to obtain a binary image J2 that should be equal to logical one
#   at pixels where there is a high probability that the letter “T” is present in the input
#   image.

import cv2
import numpy as np

# Read binary image
filename = "actontBin.bin"
img = np.fromfile(filename, dtype=np.uint8).reshape((256, 256))

# Ensure binary: 0 or 255
img = np.where(img > 0, 1, 0).astype(np.uint8)  # convert 255->1

# Example T template 15x15
T_template = np.zeros((15,15), dtype=np.uint8)
T_template[0:3,:] = 1          # horizontal top bar
T_template[:,6:9] = 1          # vertical stem in center


h, w = img.shape
th, tw = T_template.shape

# Initialize output image J1
J1 = np.zeros_like(img, dtype=np.float32)

# Slide template over image
for i in range(h - th + 1):
    for j in range(w - tw + 1):
        region = img[i:i+th, j:j+tw]
        xor = np.bitwise_xor(region, T_template)  # XOR
        M2 = np.sum(xor) / (th*tw)                # fraction of mismatch
        J1[i + th//2, j + tw//2] = 1 - M2        # similarity measure

# Threshold: pick value empirically
threshold = 0.8  # similarity > 0.8
J2 = np.where(J1 >= threshold, 255, 0).astype(np.uint8)

# Save results
cv2.imshow("M2_Output.png", (J1*255))  # visualize match measure
cv2.waitKey(0)

cv2.imshow("J2_T_Detection.png", J2)
cv2.waitKey(0)

img_display = (img*255).astype(np.uint8).copy()
img_display[J2==255] = 128  # mark detected T’s in gray
cv2.imshow("Detected_Ts.png", img_display)
cv2.waitKey(0)

cv2.destroyAllWindows()
