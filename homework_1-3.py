import cv2
import numpy as np

# (b) Read image
J1 = cv2.imread("lena512color.jpg", cv2.IMREAD_COLOR)
if J1 is None:
    raise FileNotFoundError("Image not found. Place lena512color.jpg in this folder.")

# Display original
cv2.imshow("Original J1", J1)

# (c) Create J2 with swapped channels
J2 = np.zeros_like(J1)
J2[:,:,2] = J1[:,:,0]   # Red <- Blue
J2[:,:,1] = J1[:,:,2]   # Green <- Red
J2[:,:,0] = J1[:,:,1]   # Blue <- Green

# (d) Display modified image
cv2.imshow("Modified J2 (Swapped Bands)", J2)

# Save result
cv2.imwrite("lena512color_swapped.jpg", J2)

cv2.waitKey(0)
cv2.destroyAllWindows()
