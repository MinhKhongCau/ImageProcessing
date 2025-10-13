import cv2
import numpy as np

# (c) Read the grayscale image
J1 = cv2.imread("lenagray.jpg", cv2.IMREAD_GRAYSCALE)

if J1 is None:
    raise FileNotFoundError("Image not found. Make sure lenagray.jpg is in your folder.")

# Show original
cv2.imshow("Original J1", J1)

# (d) Create negative
J2 = 255 - J1

# Show negative
cv2.imshow("Negative J2", J2)

# Save negative
cv2.imwrite("lenagray_negative.jpg", J2)

cv2.waitKey(0)
cv2.destroyAllWindows()
