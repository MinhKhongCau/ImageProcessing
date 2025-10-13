import cv2
import numpy as np

# a. Read and display the images.
def readImage(path):
    img = np.fromfile(path, dtype=np.uint8)
    if img.size != 256*256:
        raise ValueError(f"Unexpected file size in {path}, got {img.size} pixels")
    img = img.reshape((256, 256))
    print(f"Loaded {path} -> shape: {img.shape}, dtype: {img.dtype}")
    return img

# b. Merge left half of Lena with right half of Peppers
def mergeHalfImage(image1, image2):
    J = np.zeros_like(image1)
    J[:, :128] = image1[:, :128]   # left half from Lena
    J[:, 128:] = image2[:, 128:]   # right half from Peppers
    return J

# c. Swap left/right halves
def swapping(image):
    K = np.zeros_like(image)
    K[:, :128] = image[:, 128:]
    K[:, 128:] = image[:, :128]
    return K

# --- MAIN ---
lena = readImage("bins/lena.bin")
peppers = readImage("bins/peppers.bin")

J = mergeHalfImage(lena, peppers)
K = swapping(J)

# Display
cv2.imshow("Lena", lena)
cv2.imshow("Peppers", peppers)
cv2.imshow("J (Lena left + Peppers right)", J)
cv2.imshow("K (Swapped halves of J)", K)

# Save results for your report
# cv2.imwrite("lena.png", lena)
# cv2.imwrite("peppers.png", peppers)
# cv2.imwrite("J.png", J)
# cv2.imwrite("K.png", K)

cv2.waitKey(0)
cv2.destroyAllWindows()



