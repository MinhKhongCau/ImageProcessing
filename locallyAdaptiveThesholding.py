import cv2
import numpy as np
import matplotlib.pyplot as plt

def locally_adaptive_threshold(img, window_size=32, var_threshold=500):
    """
    Locally adaptive thresholding with variance check and local Otsu.
    
    Parameters:
        img: grayscale image (numpy array)
        window_size: size of sliding window
        var_threshold: variance threshold to decide uniformity
    Returns:
        Binary image
    """
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.uint8)

    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)
            window = img[y:y_end, x:x_end]

            # compute variance
            variance = np.var(window)

            if variance < var_threshold:
                # uniform area → classify whole block based on mean
                mean_val = np.mean(window)
                if mean_val > 127:
                    out[y:y_end, x:x_end] = 255
                else:
                    out[y:y_end, x:x_end] = 0
            else:
                # non-uniform area → Otsu threshold
                _, local_thresh = cv2.threshold(
                    window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                out[y:y_end, x:x_end] = local_thresh

    return out


# === Example usage ===
# Load grayscale image
img = cv2.imread("johnny.bin", cv2.IMREAD_GRAYSCALE)  # hoặc ảnh khác
if img is None:
    # fallback demo image
    img = cv2.imread(cv2.samples.findFile("sudoku.png"), cv2.IMREAD_GRAYSCALE)

result = locally_adaptive_threshold(img, window_size=32, var_threshold=500)

# Show
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Locally Adaptive Threshold")
plt.imshow(result, cmap='gray')
plt.axis("off")
plt.show()

