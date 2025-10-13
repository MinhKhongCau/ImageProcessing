#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import grey_erosion, grey_dilation

# -----------------------------
# Utility: đọc ảnh .bin 256x256
# -----------------------------
def read_bin_image(fname, shape=(256, 256), dtype=np.uint8):
    with open(fname, 'rb') as f:
        arr = np.fromfile(f, dtype=dtype)
    return arr.reshape(shape)

# -----------------------------
# Median filter 3x3
# -----------------------------
def median_filter(img):
    h, w = img.shape
    output = np.zeros_like(img)
    pad = 1
    padded = np.pad(img, pad, mode='constant', constant_values=0)

    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            window = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            output[i - pad, j - pad] = np.median(window)
    return output

# -----------------------------
# Morphological operations
# -----------------------------
def morph_open(img):
    selem = np.ones((3, 3), dtype=np.uint8)
    eroded = grey_erosion(img, footprint=selem, mode='constant', cval=0)
    opened = grey_dilation(eroded, footprint=selem, mode='constant', cval=0)
    return opened

def morph_close(img):
    selem = np.ones((3, 3), dtype=np.uint8)
    dilated = grey_dilation(img, footprint=selem, mode='constant', cval=0)
    closed = grey_erosion(dilated, footprint=selem, mode='constant', cval=0)
    return closed

# -----------------------------
# Hiển thị ảnh
# -----------------------------
def show_images(title, imgs, names):
    plt.figure(figsize=(12, 4))
    for i, im in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.title(names[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    files = ['camera9.bin', 'camera99.bin']
    for fname in files:
        if not os.path.exists(fname):
            print(f"⚠️ File {fname} không tồn tại. Vui lòng tải từ trang khóa học.")
            continue

        img = read_bin_image(fname)
        median = median_filter(img)
        opened = morph_open(img)
        closed = morph_close(img)

        show_images(f"Results for {fname}",
                    [img, median, opened, closed],
                    ["Original", "Median", "Opening", "Closing"])

if __name__ == "__main__":
    main()
