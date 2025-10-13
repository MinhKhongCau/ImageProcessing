import numpy as np
import matplotlib.pyplot as plt
import os
def load_bin_image(path, shape=(256,256)):
    return np.fromfile(path, dtype=np.uint8).reshape(shape)
path = "bins/"

for fname in ['camera.bin','salesman.bin','head.bin','eyeR.bin']:
    try:
        img = load_bin_image(os.path.join(path, fname))
    except:
        print(f"Không tìm thấy {fname}, bỏ qua.")
        continue

    plt.figure(); plt.imshow(img, cmap='gray'); plt.title(f"Original {fname}")

    F = np.fft.fftshift(np.fft.fft2(img))
    plt.figure(); plt.imshow(np.real(F), cmap='gray'); plt.title(f"Re DFT {fname}")
    plt.figure(); plt.imshow(np.imag(F), cmap='gray'); plt.title(f"Im DFT {fname}")
    plt.figure(); plt.imshow(np.log(1+np.abs(F)), cmap='gray'); plt.title(f"Log Magnitude {fname}")
    plt.figure(); plt.imshow(np.angle(F), cmap='gray'); plt.title(f"Phase {fname}")

plt.show()
