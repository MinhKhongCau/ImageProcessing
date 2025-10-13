
    import numpy as np
    import matplotlib.pyplot as plt

    def load_bin_image(path, shape=(256,256)):
        return np.fromfile(path, dtype=np.uint8).reshape(shape)

    fname = 'bins/camera.bin'
    try:
        I6 = load_bin_image(fname)
    except:
        raise FileNotFoundError("Cần có file camera.bin trong thư mục chạy")

    F = np.fft.fft2(I6)

    # J1: giữ magnitude, phase=0
    J1 = np.abs(F)
    JJ1 = np.log(1+J1)

    # J2: magnitude=1, giữ nguyên phase
    J2 = np.exp(1j*np.angle(F))
    J2 = np.real(np.fft.ifft2(J2))

    plt.figure(); plt.imshow(J2, cmap='gray'); plt.title("J2 (phase only)")
    plt.figure(); plt.imshow(JJ1, cmap='gray'); plt.title("log(J1) (magnitude only)")

    plt.show()