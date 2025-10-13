
import numpy as np
import matplotlib.pyplot as plt

N = 8
u0, v0 = 2, 2
rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

I2 = np.exp(-1j * 2 * np.pi / N * (u0*cols + v0*rows))

plt.imshow(np.real(I2), cmap='gray')
plt.title('Real(I2)')
plt.colorbar()
plt.show()

plt.imshow(np.imag(I2), cmap='gray')
plt.title('Imag(I2)')
plt.colorbar()
plt.show()

F2 = np.fft.fftshift(np.fft.fft2(I2))

print("Re[DFT(I2)]:")
print(np.round(np.real(F2),4))
print("Im[DFT(I2)]:")
print(np.round(np.imag(F2),4))
