
import numpy as np
import matplotlib.pyplot as plt

N = 8
u0, v0 = 2, 2
rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

I4 = np.sin(2 * np.pi / N * (u0*cols + v0*rows))

plt.imshow(I4, cmap='gray')
plt.title('I4 = sin(...)')
plt.colorbar()
plt.show()

F4 = np.fft.fftshift(np.fft.fft2(I4))

print("Re[DFT(I4)]:")
print(np.round(np.real(F4),4))
print("Im[DFT(I4)]:")
print(np.round(np.imag(F4),4))
