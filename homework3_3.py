
import numpy as np
import matplotlib.pyplot as plt

N = 8
u0, v0 = 2, 2
rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

I3 = np.cos(2 * np.pi / N * (u0*cols + v0*rows))

plt.imshow(I3, cmap='gray')
plt.title('I3 = cos(...)')
plt.colorbar()
plt.show()

F3 = np.fft.fftshift(np.fft.fft2(I3))

print("Re[DFT(I3)]:")
print(np.round(np.real(F3),4))
print("Im[DFT(I3)]:")
print(np.round(np.imag(F3),4))
