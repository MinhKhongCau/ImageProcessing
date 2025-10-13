
import numpy as np
import matplotlib.pyplot as plt

N = 8
u1, v1 = 1.5, 1.5
rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

I5 = np.cos(2 * np.pi / N * (u1*cols + v1*rows))

plt.imshow(I5, cmap='gray')
plt.title('I5 = cos(...) with freq 1.5 cpi')
plt.colorbar()
plt.show()

F5 = np.fft.fftshift(np.fft.fft2(I5))

print("Re[DFT(I5)]:")
print(np.round(np.real(F5),4))
print("Im[DFT(I5)]:")
print(np.round(np.imag(F5),4))
