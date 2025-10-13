
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("TkAgg")

# Bài 1: I1(m,n) = exp(j*2π(u0*m + v0*n)), u0 = v0 = 2 cpi
N = 8
u0, v0 = 2, 2
rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

I1 = np.exp(1j * 2 * np.pi / N * (u0*cols + v0*rows))

# Hiển thị phần thực và ảo
plt.imshow(np.real(I1), cmap='gray')
plt.title('Real(I1)')
plt.colorbar()
plt.show()

plt.imshow(np.imag(I1), cmap='gray')
plt.title('Imag(I1)')
plt.colorbar()
plt.show()


# Tính DFT
F1 = np.fft.fftshift(np.fft.fft2(I1))

print("Re[DFT(I1)]:")
print(np.round(np.real(F1),4))
print("Im[DFT(I1)]:")
print(np.round(np.imag(F1),4))
