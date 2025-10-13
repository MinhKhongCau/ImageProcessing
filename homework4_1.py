
"""
HW: 7x7 average filter on salesman.bin — methods (a), (b), (c)
Requirements: numpy, scipy, matplotlib
Save this as hw05_salesman.py and run in folder containing salesman.bin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.signal import convolve2d
import os

# --- utility functions ---
def show_im(title, im, cmap='gray'):
    plt.figure(figsize=(5,5))
    plt.imshow(im, cmap=cmap, vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_spectrum(title, complex_ft):
    # centered log-magnitude spectrum, scaled for display
    mag = np.log(1 + np.abs(np.fft.fftshift(complex_ft)))
    # stretch to full 0..255
    mmin, mmax = mag.min(), mag.max()
    disp = 255 * (mag - mmin) / (mmax - mmin) if mmax>mmin else mag*0
    show_im(title, disp)

def fullscale_cast_and_round(im):
    # linearly map array to 0..255 then round and clamp to uint8
    a = im.copy().astype(np.float64)
    amin, amax = a.min(), a.max()
    if amax==amin:
        scaled = np.clip(np.round(a), 0, 255)
    else:
        scaled = np.round(255 * (a - amin) / (amax - amin))
        scaled = np.clip(scaled, 0, 255)
    return scaled.astype(np.uint8)

# --- read salesman.bin ---
fname = "bins/salesman.bin"
if not os.path.exists(fname):
    raise FileNotFoundError(f"File '{fname}' not found in current directory. Place salesperson.bin here.")

# file is 256x256, 8-bit grayscale
X = np.fromfile(fname, dtype=np.uint8)
if X.size != 256*256:
    raise ValueError(f"Expected 256*256 = 65536 bytes but found {X.size} bytes in {fname}")
X = X.reshape((256,256))
Xf = X.astype(np.float64)

# display original
print(X.shape)
show_im("Original input image (full scale)", X)

# -------------------------
# (a) Image-domain convolution (zero-padding)
# -------------------------
k = np.ones((7,7), dtype=np.float64) / 49.0

# Method: pad image with 3 zeros on each side and perform valid convolution sliding
pad = 3
Xpad = np.pad(Xf, pad_width=pad, mode='constant', constant_values=0.0)  # 262 x 262
# Now convolve using straightforward sliding (we'll use scipy convolve2d on padded image)
# But be careful: convolve2d with mode='valid' with kernel center at kernel center yields desired mapping
Y_a_full = convolve2d(Xpad, k, mode='valid')  # result size = 256 x 256 (since Xpad 262 convolved with 7 -> 256)
# Y_a_full should now be the filtered 256x256 result
Y1a = Y_a_full.copy()
show_im("(a) Output image from image-domain 7x7 average (full scale)", fullscale_cast_and_round(Y1a))
plt.close('all')

# -------------------------
# (b) DFT pointwise multiplication (Example 3 style)
# Zero-padded sizes: input X 256x256, H image size 128x128, pad size = 256 + 128 - 1 = 383
# per assignment, use 384x384 (even).
# H (128x128) has a 7x7 square centered with center (p0,q0)=(64,64) (we place 7x7 centered at index 64)
# We'll follow the MATLAB/assignment convention and then zero-pad both into 384x384 and multiply
# -------------------------
# Make H128 with 7x7 centered at p0=64 (0-based)
H128 = np.zeros((128,128), dtype=np.float64)

# center index p0=64 (0-based as in assignment). Put 7x7 block indices 61..67 inclusive
c = 64
r0 = c - 3
r1 = c + 3
H128[r0:r1+1, r0:r1+1] = 1.0 / 49.0  # 7x7 block

# Zero-pad input and H to Pad = 384 x 384
Pad = 384
ZPX = np.zeros((Pad, Pad), dtype=np.float64)
ZPX[0:256, 0:256] = Xf  # top-left placement as in example

ZPH = np.zeros((Pad, Pad), dtype=np.float64)
ZPH[0:128, 0:128] = H128  # top-left placement

# compute centered DFT log-magnitude displays:
Xtilde = np.fft.fft2(ZPX)
Htilde = np.fft.fft2(ZPH)
Ytilde = Xtilde * Htilde
ZPY = np.fft.ifft2(Ytilde).real  # zero-padded output, real part

# Display requested images/spectra
show_im("(b) Zero padded original image (top-left placed)", fullscale_cast_and_round(ZPX))
show_im("(b) Zero padded impulse response image (top-left placed)", fullscale_cast_and_round(ZPH))
show_spectrum("(b) Centered DFT log-magnitude of zero padded input", Xtilde)
show_spectrum("(b) Centered DFT log-magnitude of zero padded impulse response", Htilde)
show_spectrum("(b) Centered DFT log-magnitude of zero padded output", Ytilde)
show_im("(b) Zero padded output image (real part)", fullscale_cast_and_round(ZPY))

# Crop final 256x256 output image:
# Assignment/MATLAB used Y = ZPY(65:320,65:320) which is MATLAB 1-based.
# Converting: python 0-based slice 64:320
Y1b = ZPY[64:320, 64:320].copy()  # should be 256x256
show_im("(b) Final 256x256 output image (from DFT multiplication)", fullscale_cast_and_round(Y1b))
plt.close('all')

# Compare (a) and (b)
diff_ab = np.abs(fullscale_cast_and_round(Y1b).astype(int) - fullscale_cast_and_round(Y1a).astype(int))
print("(b): max difference from part (a):", diff_ab.max())

# -------------------------
# (c) Zero-phase impulse response and DFT method (Example 5 style)
# - Create H 256x256 with 7x7 square centered at (p0,q0) = (128,128) (they suggested H[126:132]=1/49)
# - Then fftshift to get zero-phase impulse response H2,
# - Zero-pad to 512x512 (Xpad 512 with X at top-left 0:256)
# - Multiply FFTs and crop
# -------------------------
H = np.zeros((256,256), dtype=np.float64)
# As assignment suggests: set python indices 126:132 inclusive (that's 126..132 -> 7 elements)
# Python slice 126:133
H[126:133, 126:133] = 1.0/49.0

# Zero-phase impulse response
H2 = np.fft.fftshift(H)

show_im("(c) 256x256 zero-phase impulse response image (h2)", fullscale_cast_and_round(H2))

# Zero-pad to 512x512
PadC = 512
ZPXc = np.zeros((PadC, PadC), dtype=np.float64)
ZPXc[0:256, 0:256] = Xf

ZPH2 = np.zeros((PadC, PadC), dtype=np.float64)
ZPH2[0:256, 0:256] = H2  # place top-left as in example

# Compute FFTs and multiply
Xtildec = np.fft.fft2(ZPXc)
Htildec = np.fft.fft2(ZPH2)
Ytildec = Xtildec * Htildec
ZPYc = np.fft.ifft2(Ytildec).real

# Display zero-padded h2 and final result and crops
show_im("(c) 512x512 zero padded zero-phase impulse response (h2ZP)", fullscale_cast_and_round(ZPH2))
Y1c = ZPYc[0:256, 0:256].copy()  # example uses cropping to top-left 256x256
show_im("(c) Final 256x256 output image (zero-phase DFT method)", fullscale_cast_and_round(Y1c))
plt.close('all')

# Compare (a) and (c)
diff_ac = np.abs(fullscale_cast_and_round(Y1c).astype(int) - fullscale_cast_and_round(Y1a).astype(int))
