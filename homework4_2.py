#!/usr/bin/env python3
# problem2_filters.py
# Full implementation for parts (a)-(d)

import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Utilities
# -------------------------
def read_bin_image(fname, shape=(256,256), dtype=np.uint8):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")
    arr = np.fromfile(fname, dtype=dtype)
    if arr.size != shape[0]*shape[1]:
        raise ValueError(f"{fname}: expected {shape[0]*shape[1]} bytes, got {arr.size}")
    return arr.reshape(shape).astype(np.float64)

def fullscale_display_im(im):
    """Return image scaled linearly to 0..255 for display (uint8)."""
    a = im.astype(np.float64)
    amin, amax = a.min(), a.max()
    if amax == amin:
        out = np.zeros_like(a)
    else:
        out = 255.0 * (a - amin) / (amax - amin)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def show(img, title="", cmap='gray'):
    plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

def log_mag_spectrum(fft_complex):
    # return scaled 0..255 uint8 image of centered log magnitude
    mag = np.log(1.0 + np.abs(np.fft.fftshift(fft_complex)))
    return fullscale_display_im(mag)

def mse(a, b):
    return np.mean((a - b)**2)

def isnr(orig, noisy, filtered):
    num = np.sum((orig - noisy)**2)
    den = np.sum((orig - filtered)**2)
    if den <= 0:
        return np.inf
    return 10.0 * np.log10(num / den)

# -------------------------
# Load images (part a)
# -------------------------
print("Reading images...")
f = read_bin_image('bins/girl2.bin')               # original
x_broad = read_bin_image('bins/girl2Noise32.bin')  # broadband noise
x_high = read_bin_image('bins/girl2Noise32Hi.bin') # hi-pass noise

# Display originals (full-scale)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
show(fullscale_display_im(f), "Original: girl2.bin")
plt.subplot(1,3,2)
show(fullscale_display_im(x_broad), "girl2Noise32 (broadband)")
plt.subplot(1,3,3)
show(fullscale_display_im(x_high), "girl2Noise32Hi (hi-pass)")
plt.tight_layout()
plt.show()

# Compute MSEs (use floating point images directly)
mse_broad = mse(f, x_broad)
mse_high = mse(f, x_high)
print(f"(a) MSE(girl2Noise32 vs girl2)  = {mse_broad:.6f}")
print(f"(a) MSE(girl2Noise32Hi vs girl2) = {mse_high:.6f}")

# -------------------------
# (b) Ideal isotropic LPF Ucutoff = 64 (circular convolution)
# Build Htilde as in assignment:
# [U,V] = meshgrid(-128:127,-128:127);
# HLtildeCenter = double(sqrt(U.^2 + V.^2) <= U_cutoff);
# HLtilde = fftshift(HLtildeCenter);
# -------------------------
N = 256
Ucutoff = 64.0
coords = np.arange(-128, 128)
Ugrid, Vgrid = np.meshgrid(coords, coords)
HLtildeCenter = ((Ugrid**2 + Vgrid**2) <= (Ucutoff**2)).astype(float)
HLtilde = np.fft.fftshift(HLtildeCenter)  # per assignment instruction

# For verification: show HLtildeCenter and HLtilde magnitude
plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
show(fullscale_display_im(HLtildeCenter), "HLtildeCenter (centered mask)")
plt.subplot(1,2,2)
show(fullscale_display_im(HLtilde), "HLtilde (fftshifted)")
plt.tight_layout()
plt.show()

# Apply circular convolution (pointwise multiply DFT) to all three images
def apply_pointwise_dft_filter(image, Htilde):
    X = np.fft.fft2(image)
    Y = X * Htilde
    y = np.fft.ifft2(Y).real
    return y, X, Y

y_f_ideal, Xf_ideal, Yf_ideal = apply_pointwise_dft_filter(f, HLtilde)
y_broad_ideal, Xb_ideal, Yb_ideal = apply_pointwise_dft_filter(x_broad, HLtilde)
y_high_ideal, Xh_ideal, Yh_ideal = apply_pointwise_dft_filter(x_high, HLtilde)

# Display filtered images (full-scale for viewing)
plt.figure(figsize=(12,6))
plt.subplot(2,3,1); show(fullscale_display_im(y_f_ideal), "IdealLPF on original (f)")
plt.subplot(2,3,2); show(fullscale_display_im(y_broad_ideal), "IdealLPF on broadband")
plt.subplot(2,3,3); show(fullscale_display_im(y_high_ideal), "IdealLPF on hi-pass")
# display spectra for original noisy image before/after (example)
plt.subplot(2,3,4); show(log_mag_spectrum(Xb_ideal), "Spectrum: x_broad (centered log-mag)")
plt.subplot(2,3,5); show(log_mag_spectrum(Yb_ideal), "Spectrum after multiply (Yb)")
plt.subplot(2,3,6); show(log_mag_spectrum(np.fft.fft2(y_broad_ideal)), "FFT of filtered image")
plt.tight_layout()
plt.show()

# Compute MSEs (use floating point result y)
mse_f_ideal = mse(f, y_f_ideal)
mse_broad_ideal = mse(f, y_broad_ideal)
mse_high_ideal = mse(f, y_high_ideal)
print("\n(b) Ideal LPF results (circular conv):")
print(f" MSE(original vs filtered original)  = {mse_f_ideal:.6f}")
print(f" MSE(original vs filtered broadband) = {mse_broad_ideal:.6f}")
print(f" MSE(original vs filtered hi-pass)   = {mse_high_ideal:.6f}")

# ISNR for noisy images
isnr_broad_ideal = isnr(f, x_broad, y_broad_ideal)
isnr_high_ideal = isnr(f, x_high, y_high_ideal)
print(f" ISNR broadband (Ideal LPF) = {isnr_broad_ideal:.4f} dB")
print(f" ISNR highpass (Ideal LPF) = {isnr_high_ideal:.4f} dB")

# -------------------------
# (c) Gaussian LPF with Ucutoff = 64 using Example 4 procedure
# Steps:
#  - Sigma = 0.19 * N / Ucutoff
#  - build HtildeCenter (centered Gaussian), Htilde = fftshift(HtildeCenter)
#  - H = ifft2(Htilde), H2 = fftshift(H)  (zero-phase impulse response centered)
#  - ZPH2 is 512x512 with H2 in (0:256,0:256)
#  - zero-pad images to 512x512 top-left, take FFT, multiply, ifft2, crop 0:256,0:256
# -------------------------
def gaussian_zero_phase_impulse(N, Ucutoff):
    SigmaH = 0.19 * float(N) / float(Ucutoff)
    coords = np.arange(-N//2, N//2)
    Ugrid, Vgrid = np.meshgrid(coords, coords)
    HtildeCenter = np.exp((-(2.0 * (np.pi**2) * (SigmaH**2)) / (N**2)) * (Ugrid**2 + Vgrid**2))
    Htilde = np.fft.fftshift(HtildeCenter)  # as described
    H = np.fft.ifft2(Htilde)
    H2 = np.fft.fftshift(H)  # zero-phase impulse response (centered spatial)
    return H2, HtildeCenter

def apply_zero_phase_filter(image, H2, pad_size=512):
    # image: 256x256 float
    # H2: 256x256 zero-phase impulse response (complex)
    ZP_img = np.zeros((pad_size, pad_size), dtype=np.complex128)
    ZP_H2  = np.zeros((pad_size, pad_size), dtype=np.complex128)
    ZP_img[0:256, 0:256] = image
    ZP_H2[0:256, 0:256] = H2
    # multiply in freq domain
    Xzp = np.fft.fft2(ZP_img)
    Hzp = np.fft.fft2(ZP_H2)
    Yzp = Xzp * Hzp
    yzp = np.fft.ifft2(Yzp).real
    # crop top-left 256x256 (as in Example 4)
    return yzp[0:256, 0:256], Xzp, Hzp, Yzp, yzp

# compute H2 for Ucutoff=64
H2_gauss_64, HtildeCenter_64 = gaussian_zero_phase_impulse(N=256, Ucutoff=64.0)

# show H2 center portion magnitude
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
show(fullscale_display_im(np.real(H2_gauss_64)), "H2 (real part, zero-phase, Uc=64)")
plt.subplot(1,2,2)
show(fullscale_display_im(np.abs(HtildeCenter_64)), "HtildeCenter (gaussian, centered)")
plt.tight_layout(); plt.show()

# Apply to three images
y_f_gauss64, Xf_zp, Hf_zp, Yf_zp, full_yf_zp = apply_zero_phase_filter(f, H2_gauss_64, pad_size=512)
y_broad_gauss64, Xb_zp, Hb_zp, Yb_zp, full_yb_zp = apply_zero_phase_filter(x_broad, H2_gauss_64, pad_size=512)
y_high_gauss64, Xh_zp, Hh_zp, Yh_zp, full_yh_zp = apply_zero_phase_filter(x_high, H2_gauss_64, pad_size=512)

# Display
plt.figure(figsize=(12,6))
plt.subplot(2,3,1); show(fullscale_display_im(y_f_gauss64), "Gaussian Uc=64 on original")
plt.subplot(2,3,2); show(fullscale_display_im(y_broad_gauss64), "Gaussian Uc=64 on broadband")
plt.subplot(2,3,3); show(fullscale_display_im(y_high_gauss64), "Gaussian Uc=64 on hi-pass")
plt.subplot(2,3,4); show(log_mag_spectrum(np.fft.fft2(full_yb_zp)), "FFT of ZP-filtered broadband (512)")
plt.subplot(2,3,5); show(fullscale_display_im(np.abs(H2_gauss_64)), "H2 magnitude (center 256x256)")
plt.subplot(2,3,6); show(fullscale_display_im(np.real(y_broad_gauss64)), "Cropped result broadband")
plt.tight_layout(); plt.show()

# Compute MSE and ISNR (use floating point cropped results)
mse_f_gauss64 = mse(f, y_f_gauss64)
mse_broad_gauss64 = mse(f, y_broad_gauss64)
mse_high_gauss64 = mse(f, y_high_gauss64)
isnr_broad_gauss64 = isnr(f, x_broad, y_broad_gauss64)
isnr_high_gauss64 = isnr(f, x_high, y_high_gauss64)

print("\n(c) Gaussian LPF Uc=64 (Example 4):")
print(f" MSE(original vs filtered original)  = {mse_f_gauss64:.6f}")
print(f" MSE(original vs filtered broadband) = {mse_broad_gauss64:.6f}")
print(f" MSE(original vs filtered hi-pass)   = {mse_high_gauss64:.6f}")
print(f" ISNR broadband (Gauss Uc=64) = {isnr_broad_gauss64:.4f} dB")
print(f" ISNR highpass (Gauss Uc=64) = {isnr_high_gauss64:.4f} dB")

# -------------------------
# (d) Gaussian LPF with Ucutoff = 77.5 (same method)
# -------------------------
H2_gauss_77, HtildeCenter_77 = gaussian_zero_phase_impulse(N=256, Ucutoff=77.5)

# Apply
y_f_gauss77, _, _, _, _ = apply_zero_phase_filter(f, H2_gauss_77, pad_size=512)
y_broad_gauss77, _, _, _, _ = apply_zero_phase_filter(x_broad, H2_gauss_77, pad_size=512)
y_high_gauss77, _, _, _, _ = apply_zero_phase_filter(x_high, H2_gauss_77, pad_size=512)

# Display
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); show(fullscale_display_im(y_f_gauss77), "Gaussian Uc=77.5 on original")
plt.subplot(1,3,2); show(fullscale_display_im(y_broad_gauss77), "Gaussian Uc=77.5 on broadband")
plt.subplot(1,3,3); show(fullscale_display_im(y_high_gauss77), "Gaussian Uc=77.5 on hi-pass")
plt.tight_layout(); plt.show()

# Compute MSE and ISNR
mse_f_gauss77 = mse(f, y_f_gauss77)
mse_broad_gauss77 = mse(f, y_broad_gauss77)
mse_high_gauss77 = mse(f, y_high_gauss77)
isnr_broad_gauss77 = isnr(f, x_broad, y_broad_gauss77)
isnr_high_gauss77 = isnr(f, x_high, y_high_gauss77)

print("\n(d) Gaussian LPF Uc=77.5 (Example 4):")
print(f" MSE(original vs filtered original)  = {mse_f_gauss77:.6f}")
print(f" MSE(original vs filtered broadband) = {mse_broad_gauss77:.6f}")
print(f" MSE(original vs filtered hi-pass)   = {mse_high_gauss77:.6f}")
print(f" ISNR broadband (Gauss Uc=77.5) = {isnr_broad_gauss77:.4f} dB")
print(f" ISNR highpass (Gauss Uc=77.5) = {isnr_high_gauss77:.4f} dB")

# -------------------------
# Summary printout of key numbers
# -------------------------
print("\n--- Summary ---")
print(f"Original MSEs (no filtering): broadband {mse_broad:.6f}, highpass {mse_high:.6f}")
print("Ideal LPF (Uc=64) MSEs:", f"{mse_f_ideal:.6f}, {mse_broad_ideal:.6f}, {mse_high_ideal:.6f}")
print("Gaussian Uc=64 MSEs:", f"{mse_f_gauss64:.6f}, {mse_broad_gauss64:.6f}, {mse_high_gauss64:.6f}")
print("Gaussian Uc=77.5 MSEs:", f"{mse_f_gauss77:.6f}, {mse_broad_gauss77:.6f}, {mse_high_gauss77:.6f}")
