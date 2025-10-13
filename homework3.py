
"""
Homework 4 — Problems 1..7
Python script to generate images I1..I5 (8x8), compute centered 2D DFTs, show/save real/imag parts
and to process Problems 6-7 for 256x256 raw .bin images (camera.bin, salesman.bin, head.bin, eyeR.bin).

Usage examples:
  python3 homework4_solutions.py                # run Problems 1..5 and save outputs to ./output
  python3 homework4_solutions.py --bins ./bins  # also process .bin files in ./bins (expect 256x256 raw files)
  python3 homework4_solutions.py --outdir myout --bins ./bins

Requirements:
  - Python 3.8+ (script tested on Python 3.10+)
  - numpy
  - matplotlib
  - imageio (optional, used to write PNGs)

Outputs (default ./output):
  - PNG images of real/imag parts and DFT visualizations
  - TXT files with printed 8x8 arrays (real and imag of DFTs)

"""

import os
import argparse
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

np.set_printoptions(precision=4, suppress=True)

# ------- Helper utilities -------

def zero_based_meshgrid(N=8):
    cols, rows = np.meshgrid(np.arange(N), np.arange(N))
    return cols, rows


def to_8bpp_fullscale(img):
    """Scale arr to 0..255 and return uint8 image (full-scale contrast)."""
    a = np.asarray(img)
    mi = a.min()
    ma = a.max()
    if np.isclose(ma, mi):
        return np.zeros(a.shape, dtype=np.uint8)
    norm = (a - mi) / (ma - mi)
    return (norm * 255.0).astype(np.uint8)


def show_and_maybe_save(img8, title, outpath=None):
    plt.figure(figsize=(4,3))
    plt.imshow(img8, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(title)
    if outpath:
        if _HAS_IMAGEIO:
            imageio.imwrite(str(outpath), img8)
        else:
            plt.savefig(str(outpath), bbox_inches='tight', pad_inches=0)
    plt.close()


def print_array_to_file(arr, label, filehandle):
    filehandle.write(label + "\n")
    for r in arr:
        filehandle.write(' '.join(f"{v:9.4f}" for v in r) + "\n")
    filehandle.write("\n")

# ------- Problem-specific functions -------

N = 8
cols, rows = zero_based_meshgrid(N=N)


def make_complex_exponential(u0, v0, amplitude=1.0, sign=1):
    phase = 2.0 * np.pi / N * (u0 * cols + v0 * rows)
    return amplitude * np.exp(1j * sign * phase)


def centered_dft(I):
    return fftshift(fft2(I))


def process_small_image(I, name, outdir):
    """Show/save real/imag and compute/print centered DFT (8x8 arrays written to file)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Re = np.real(I)
    Im = np.imag(I)
    # display/save 8bpp images
    Re8 = to_8bpp_fullscale(Re)
    Im8 = to_8bpp_fullscale(Im)
    show_and_maybe_save(Re8, f"{name} - Real", outdir / f"{name}_real.png")
    show_and_maybe_save(Im8, f"{name} - Imag", outdir / f"{name}_imag.png")
    # centered DFT
    F = centered_dft(I)
    ReF = np.real(F)
    ImF = np.imag(F)
    # save txt arrays
    txtpath = outdir / f"{name}_DFT_arrays.txt"
    with open(txtpath, 'w') as fh:
        print_array_to_file(ReF, f"Re[DFT({name})]:", fh)
        print_array_to_file(ImF, f"Im[DFT({name})]:", fh)
    # also save visualizations of Re/Im of DFT (scaled to 8bpp)
    show_and_maybe_save(to_8bpp_fullscale(ReF), f"Re[DFT({name})] (scaled)", outdir / f"{name}_DFT_Re.png")
    show_and_maybe_save(to_8bpp_fullscale(ImF), f"Im[DFT({name})] (scaled)", outdir / f"{name}_DFT_Im.png")
    return F

# ------- Problems 1..5 runner -------

def run_problems_1_to_5(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Problem 1: I1 = exp(+j*2pi/8*(2*m + 2*n))
    I1 = make_complex_exponential(2.0, 2.0, amplitude=1.0, sign=+1)
    F1 = process_small_image(I1, "I1_exp_plus", outdir)
    # print raw I1 arrays to text
    with open(outdir / "I1_raw.txt", 'w') as fh:
        print_array_to_file(np.real(I1), "Re[I1]:", fh)
        print_array_to_file(np.imag(I1), "Im[I1]:", fh)

    # Problem 2: I2 = exp(-j*...)
    I2 = make_complex_exponential(2.0, 2.0, amplitude=1.0, sign=-1)
    F2 = process_small_image(I2, "I2_exp_minus", outdir)
    with open(outdir / "I2_raw.txt", 'w') as fh:
        print_array_to_file(np.real(I2), "Re[I2]:", fh)
        print_array_to_file(np.imag(I2), "Im[I2]:", fh)

    # Problem 3: I3 = cos(2pi/8*(2*m+2*n))
    phase = 2.0 * np.pi / N * (2.0 * cols + 2.0 * rows)
    I3 = np.cos(phase)
    F3 = process_small_image(I3, "I3_cos", outdir)
    with open(outdir / "I3_raw.txt", 'w') as fh:
        print_array_to_file(I3, "I3:", fh)

    # Problem 4: I4 = sin(...)
    I4 = np.sin(phase)
    F4 = process_small_image(I4, "I4_sin", outdir)
    with open(outdir / "I4_raw.txt", 'w') as fh:
        print_array_to_file(I4, "I4:", fh)

    # Problem 5: I5 with u=v=1.5 cpi
    I5 = np.cos(2.0 * np.pi / N * (1.5 * cols + 1.5 * rows))
    F5 = process_small_image(I5, "I5_cos_1p5", outdir)
    with open(outdir / "I5_raw.txt", 'w') as fh:
        print_array_to_file(I5, "I5:", fh)

    # short automatic discussion saved to a text file
    with open(outdir / "discussion_I5.txt", 'w') as fh:
        fh.write("I5 has non-integer frequency (1.5 cpi). On an 8x8 sampled grid this creates spatial beating/aliasing and the cosine does not tile evenly across the 8-sample periodicity. In the DFT, integer-frequency signals produce impulses at integer bins while non-integer frequencies spread energy across bins.\n")

    print(f"Problems 1-5 processed. Outputs in: {outdir}")

# ------- Problems 6 and 7 (256x256 .bin images) -------

def load_raw_gray(path, shape=(256,256), dtype=np.uint8):
    data = np.fromfile(str(path), dtype=dtype)
    if data.size != shape[0]*shape[1]:
        raise ValueError(f"File {path} has size {data.size}, expected {shape[0]*shape[1]}")
    data = data.reshape(shape)
    return data.astype(float)


def process_256_image(path, name, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    I = load_raw_gray(path, shape=(256,256))
    # save original preview
    orig8 = to_8bpp_fullscale(I)
    show_and_maybe_save(orig8, f"{name} - original", outdir / f"{name}_original.png")
    # compute centered DFT
    F = fftshift(fft2(I))
    ReF = np.real(F)
    ImF = np.imag(F)
    logmag = np.log(np.abs(F) + 1e-9)
    phase = np.angle(F)
    # save visualizations
    show_and_maybe_save(to_8bpp_fullscale(ReF), f"{name} - Re[DFT]", outdir / f"{name}_DFT_Re.png")
    show_and_maybe_save(to_8bpp_fullscale(ImF), f"{name} - Im[DFT]", outdir / f"{name}_DFT_Im.png")
    show_and_maybe_save(to_8bpp_fullscale(logmag), f"{name} - LogMag", outdir / f"{name}_DFT_LogMag.png")
    show_and_maybe_save(to_8bpp_fullscale(phase), f"{name} - Phase", outdir / f"{name}_DFT_Phase.png")
    # Save arrays as npy for inspection
    np.save(outdir / f"{name}_DFT_Re.npy", ReF)
    np.save(outdir / f"{name}_DFT_Im.npy", ImF)
    return I, F


def problem_6_and_7(bins_dir, outdir):
    bins_dir = Path(bins_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = ["camera.bin", "salesman.bin", "head.bin", "eyeR.bin"]
    found = []
    for nm in names:
        p = bins_dir / nm
        if p.exists():
            print(f"Processing {p}")
            I, F = process_256_image(p, nm.replace('.bin',''), outdir / nm.replace('.bin',''))
            found.append(nm)
        else:
            print(f"Not found: {p} (skipping)")
    # Problem 7: use camera.bin if present
    cam = bins_dir / 'camera.bin'
    if cam.exists():
        I6 = load_raw_gray(cam)
        F = fftshift(fft2(I6))
        mag = np.abs(F)
        arg = np.angle(F)
        J1_freq = mag * np.exp(1j * 0.0)
        J2_freq = (1.0) * np.exp(1j * arg)
        # inverse
        J1 = np.real(ifft2(ifftshift(J1_freq)))
        J2 = np.real(ifft2(ifftshift(J2_freq)))
        # save
        outp = outdir / 'camera_reconstructions'
        outp.mkdir(parents=True, exist_ok=True)
        show_and_maybe_save(to_8bpp_fullscale(J2), 'J2 (phase-only recon)', outp / 'J2_phase_only.png')
        show_and_maybe_save(to_8bpp_fullscale(np.log(J1 + 1e-9)), 'log(J1) (mag-only recon)', outp / 'logJ1_mag_only.png')
        np.save(outp / 'J1.npy', J1)
        np.save(outp / 'J2.npy', J2)
        print(f"Problem 7 reconstructions saved in {outp}")
    else:
        print("camera.bin not found; Problem 7 skipped.")

# ------- Main CLI -------

def main():
    p = argparse.ArgumentParser(description='Homework4 solver: Problems 1..7')
    p.add_argument('--outdir', default='output', help='output directory')
    p.add_argument('--bins', default=None, help='directory containing .bin files for Problems 6-7')
    args = p.parse_args()

    outdir = Path(args.outdir)
    run_problems_1_to_5(outdir / 'problems_1_5')

    if args.bins:
        problem_6_and_7(args.bins, outdir / 'problems_6_7')
    else:
        print('No --bins provided; Problems 6-7 not run. To run them, place camera.bin, salesman.bin, head.bin, eyeR.bin in a folder and pass --bins <folder>')

if __name__ == '__main__':
    main()
