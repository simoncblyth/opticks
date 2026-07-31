#!/usr/bin/env python
"""
G4CXTest_raindrop_rainbow.py
============================

Gemini Prompt::

    I am using an Opticks optical photon simulation of a sphere of water and with
    millions of photons with wavelengths picked from Planck 6500 K black body
    distribution. I select photons with histories corresponding to a single bounce
    and examine the deviation angle.  Looking in an angular range close to the
    first bow angle I will get distributions of wavelengths. How should I combine
    all the wavelengths to yield a visual output like a rainbow ?


To convert your distribution of simulated optical photon wavelengths into a
physically accurate, visual rainbow, you need to aggregate the wavelengths in
each angular bin into a spectral power distribution (SPD), project that
spectrum into CIE XYZ space, and finally convert to sRGB for rendering.

I seek clarification on the normalization
------------------------------------------

::

    So this : X = S_lambda_theta @ cmf_x : is summing up X contributions from all
    wavelength bins - giving a very non-normalized X ?


Yes, exactly. S_lambda_theta @ cmf_x performs an unweighted sum across all
wavelength bins, yielding an absolute, unnormalized X value for each angular
bin θ.

Because X, Y, and Z scale linearly with the total physical photon intensity
(radiance), your raw values scale directly with the total number of photons
simulated and your choice of bin width (Δλ).

In physical optics, keeping X, Y, and Z unnormalized until the end is crucial
because Y carries absolute visual brightness (luminance).
If you normalized X, Y, and Z for every angular bin individually (e.g. dividing
by X+Y+Z), every single angular bin—even dark regions like Alexander's dark
band—would be stretched to 100% full brightness, destroying the intensity
profile of the rainbow.

Instead of normalizing X, Y, and Z individually per bin, normalization happens
globally across the whole angular spectrum when converting to sRGB.




Gemini points out:
    CIE color matching functions expect energy radiance S(λ) (Watts),
    not raw photon count N(λ).

    Since each photon carries energy E=hc/lambda - a blue photon (400 nm)
    carries almost twice the energy of a red photon (700 nm).



"""


from opticks.sysrap.sevt import SEvt

import os, numpy as np
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])



# -------------------------------------------------------------
# 1. Analytic Fit for CIE 1931 CMFs (Wyman et al. / scie.h)
# -------------------------------------------------------------
def cie_xyz_cmf(wave):
    """Returns (x, y, z) CMFs for input wavelength in nm."""
    # xFit
    t1 = (wave - 442.0) * np.where(wave < 442.0, 0.0624, 0.0374)
    t2 = (wave - 599.8) * np.where(wave < 599.8, 0.0264, 0.0323)
    t3 = (wave - 501.1) * np.where(wave < 501.1, 0.0490, 0.0382)
    x = 0.362 * np.exp(-0.5 * t1**2) + 1.056 * np.exp(-0.5 * t2**2) - 0.065 * np.exp(-0.5 * t3**2)

    # yFit
    t1 = (wave - 568.8) * np.where(wave < 568.8, 0.0213, 0.0247)
    t2 = (wave - 530.9) * np.where(wave < 530.9, 0.0613, 0.0322)
    y = 0.821 * np.exp(-0.5 * t1**2) + 0.286 * np.exp(-0.5 * t2**2)

    # zFit
    t1 = (wave - 437.0) * np.where(wave < 437.0, 0.0845, 0.0278)
    t2 = (wave - 459.0) * np.where(wave < 459.0, 0.0385, 0.0725)
    z = 1.217 * np.exp(-0.5 * t1**2) + 0.681 * np.exp(-0.5 * t2**2)

    return x, y, z

# -------------------------------------------------------------
# 2. XYZ to sRGB Conversion Matrix (D65 White Point)
# -------------------------------------------------------------
XYZ_TO_sRGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])

def xyz_to_srgb(XYZ, normalize=True):
    """Converts (N, 3) XYZ array to non-linear sRGB in [0, 1]."""
    # Matrix multiply to Linear sRGB
    rgb_linear = XYZ @ XYZ_TO_sRGB.T

    # Simple gamut mapping: Desaturate negative values by shifting baseline
    min_vals = np.minimum(0.0, np.min(rgb_linear, axis=-1, keepdims=True))
    rgb_linear -= min_vals

    # Normalize brightness relative to maximum intensity across the rainbow
    max_peak = np.max(rgb_linear)

    if normalize and max_peak > 0:
        rgb_linear /= max_peak # Scales the primary bow peak to 1.0, dark band stays dark

    # Apply sRGB Gamma Transfer Function
    mask = rgb_linear <= 0.0031308
    rgb = np.zeros_like(rgb_linear)
    rgb[mask] = 12.92 * rgb_linear[mask]
    rgb[~mask] = 1.055 * np.power(np.maximum(0.0, rgb_linear[~mask]), 1.0 / 2.4) - 0.055

    return np.clip(rgb, 0.0, 1.0)

# -------------------------------------------------------------
# 3. Process Opticks Simulated Photons
# -------------------------------------------------------------
# Simulated mock Opticks data: arrays of angles (degrees) and wavelengths (nm)
# Replace these with your actual Opticks GPU output arrays:
# angles = opticks_hits['deviation_angle']
# wavelengths = opticks_hits['wavelength']



def make_wav_angle_histogram(wav, deg, num_angle_bins = 100, num_wave_bins = 400):
    # Binned setup
    angle_edges = np.linspace(137.4, 139.0, num_angle_bins + 1)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    wave_edges = np.linspace(380.0, 780.0, num_wave_bins + 1)
    wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])

    # to make the unnormalized X,Y,Z sums not depend on wavelength bin width
    # need to scale by bin width
    delta_lambda = (wave_edges[-1] - wave_edges[0]) / (len(wave_edges) - 1)

    # Weight photon counts by energy (1 / lambda)
    # wave_centers in nm
    photon_energy_weight = 1.0 / wave_centers

    # 2D Histogram: Spectrum per angle bin S(angle, wavelength)
    S_lambda_theta0, _, _ = np.histogram2d(deg, wav, bins=[angle_edges, wave_edges])

    # Element-wise multiply columns by energy weight
    S_lambda_theta = S_lambda_theta0 * photon_energy_weight

    return S_lambda_theta, wave_centers, delta_lambda, angle_edges


def accumulate_XYZ(S_lambda_theta, wave_centers, delta_lambda):
    """
    NOTE HOW THIS X,Y,Z IS VERY UN-NORMALIZED
    NORMALIZATION IS DONE ONCE AT LAST MOMENT
    """

    # Evaluate CMFs at wavelength centers
    cmf_x, cmf_y, cmf_z = cie_xyz_cmf(wave_centers)

    # Compute XYZ for each angle bin
    # X(theta) = sum_lambda S(theta, lambda) * x_bar(lambda)
    X = (S_lambda_theta @ cmf_x) * delta_lambda
    Y = (S_lambda_theta @ cmf_y) * delta_lambda
    Z = (S_lambda_theta @ cmf_z) * delta_lambda

    XYZ = np.column_stack([X, Y, Z])
    return XYZ



def plot_rainbow_strip(RGB, angle_edges):

    # -------------------------------------------------------------
    # 4. Render the Visual Rainbow Strip
    # -------------------------------------------------------------
    # Create 2D strip by repeating the 1D angular colors vertically
    rainbow_strip = np.tile(RGB[np.newaxis, :, :], (50, 1, 1))

    plt.figure(figsize=(10, 3))
    plt.imshow(rainbow_strip, extent=[angle_edges[0], angle_edges[-1], 0, 1])
    plt.xlabel("Deviation Angle (degrees)")
    plt.yticks([])
    plt.title("Simulated Rainbow Spectrum from Opticks Photons")
    plt.tight_layout()
    plt.show()



class Rainbow:
    """
    salvage from ~/env/opticksnpy/rainbow.py
    """
    def __init__(self, a, k=1):
        """
        :param k: integer specifying number of internal reflections

        0  1  2  3  4
        TO BT BR BT SA

        """
        k_internal = "TO BT " + "BR " * k + "BT SA"
        a_sel = a.q_startswith(k_internal)
        ar = a.f.record[a_sel]

        r0 = 0
        r1 = k+3
        mom0 = ar[:,r0,1,:3]
        mom1 = ar[:,r1,1,:3]

        ct = np.sum(mom0 * mom1, axis=1)
        rad = np.arccos(np.clip(ct, -1.0, 1.0))
        deg = 180.*rad/np.pi







if __name__ == '__main__':

    a = SEvt.Load("$BFOLD", symbol="a")
    a_sel = a.q_startswith("TO BT BR BT SA")

    ar = a.f.record[a_sel]
    wav = ar[:,0,2,3]
    mom0 = ar[:,0,1,:3]
    mom1 = ar[:,4,1,:3]
    ct = np.sum(mom0 * mom1, axis=1)

    rad = np.arccos(np.clip(ct, -1.0, 1.0))
    deg = 180.*rad/np.pi


    if "ANGLE" in os.environ:
        bins = np.linspace( 137, 147, 100 )
        hn, hd = np.histogram(deg, bins)
        fig, ax = plt.subplots(1, figsize=SIZE/100 )
        ax.plot( hd[:-1], hn , drawstyle="steps-post", label="deg")
        fig.show()
    pass

    visible = np.logical_and( wav > 380., wav < 780. )

    vwav = wav[visible]
    vdeg = deg[visible]

    if "GEMINI" in os.environ:
        num_angle_bins = 10
        num_wave_bins = 10
        S_lambda_theta, wave_centers, delta_lambda, angle_edges = make_wav_angle_histogram(vwav, vdeg, num_angle_bins, num_wave_bins )
        XYZ = accumulate_XYZ(S_lambda_theta, wave_centers, delta_lambda)
        RGB = xyz_to_srgb(XYZ)  # Convert XYZ spectrum per bin into RGB colors
        plot_rainbow_strip(RGB, angle_edges)
    pass

    bins = np.linspace(137.4, 139, 10)

    wx,wy,wz = cie_xyz_cmf(vwav)

    hX, _ = np.histogram(vdeg,bins=bins, weights=wx)
    hY, _ = np.histogram(vdeg,bins=bins, weights=wy)
    hZ, _ = np.histogram(vdeg,bins=bins, weights=wz)

    hXYZ_raw = np.dstack([hX,hY,hZ])

    Ymax = hXYZ_raw[:,:,1].max()
    hXYZ = hXYZ_raw/Ymax   # normalize by maximum Y


    #hRGB_raw = hXYZ @ XYZ_TO_sRGB.T
    hRGB_raw = xyz_to_srgb(hXYZ, normalize=True)


    hRGB_1d = np.clip(hRGB_raw, 0, 1)

    ntile = 50
    hRGB = np.tile(hRGB_1d, ntile ).reshape(-1,ntile,3)

    extent = [0,2,bins[0],bins[-1]]

    #interpolation = 'none'
    #interpolation = 'mitchell'
    interpolation = 'gaussian'


    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(hRGB, origin="lower", extent=extent, alpha=1, vmin=0, vmax=1, aspect='auto', interpolation=interpolation)

    plt.show()








