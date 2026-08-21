#!/usr/bin/env python
"""
cie_xyz.py
===========
"""

import numpy as np

# -------------------------------------------------------------
# 1. Analytic Fit for CIE 1931 CMFs (Wyman et al. / scie.h)
# -------------------------------------------------------------
def cie_xyz_cmf(wave):
    """
    Returns (x, y, z) CMFs for input wavelength in nm.
    """
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



