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


Historical version of rainbow plotting::

   ~/env/opticksnpy/rainbow.py


"""

from opticks.sysrap.sevt import SEvt
from opticks.ana.cie_xyz import cie_xyz_cmf, xyz_to_srgb

import os, numpy as np
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])


if __name__ == '__main__':

    a = SEvt.Load("$AFOLD", symbol="a")
    a_sel = a.q_startswith("TO BT BR BT SA")

    ar = a.f.record[a_sel]
    wav = ar[:,0,2,3]
    mom0 = ar[:,0,1,:3]
    mom1 = ar[:,4,1,:3]
    ct = np.sum(mom0 * mom1, axis=1)

    rad = np.arccos(np.clip(ct, -1.0, 1.0))
    deg = 180.*rad/np.pi


    bins = np.linspace(137.4, 139, 10)

    wx,wy,wz = cie_xyz_cmf(wav)

    hX, _ = np.histogram(deg,bins=bins, weights=wx)
    hY, _ = np.histogram(deg,bins=bins, weights=wy)
    hZ, _ = np.histogram(deg,bins=bins, weights=wz)

    hXYZ_raw = np.dstack([hX,hY,hZ])

    Ymax = hXYZ_raw[:,:,1].max()
    hXYZ = hXYZ_raw/Ymax   # normalize by maximum Y


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


