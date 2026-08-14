#!/usr/bin/env python
"""
G4CXTest_raindrop_raindev.py
==============================


"""
import os
import numpy as np
import matplotlib.pyplot as plt

from opticks.sysrap.sevt import SEvt
from opticks.ana.cie_xyz import cie_xyz_cmf, xyz_to_srgb

def deviation_angle_360(p_in, p_out, side):
    """
    Computes unfolded deviation angle [0, 2*pi) relative to p_in,
    using 'side' to break top/bottom hemisphere symmetry.

    Note that because rays at Y>0 vs Y<0 behave symmetrically with rays bouncing
    around in opposite directions around the drop there is a tendency to have
    a symmetric deviation angle - which is hiding the physics. Using the side
    to break that symmetry avoids this.

    But still need to mask to incident Y<0 in order to avoid photons
    from the other side ? HMM should be able to fold ontop with a sign flip ?

    """
    # 1. Standard 3D angle theta in [0, pi]
    cos_theta = np.clip(np.sum(p_in * p_out, axis=1), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 2. Check alignment with 'side' vector: p_out . side
    dot_side = np.sum(p_out * side, axis=1)

    # 3. If exit is on the negative side, angle is (2*pi - theta)
    unfolded_rad = np.where(dot_side >= 0, theta, 2 * np.pi - theta)

    return np.degrees(unfolded_rad)


def deviation_angle_360_folded(p_in, p_out, pos0):
    ey = pos0[:,1]
    vx = p_out[:,0]
    vy = p_out[:,1]

    # 2. Virtual Reflection: Flip vy for bottom-half rays (y_entry < 0)
    #    This simulates what the trajectory WOULD be if the photon had hit the top half (+y)
    vy_folded = np.where(ey < 0, -vy, vy)

    # 3. Compute the unfolded 0 to 360 degree deviation angle on 100% of rays
    dev_360_folded = np.degrees(np.arctan2(-vy_folded, vx)) % 360.0

    return dev_360_folded



if __name__ == '__main__':
    a = SEvt.Load("$AFOLD", symbol="a")

    a_sel = a.q_find("SA")  # any history ending with surface absorb
    ap = a.f.photon[a_sel]
    ar = a.f.record[a_sel]

    wav = ap[:,2,3]

    pos0 = ar[:,0,0,:3]    # start position
    y0 = pos0[:,1]         # start position y

    mom0 = np.array([1,0,0])  # +X start momentum
    side = np.array([0,1,0])  # +Y reference direction
    mom1 = ap[:,1,:3]         # photon instead of record directly gives final direction

    deg = deviation_angle_360(mom0,mom1,side)
    #deg = deviation_angle_360_folded(mom0,mom1,pos0)

    mask = y0 < 0        # simple but statistics wasting approach
    wav_sel = wav[mask]
    deg_sel = deg[mask]

    u_wav = wav_sel
    u_deg = deg_sel


    num_bins = 720
    #num_bins = 360
    bins = np.linspace(0.0, 360.0, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # CMF Tristimulus response
    wx, wy, wz = cie_xyz_cmf(u_wav)

    # Convert Photon Count -> Energy Radiance (E ~ 1/lambda)
    energy_weight = 1.0 / u_wav
    wx_e = wx * energy_weight
    wy_e = wy * energy_weight
    wz_e = wz * energy_weight

    # Calculate absolute X, Y, Z tristimulus distributions
    hX, _ = np.histogram(u_deg, bins=bins, weights=wx_e)
    hY, _ = np.histogram(u_deg, bins=bins, weights=wy_e)
    hZ, _ = np.histogram(u_deg, bins=bins, weights=wz_e)

    # -------------------------------------------------------------------------
    # 2. Global Normalization & Color Conversion
    # -------------------------------------------------------------------------
    hXYZ_raw = np.dstack([hX, hY, hZ])  # Shape: (1, num_bins, 3)

    # Global scale by maximum Y across the entire angular domain
    Ymax = hXYZ_raw[:, :, 1].max()
    hXYZ_norm = hXYZ_raw / (Ymax if Ymax > 0 else 1.0)

    # Convert to sRGB (returns linear or gamma-corrected RGB depending on utility)
    hRGB_raw = xyz_to_srgb(hXYZ_norm, normalize=False) # Shape: (1,num_bins,3)
    hRGB = np.clip(hRGB_raw[0], 0.0, 1.0)              # Shape: (num_bins, 3)

    # -------------------------------------------------------------------------
    # 3. Plot Deviation Angle Histogram with Colored Under-Curve Fill
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Total Photon Intensity per bin
    counts, _ = np.histogram(u_deg, bins=bins)

    # Plot line outline
    ax.plot(bin_centers, counts, color='black', lw=1.2, label='Photon Count Distribution')

    # Color individual vertical bars/polygons under the curve

    if "NOFILL" in os.environ:
         print("NOFILL skip coloring")
    else:
        for i in range(num_bins):
            ax.fill_between(
                [bins[i], bins[i+1]],
                [0, 0],
                [counts[i], counts[i]],
                color=hRGB[i],
                edgecolor='none'
            )
    pass

    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=1)
    ax.set_yscale('log')
    ax.set_xlabel('Deviation Angle $\\theta$ (degrees)')
    ax.set_ylabel('Photon Count')
    ax.set_title('Opticks Rainbow Simulation: $0^\\circ$ to $360^\\circ$ Deviation Angle Spectrum')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
