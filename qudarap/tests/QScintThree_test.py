#!/usr/bin/env python
"""

~/o/qudarap/tests/QScintThree_test.sh pdb

"""

import os, sys, numpy as np
from opticks.ana.fold import Fold
MODE = int(os.environ.get("MODE","0"))


def chi2(obs, ref, threshold=10):
    """
    obs: Opticks bin counts (q0[0])
    ref: Geant4 bin counts (u0[0])
    threshold: Min counts to include a bin
    """
    # Apply the same count cut you used for the ratio plot
    mask = (ref > threshold) & (obs > threshold)
    q = obs[mask]
    u = ref[mask]

    # Calculate Chi-squared components
    chi2_val = np.sum((q - u)**2 / (q + u))
    ndf = len(q) - 1  # Degrees of freedom

    chi2_per = chi2_val / ndf
    return chi2_val, ndf, chi2_per



if MODE in [-2,2,3]:
    import matplotlib.pyplot as plt
    from opticks.ana.pvplt import *
pass


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    wl = f.wl
    print("wl\n",wl)
    u4wl = f.u4wls
    print("u4wl\n",u4wl)

    COMP = os.environ.get("COMP","ABC")
    A = "A" in COMP
    B = "B" in COMP
    C = "C" in COMP


    #bins = np.arange(80, 800, 4)
    bins = np.arange(300, 600, 1)

    q0 = np.histogram( wl[0] , bins )
    q1 = np.histogram( wl[1] , bins )
    q2 = np.histogram( wl[2] , bins )

    u0 = np.histogram( u4wl[0] , bins )
    u1 = np.histogram( u4wl[1] , bins )
    u2 = np.histogram( u4wl[2] , bins )


    threshold = 100
    mask0 = (q0[0] > threshold) & (u0[0] > threshold)
    mask1 = (q1[0] > threshold) & (u1[0] > threshold)
    mask2 = (q2[0] > threshold) & (u2[0] > threshold)

    # --- Bottom Plot (The Ratio or Difference) ---
    # Avoid division by zero by using where mask
    ratio0 = np.divide(q0[0]-u0[0], u0[0], out=np.ones_like(q0[0], dtype=float), where=mask0)
    ratio1 = np.divide(q1[0]-u1[0], u1[0], out=np.ones_like(q1[0], dtype=float), where=mask1)
    ratio2 = np.divide(q2[0]-u2[0], u2[0], out=np.ones_like(q2[0], dtype=float), where=mask2)


    # Usage
    c2_0, ndf_0, c2p_0 = chi2(q0[0], u0[0])
    c2_1, ndf_1, c2p_1 = chi2(q1[0], u1[0])
    c2_2, ndf_2, c2p_2 = chi2(q2[0], u2[0])
    print(f"Chi2/NDF: {c2_0:.4f} (NDF: {ndf_0})  {c2p_0} ")
    print(f"Chi2/NDF: {c2_1:.4f} (NDF: {ndf_0})  {c2p_1} ")
    print(f"Chi2/NDF: {c2_2:.4f} (NDF: {ndf_0})  {c2p_2} ")


    if MODE == 2:
        label = "QScintThree_test.sh"
        pl = mpplt_plotter(label=label)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]

        ax.set_aspect('auto')

        ax.plot( bins[:-1], q0[0], drawstyle="steps-post", label="qs3_0", c="r" )
        ax.plot( bins[:-1], q1[0], drawstyle="steps-post", label="qs3_1", c="g" )
        ax.plot( bins[:-1], q2[0], drawstyle="steps-post", label="qs3_2", c="b" )

        ax.plot( bins[:-1], u0[0], drawstyle="steps-post", label="u4_0", c="r" )
        ax.plot( bins[:-1], u1[0], drawstyle="steps-post", label="u4_1", c="g" )
        ax.plot( bins[:-1], u2[0], drawstyle="steps-post", label="u4_2", c="b" )

        ylim = ax.get_ylim()
        #for w in [320,340,360,380,400,420,440,460,480,500,520,540]: ax.plot( [w,w], ylim )
        pass
        ax.legend()
        plt.show()

    elif MODE == -2:

        fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        if A:
            ax.plot(bins[:-1], q0[0], drawstyle="steps-post", label="qs3_0", c="r")
            ax.plot(bins[:-1], u0[0], drawstyle="steps-post", label="u4_0", c="r", linestyle="--")
        pass
        if B:
            ax.plot(bins[:-1], q1[0], drawstyle="steps-post", label="qs3_1", c="g")
            ax.plot(bins[:-1], u1[0], drawstyle="steps-post", label="u4_1", c="g", linestyle="--")
        pass
        if C:
            ax.plot(bins[:-1], q2[0], drawstyle="steps-post", label="qs3_2", c="b")
            ax.plot(bins[:-1], u2[0], drawstyle="steps-post", label="u4_2", c="b", linestyle="--")
        pass

        ax.set_ylabel("Counts")
        ax.set_yscale('log')
        ax.legend()

        if A: ax_ratio.plot(bins[:-1][mask0], ratio0[mask0], drawstyle="steps-post", c="r")
        if B: ax_ratio.plot(bins[:-1][mask1], ratio1[mask1], drawstyle="steps-post", c="g")
        if C: ax_ratio.plot(bins[:-1][mask2], ratio2[mask2], drawstyle="steps-post", c="b")

        # Reference line at 1.0 (for ratio) or 0.0 (for difference)
        ax_ratio.axhline(1.0, color='black', lw=0.5, linestyle=':')
        ax_ratio.set_ylabel(" (OK-G4)/G4")

        center = 0.0
        delta = 0.10
        ax_ratio.set_ylim(center-delta, center+delta)  # Tight zoom for high-stat agreement
        plt.xlabel("Wavelength [nm]")

        plt.show()
    pass






