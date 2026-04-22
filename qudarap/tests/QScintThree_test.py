#!/usr/bin/env python
"""

~/o/qudarap/tests/QScintThree_test.sh pdb


Observe Chi2 spikes at the edges of the HD zoom regions.


COMP=012 BINS=200:800:1 ~/o/qudarap/tests/QScintThree_test.sh pdb
    QScintThree_test.sh HD:1 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.4733 (NDF: 598) ) (PPO  3.6821 (NDF: 242) ) (bisMSB  1.6205 (NDF: 598) )


With HD:0 LAB has big chi2 deviation at about 330 nm close to a step in the distribution.



COMP=0 BINS=201:800:1 LOGY=1 ~/o/qudarap/tests/QScintThree_test.sh pdb
COMP=1 BINS=310:545:1 LOGY=1 ~/o/qudarap/tests/QScintThree_test.sh pdb



FIXED Issue : LAB small number of unexpected ICDF wavelengths in 80->200
--------------------------------------------------------------------------

FIXED using NP::MakePCopyStripZeroPadding

Are generating wavelengths in range 80->200 in the GPU icdf lookups
which do not happen in the Geant4 GetEnergy generation.

HMM, the problem may be the stray 79.99 in the icdf::

    In [7]: icdf[0].reshape(3,4096)
    Out[7]:
    array([[487.695, 487.568, 487.441, 487.314, 487.188, ..., 390.932, 390.899, 390.865, 390.832, 390.799],
           [799.898, 799.115, 798.333, 797.553, 796.775, ..., 482.605, 482.599, 482.592, 482.586, 482.58 ],
           [392.081, 392.08 , 392.078, 392.077, 392.075, ..., 200.607, 200.448, 200.29 , 200.132,  79.99 ]], shape=(3, 4096))


That 79.99 looks like a placeholder "energy" edge. Not a real measurement ?
Actually, I vaguely recall smth similar before, where have to squeeze past
outer placeholders ?

BINGO, confirmed arbitrary edge sneaking into ICDF::

    In [17]: np.c_[hc_eVnm/(1e6*f.U4ScintThree.lab_cmp[:,0]),f.U4ScintThree.lab_cmp[:,1]*1e6]
    Out[17]:
    array([[    799.898,       0.   ],
           [    600.001,    7298.   ],
           [    599.001,    7011.   ],
           [    598.001,    7932.   ],
           [    596.999,    8065.   ],
           [    596.   ,    7699.   ],
           [    594.999,    6620.   ],
           [    593.999,    7727.   ],
           [    592.999,    7464.   ],
           [    592.   ,    7931.   ],
           [    591.001,    6825.   ],
           [    589.999,    7623.   ],
           [    589.001,    6487.   ],
           ...
           [    334.   ,    2218.   ],
           [    333.   ,    1887.   ],
           [    332.001,    1981.   ],
           [    331.   ,    2153.   ],
           [    330.   ,    2269.   ],
           [    199.975,       0.   ],
           [    120.023,       0.   ],
           [     79.99 ,       0.   ]])

"""

import os, sys, numpy as np
from opticks.ana.fold import Fold
MODE = int(os.environ.get("MODE","0"))
YLIM = tuple(map(float, os.environ["YLIM"].split(","))) if "YLIM" in os.environ else None
DPI = int(os.environ.get("DPI","300"))
SIZE = np.array([1280, 720])


class Chi2:
    def __init__(self, obs, ref, mask):
        q = obs[mask]
        u = ref[mask]
        a = (q - u)*(q - u)/(q + u)
        self.a = a

    def __str__(self):
        ndf = len(self.a) - 1
        chi2_per = np.sum(self.a) / ndf
        return f" {chi2_per:.4f} (NDF: {ndf}) "


if MODE in [-2,2,3]:
    import matplotlib.pyplot as plt
    from opticks.ana.pvplt import *
pass


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    HD = int(f.NPFold_meta.HD)
    a_wl = f.wl
    b_wl = f.U4ScintThree.wls
    icdf = f.U4ScintThree.icdf

    WITH_LERP = f.wl_meta.WITH_LERP


    print(f"HD {HD} YLIM {YLIM}")
    print("a_wl\n",a_wl)
    print("b_wl\n",b_wl)

    _COMP = os.environ.get("COMP","012")
    COMP = list(map(int,list(_COMP)))

    #bins_ = "80:800:4"
    #bins_ = "200:800:1"
    bins_ = "305:605:1"
    #bins_ = "300:600:1"
    #bins_ = "300:640:1"
    #bins_ = "300:540:1"  # avoid 550

    BINS_ = os.environ.get("BINS", bins_)
    BINS = tuple(map(float,BINS_.split(":")))
    bins = np.arange(*BINS)

    threshold = 100
    a = {}
    b = {}
    mask = {}
    ratio = {}
    cc = {}
    metric = {}
    label = "QScintThree_test.sh"
    name = label.replace(".","_")
    title = f"{name}\n HD:{HD} LERP:{WITH_LERP} COMP:{_COMP} BINS:{BINS} : Chi2/ndf "
    names = ["LAB","PPO","bisMSB"]
    color = ["r","g","b"]
    LOGY = int(os.environ.get("LOGY","0"))
    EDGE = int(os.environ.get("EDGE","0"))

    for i in range(3):
        if not i in COMP: continue

        a_wl_r = (a_wl[i].min(), a_wl[i].max())
        b_wl_r = (b_wl[i].min(), b_wl[i].max())
        print(f"i {i} a_wl_r:{a_wl_r} b_wl_r:{b_wl_r} ")

        a[i] = np.histogram( a_wl[i] , bins )
        b[i] = np.histogram( b_wl[i] , bins )
        mask[i] = (a[i][0] > threshold) & (b[i][0] > threshold)
        ratio[i] = np.divide(a[i][0]-b[i][0], b[i][0], out=np.ones_like(a[i][0], dtype=float), where=mask[i])
        cc[i] = Chi2(a[i][0],b[i][0],mask[i])
        title += " (" + names[i] + " " + str(cc[i]) + ")"
        metric[i] = cc[i].a
    pass
    print(title)

    print("wavelength ranges of the 9 texture layers")

    ww = { 0:[], 1:[], 2:[] }  # collect edge wavelengths
    for j in range(3):
        for i in range(3):
            mx = f.U4ScintThree.icdf[i,j,0,0]
            mn = f.U4ScintThree.icdf[i,j,-1,0]
            ww[i].append(mn)
            ww[i].append(mx)
            print(f" i-species-{i} j-zoom-{j}  0:{mx} -1:{mn} ")
        pass
    pass


    if MODE == 2:
        pl = mpplt_plotter(label=label)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]

        ax.set_aspect('auto')

        for i in range(3):
            if not i in COMP: continue
            ax.plot( bins[:-1], a[i][0], drawstyle="steps-post", label=f"a[{i}]", c=color[i] )
            ax.plot( bins[:-1], b[i][0], drawstyle="steps-post", label="b[{i}]", c=color[i] )
        pass

        if YLIM is None:
           print(f"not setting YLIM {YLIM}")
        else:
           print(f"setting YLIM {YLIM}")
           ax.set_ylim(*YLIM)
        pass
        ylim = ax.get_ylim()
        #for w in [320,340,360,380,400,420,440,460,480,500,520,540]: ax.plot( [w,w], ylim )
        pass
        ax.legend()
        plt.show()

    elif MODE == -2:


        fig, (ax, ax_metric) = plt.subplots(2, 1, figsize=SIZE*2/100, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        for i in range(3):
            if not i in COMP: continue
            ax.plot(bins[:-1], a[i][0], drawstyle="steps-post", label=f"a[{i}]", c=color[i])
            ax.plot(bins[:-1], b[i][0], drawstyle="steps-post", label=f"b[{i}]", c=color[i], linestyle="--", marker="o", markersize=1 )
            ylim = ax.get_ylim()

            if EDGE:
                for w in ww[i]: ax.plot( [w,w], ylim, c=color[i], linestyle="--" )
            pass
            ax_metric.plot(bins[:-1][mask[i]], metric[i], drawstyle="steps-post", c=color[i])
        pass

        if YLIM is None:
           print(f"not setting YLIM {YLIM}")
        else:
           print(f"setting YLIM {YLIM}")
           ax.set_ylim(*YLIM)
        pass

        ax.set_title(title)
        ax.set_ylabel("Counts")
        if LOGY: ax.set_yscale('log')
        ax.legend()

        # Reference line at 1.0 (for ratio) or 0.0 (for difference)
        ax_metric.axhline(1.0, color='black', lw=0.5, linestyle=':')
        ax_metric.set_ylabel(" (A-B)(A-B)/(A+B)")

        center = 1.0
        delta = 1.0
        #ax_metric.set_ylim(center-delta, center+delta)  # Tight zoom for high-stat agreement
        plt.xlabel("Wavelength [nm]")

        plt.show()

        identity = f"{name}_COMP_{_COMP}_LOGY{LOGY}_EDGE{EDGE}_DPI{DPI}"
        txtpath = os.path.expandvars(f"$FOLD/{identity}_meta.txt")
        figpath = os.path.expandvars(f"$FOLD/{identity}.png")
        print(f"savefig to figpath {figpath}")
        fig.savefig(figpath, dpi=DPI, bbox_inches='tight')

        print(f"write meta to txtpath {txtpath}")
        with open(txtpath,"w") as fp:
            fp.write(identity)
            fp.write(figpath)
            fp.write(txtpath)
            fp.write(title)
        pass


        pass
    elif MODE == -4:
        print("TBD")
    pass


