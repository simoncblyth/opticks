#!/usr/bin/env python
"""
cxs_min_hlm.py
==============

~/o/CSGOptiX/cxs_min.sh hlm

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.fold import Fold, EVAL
from opticks.sysrap.sphoton import SPhoton
from opticks.sysrap.sphotonlite import SPhotonLite, SPhotonLite_Merge

MERGER = SPhotonLite_Merge()
TEST = os.environ.get("TEST", "")



def plot_hcb(hcb, max_to_plot=3000):
    """
    Frequency of different hitcounts
    """
    plt.bar(np.arange(1, max_to_plot+1), hcb[1:max_to_plot+1], width=1)
    plt.xlabel('hitcount')
    plt.ylabel('number of "hitlitemerged" hits with various hitcount ')
    plt.yscale('log')    # probably necessary because of huge dynamic range
    plt.show()



def check_submerge(f):
    for fsub_name in f.ff:
        fsub = getattr(f,fsub_name)
        _hlm = getattr(fsub, "hitlitemerged")
        hlm = MERGER(_hlm)
        remerge_same = np.all(_hlm == hlm)
        print(" fsub_name %s remerge_same %s" % ( fsub_name, remerge_same ))
    pass



if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )


    f = Fold.Load("$AFOLD", symbol="f")
    print(repr(f))

    hlm = f.hitlitemerged

    aa = f.allsub("hitlitemerged")
    c_hlm = np.concat(aa)
    hlm2 = MERGER(c_hlm)


    if 1:EVAL(r"""

    TEST
    f.hitlitemerged_meta

    len(aa)
    c_hlm.shape
    hlm2.shape

    np.all( hlm == hlm2 )

    """, ns=locals())


if 0:

    _hlm = f.hitlitemerged
    hlm = SPhotonLite.view(_hlm)

    EVAL(r"""

    TEST
    f.hitlitemerged_meta

    _hlm.shape  # hitlitemerged
    hlm.shape

    hlm.identity.min(),hlm.identity.max()  ## expected min is 1, not 0
    hlm.time.min(),hlm.time.max()

    hlm.hitcount.min(),hlm.hitcount.max()
    hcb = np.bincount(hlm.hitcount)
    hcb[:100]
    hcb[-100:]

    hlm.phi.min(),hlm.phi.max()
    np.histogram( hlm.phi )

    hlm.cost.min(),hlm.cost.max()
    np.histogram( hlm.cost )

    hlm.cos.min(),hlm.cos.max()
    np.histogram( hlm.cos )

    hlm.hitcount.sum()/1e6


    """, ns=locals())

    #plot_hcb(hcb)

    



