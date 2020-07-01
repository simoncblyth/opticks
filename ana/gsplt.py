#!/usr/bin/env python
"""

TODO:

* genstep deltaPosition arrows 

"""

from opticks.ana.gs import GS 
from opticks.analytic.gdml import GDML
import os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import OrderedDict as odict

X,Y,Z,T = 0,1,2,3


if __name__ == '__main__':

    g = GDML.parse("$OPTICKS_PREFIX/tds_ngt_pcnk.gdml")

    lvs = odict()
    lvs[0] = g.find_one_volume("lPoolLining")
    lvs[1] = g.find_one_volume("lInnerWater")
    lvs[2] = g.find_one_volume("lAcrylic")
    lvs[3] = g.find_one_volume("lOuterWaterPool")

    plt.close()
    plt.ion()
    fig = plt.figure(figsize=(6,6))
    fig.suptitle("suptitle") 

    plt.title("plt.title")

    ax = fig.add_subplot(111)
    ax.set_xlim(-26000,26000)
    ax.set_ylim(-26000,26000)

    for lv in lvs.values():
        s = lv.solid  
        kwa = {}
        sh = s.as_shape(**kwa)
        for pt in sh.patches():
            ax.add_patch(pt)
        pass
    pass

    ysli = 1000   # slice around y=0
    hpos = lvs[1].physvol_xyz('pLPMT_Hamamatsu','position')
    npos = lvs[1].physvol_xyz('pLPMT_NNVT','position')
    wpos = lvs[3].physvol_xyz("mask_PMT_20inch_vetolMaskVirtual_phys", "position")

    hpos_ = hpos[np.abs(hpos[:,1]) < ysli] 
    npos_ = npos[np.abs(npos[:,1]) < ysli] 
    wpos_ = wpos[np.abs(wpos[:,1]) < ysli] 

    ax.scatter( hpos_[:,0], hpos_[:,2], 5.0 )
    ax.scatter( npos_[:,0], npos_[:,2], 5.0 )
    ax.scatter( wpos_[:,0], wpos_[:,2], 5.0 )


    path = "$TMP/evt/g4live/natural/1/gs.npy"
    args = GS.parse_args(__doc__, path=path)
    gs = GS(args.path)

    xyzt = gs.xyzt
    ijku = gs.deltaPositionLength
    pdgc = gs.pdgCode

    sli = np.abs(xyzt[:,Y]) < ysli
    nsli = np.count_nonzero(sli)

    xyzt_ = xyzt[sli]
    ijku_ = ijku[sli]
    
    #ax.scatter(xyzt_[:,0],xyzt_[:,2], 0.1)

    kwa = dict(linewidth=0.1)

    for i in range(nsli):
        l = mlines.Line2D(
                   [xyzt_[i,X], xyzt_[i,X] + ijku_[i,X]], 
                   [xyzt_[i,Z], xyzt_[i,Z] + ijku_[i,Z]], **kwa ) 
        ax.add_line(l)
    pass
    




