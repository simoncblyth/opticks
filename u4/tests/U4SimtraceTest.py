#!/usr/bin/env python
"""
U4SimtraceTest.py
====================


"""
import os, logging, numpy as np
from collections import OrderedDict as odict 
from opticks.ana.fold import Fold
from opticks.ana.pvplt import mpplt_add_contiguous_line_segments, mpplt_add_line

colors = "red green blue cyan magenta yellow pink orange purple lightgreen".split()
gcol = "grey"

#from opticks.ana.framegensteps import FrameGensteps
#from opticks.ana.simtrace_positions import SimtracePositions
#from opticks.ana.simtrace_plot import SimtracePlot, pv, mp

SZ = float(os.environ.get("SZ",3))
REVERSE = int(os.environ.get("REVERSE","0")) == 1
size = np.array([1280, 720])
X,Y,Z = 0,1,2

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp  
except ImportError:
    mp = None
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    FOLD = os.path.expandvars("$FOLD")
    trs = np.load(os.path.join(FOLD, "trs.npy"))
    trs_names = np.loadtxt( os.path.join(FOLD, "trs_names.txt"), dtype=np.object )

    topline = "U4SimtraceTest.py"
    botline = "%s" % FOLD
    thirdline = "thirdline"


    sfs = odict()
    for i, name in enumerate(trs_names):
        sfs[name] = Fold.Load(FOLD, name, symbol="sfs%0.2d" % i )
    pass

    axes = X,Z
    H,V = axes

    if mp:
        fig, ax = mp.subplots(figsize=size/100.) # 100 dpi 
        ax.set_aspect('equal')

        num = len(trs_names)
        for j in range(num):
            i = num - 1 - j  if REVERSE else j
            soname = trs_names[i]
            sf = sfs[soname]
            tr = np.float32(trs[i])
            tr[:,3] = [0,0,0,1]   ## fixup 4th column, as may contain identity info

            lpos = sf.simtrace[:,1].copy()
            lpos[:,3] = 1

            gpos = np.dot( lpos, tr )
            pass
            color = colors[ i % len(colors)]

            label = str(soname)
            if label[0] == "_": label = label[1:]   # seems labels starting "_" have special meaning to mpl, causing problems
            label = label.replace("solid","s")

            ax.scatter( gpos[:,H], gpos[:,V], s=SZ, color=color, label=label )
        pass

        locs = ["upper left","lower left", "upper right"]
        LOC = os.environ.get("LOC",locs[0])
        if LOC != "skip":
            ax.legend(loc=LOC,  markerscale=4)
        pass
        fig.suptitle("\n".join([topline,botline,thirdline]))
        fig.show()
    pass
pass



