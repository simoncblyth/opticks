#!/usr/bin/env python

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from foundry import Foundry, BB

try:
    import matplotlib.pyplot as plt 
    import matplotlib.lines as mlines
except ImportError:
    plt = None
pass


def plot_xz(self, ax, c="r", l="-"):
    ax.add_line(mlines.Line2D([self.x0, self.x1], [self.z0, self.z0], c=c, linestyle=l))  # bottom horizonal
    ax.add_line(mlines.Line2D([self.x0, self.x0], [self.z0, self.z1], c=c, linestyle=l))  # left vertical
    ax.add_line(mlines.Line2D([self.x1, self.x1], [self.z0, self.z1], c=c, linestyle=l))  # right vertical
    ax.add_line(mlines.Line2D([self.x0, self.x1], [self.z1, self.z1], c=c, linestyle=l))  # top horizonal

def adjust_xz(self, ax, scale=1.2):
    ax.set_aspect('equal')
    ax.set_xlim(scale*self.x0, scale*self.x1) 
    ax.set_ylim(scale*self.z0, scale*self.z1) 

# dynamic addition of methods to BB class
setattr(BB, 'plot_xz', plot_xz )
setattr(BB, 'adjust_xz', adjust_xz )



def plot_solid_bb(s, pnbb = True):
    numPrim = len(s.prim)
    log.info("s %s " % repr(s)) 
    log.info("numPrim %d " % numPrim) 

    if numPrim <= 5:
       layout = (2,3)
    else:
       layout = (2,4)
    pass
    plt.ion()

    fig, axs = plt.subplots(*layout)
    if not type(axs) is np.ndarray: axs = [axs]
    plt.suptitle("foundry_plt.py %s " % s.label)

    axs = axs.ravel() 
    color = "rgbcmyk"
    sbb = BB()
    for i in range(numPrim): 
        c = color[i % len(color)]
        p = s.prim[i]
        pbb = p.bb
        sbb.include(pbb)

        log.info("primIdx %4d pbb %s " % (i,str(pbb))) 

        pbb.plot_xz(axs[0], c=c)     # collect all onto first 
        pbb.plot_xz(axs[1+i], c=c) 

        if pnbb:
            l = ":" # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            for nodeIdx, n in enumerate(p.node):
                nbb = n.bb
                nbb.plot_xz(axs[1+i], c=c, l=l)   
                log.info("nodeIdx %4d n %s " % (nodeIdx,str(n))) 
            pass 
        pass
    pass
    for ax in axs:
        sbb.adjust_xz(ax)
    pass
    fig.show()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fd = Foundry()
    args = sys.argv[1:] if len(sys.argv) > 1 else "r1".split() 
    for arg in args:
        s = fd[arg]
        plot_solid_bb(s)
    pass


