#!/usr/bin/env python
"""
gplt.py : 2d debugging/presentation of solids with rotational symmetry
========================================================================

Make connection between GDML parsing and the simple 2d matplotlib plotting 
of xplt.py while avoiding the need for manual translation as done in the x018.py etc..

"""

import os, sys, argparse, logging
log = logging.getLogger(__name__)
import numpy as np, math 

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~"))

from opticks.analytic.gdml import GDML
from opticks.ana.torus_hyperboloid import Tor, Hyp

def parse_args(doc):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "--path", default="$OPTICKS_PREFIX/tds.gdml")

    #dlv = "lInnerWater"
    #dlv = "NNVTMCPPMTlMaskVirtual"
    #dlv = "NNVTMCPPMT_PMT_20inch_log"
    dlv = "NNVTMCPPMT_PMT_20inch_inner1_log"

    parser.add_argument( "--lvx", default=dlv, help="LV name prefix" )

    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    return args


def traverse_r(lv0, depth=0):
    pvs = lv0.physvol
    indent = "   " * depth 
    print("[%2d] %s %4d %s " % (depth, indent,len(pvs),lv0.name))
    pass
    for pv in pvs:
        lv = pv.volume
        traverse_r( lv, depth+1)
    pass


if __name__ == '__main__':
    args = parse_args(__doc__)


    g = GDML.parse(args.path)
    log.info("g: %r " % g)
    ns = len(g.solids.keys())
    nv = len(g.volumes.keys())
    lvks = filter(lambda k:k.startswith(args.lvx), g.volumes.keys()) 
    nk = len(lvks)
    log.info(" ns:%d nv:%d nk:%d " % (ns,nv,nk))

    if nk > 0:
        for i,lvk in enumerate(lvks):
            log.info(" %2d : %s" % (i,lvk) )
        pass 
        lvk = lvks[0]
        lv = g.volumes[lvk]
    else:
        lv = g.world
    pass    
    traverse_r(lv, 0)


    





    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("gplt")

    ax = fig.add_subplot(111)
    ax.set_ylim([-350,200])
    ax.set_xlim([-300,300])








    for pt in x.root.patches():
        print "pt ", pt
        ax.add_patch(pt)
    pass

    fig.show()








