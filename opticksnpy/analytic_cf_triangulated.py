#!/usr/bin/env python
"""
analytic_cf_triangulated
--------------------------

Plot analytic shapes and mesh vertices together.
Making the problem of vacumm volume vertices very plain.


"""

import os, logging
import numpy as np

from env.nuwa.detdesc.pmt.plot import Pmt, PmtPlot, one_plot
from env.numerics.npy.mesh import MergedMesh
from env.numerics.npy.dae import DAE
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

X = 0
Y = 1
Z = 2

ZX = [Z,X]
ZY = [Z,Y]
XY = [X,Y]


def plot_vertices(fig, mm, N=5):
    N = 5
    for i in range(N):
        ax = fig.add_subplot(1,N,i, aspect='equal')
        ax.set_xlim(-120,120)
        ax.set_ylim(-200,150)
        rz = mm.rz_(i)
        ax.plot(rz[:,0], rz[:,1], "o")
    pass


def solids_plot(fig, pmt, mm, solids=range(5)):

    if len(solids)>4:
        ny,nx = 3,2 
    else:
        ny,nx = 2,2 

    for i,solid in enumerate(solids):
        pts = pmt.parts(solid)
        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        pp = PmtPlot(ax, pmt, axes=ZX) 
        pp.plot_shape(pts, clip=True)
        pp.limits()

        rz = mm.rz_(i)
        z = rz[:,1]
        r = rz[:,0]
        zedge = z[z>-150].min() 

        log.info("i %s z[z>-150].min() %s " % (i,zedge) )
        ax.plot(z, r, "o")
    pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)
    plt.ion()
    plt.close()

    idfold = os.path.dirname(os.path.dirname(os.path.expandvars("$IDPATH")))

    base = os.path.expandvars("$IDPATH/GMergedMesh/1")
    #base = "/tmp/GMergedMesh/baseGeometry"
    #base = "/tmp/GMergedMesh/modifyGeometry"
    #base = os.path.expandvars("$IDPATH_DPIB_ALL/GMergedMesh/0")
    #base = os.path.expandvars("$IDPATH_DPIB_PMT/GMergedMesh/0")

    if base.startswith(idfold):
        name = base[len(idfold)+1:]
    else:
        name = base 


    mm = MergedMesh(base=base)
    if base.find("export/dpib")>-1:
        mm.node_offset = 1
    else:
        mm.node_offset = 0
    pass


    pmt = Pmt()
    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    pts = pmt.parts(ALL)

    fig = plt.figure() 
    fig.suptitle("analytic vs triangulated : %s " % name )

    #one_plot(fig, pmt, pts, axes=ZX, clip=True)

    solids_plot(fig, pmt, mm, solids=range(5))

    #plot_vertices(fig, mm)

    plt.show()

    print mm.nodeinfo.view(np.int32)


    path_0 = DAE.standardpath()
    dae_0 = DAE(path_0)

    a = dae_0.float_array("pmt-hemi-vac0xc21e248-Pos-array")

    r = np.linalg.norm(a[:,:2], 2, 1) * np.sign(a[:,0] )
    z = a[:,2]
    print z[z>-150].min()

    ax = fig.add_subplot(2,3,6, aspect='equal')
    ax.plot(z, r, "o")



