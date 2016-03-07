#!/usr/bin/env python
"""

::

    In [11]: mm.nodeinfo
    Out[11]: 
    array([[ 720,  362, 3199, 3155],
           [ 672,  338, 3200, 3199],
           [ 960,  482, 3201, 3200],
           [ 480,  242, 3202, 3200],
           [  96,   50, 3203, 3200]], dtype=uint32)

    In [12]: mm.nodeinfo[:,0].sum()
    Out[12]: 2928

    In [13]: mm.vertices.shape
    Out[13]: (1474, 3)

    In [14]: mm.nodeinfo[:,1].sum()
    Out[14]: 1474

"""

import os, logging
import numpy as np
from env.nuwa.detdesc.pmt.plot import Pmt, PmtPlot, one_plot
from env.numerics.npy.dae import DAE


import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


X = 0
Y = 1
Z = 2

ZX = [Z,X]
ZY = [Z,Y]
XY = [X,Y]


NAMES = """
aiidentity
bbox
boundaries
center_extent
colors
identity
iidentity
indices
itransforms
meshes
nodeinfo
nodes
normals
sensors
transforms
vertices
""".split()


class MergedMesh(object):
    def __init__(self, base, node_offset=0):
        for name in NAMES:    
            path = os.path.join(base,"%s.npy" % name)
            if os.path.exists(path):
                log.info("path %s " % path )
                a = np.load(path)
            else:
                log.warning("NO PATH %s " % path )
                a = None
            pass
            setattr(self, name, a)
        pass
        self.node_offset = node_offset

    def vertices_(self, i):
        """
        nodeinfo[:,1] provides the vertex counts, to convert a node index
        into a vertex range potentially with a node numbering offset need
        to add up all vertices prior to the one wish to access 

        ::

            In [6]: mm.nodeinfo
            Out[6]: 
            array([[ 720,  362, 3199, 3155],
                   [ 672,  338, 3200, 3199],
                   [ 960,  482, 3201, 3200],
                   [ 480,  242, 3202, 3200],
                   [  96,   50, 3203, 3200]], dtype=uint32)


        """
        no = self.node_offset
        v_count = self.nodeinfo[no:no+i, 1]
        v_offset = v_count.sum()
        v_number = self.nodeinfo[no+i, 1]
        log.info("v_number %s v_offset %s " % (v_number, v_offset))
        return self.vertices[v_offset:v_offset+v_number] 

    def rz_(self, i):
        v = self.vertices_(i)
        r = np.linalg.norm(v[:,:2], 2, 1)

        rz = np.zeros((len(v),2), dtype=np.float32)
        rz[:,0] = r*np.sign(v[:,0])
        rz[:,1] = v[:,2]
        return rz 



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

    base = os.path.expandvars("$IDPATH/GMergedMesh/1")
    #base = "/tmp/GMergedMesh/baseGeometry"
    #base = "/tmp/GMergedMesh/modifyGeometry"
    #base = os.path.expandvars("$IDPATH_DPIB/GMergedMesh/0")

    mm = MergedMesh(base=base)
    if base.find("export/dpib")>-1:
        mm.node_offset = 1
    else:
        mm.node_offset = 0



    pmt = Pmt()
    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    pts = pmt.parts(ALL)

    fig = plt.figure() 

    #one_plot(fig, pmt, pts, axes=ZX, clip=True)

    solids_plot(fig, pmt, mm, solids=range(5))

    #plot_vertices(fig, mm)

    plt.show()

    print mm.nodeinfo

    dae = DAE("/tmp/g4_00.dae")
    a = dae.float_array("pmt-hemi-vac0xc21e248-Pos-array")
    r = np.linalg.norm(a[:,:2], 2, 1) * np.sign(a[:,0] )
    z = a[:,2]
    print z[z>-150].min()

    ax = fig.add_subplot(3,2,6, aspect='equal')
    ax.plot(z, r, "o")



