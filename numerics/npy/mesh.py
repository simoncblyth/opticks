#!/usr/bin/env python
"""
::

    In [4]: run mesh.py
    INFO:__main__:base /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 
    [[ 720  362 3199 3155]
     [ 672  338 3200 3199]
     [ 960  482 3201 3200]
     [ 480  242 3202 3200]
     [  96   50 3203 3200]]
    INFO:__main__:base /tmp/GMergedMesh/baseGeometry 
    [[ 720  362 3199 3155]
     [ 672  338 3200 3199]
     [ 960  482 3201 3200]
     [ 480  242 3202 3200]
     [  96   50 3203 3200]]
    WARNING:__main__:NO PATH /tmp/GMergedMesh/modifyGeometry/iidentity.npy 
    INFO:__main__:base /tmp/GMergedMesh/modifyGeometry 
    [[       720        362       3199       3155]
     [       672        338       3200       3199]
     [       960        482       3201       3200]
     [       480        242       3202       3200]
     [        96         50       3203       3200]
     [        12         24          0 4294967295]]
    WARNING:__main__:NO PATH /usr/local/env/geant4/geometry/export/dpib/cfg4.f7ba6061a8e024189e641c86eb847ee4.dae/GMergedMesh/0/aiidentity.npy 
    WARNING:__main__:NO PATH /usr/local/env/geant4/geometry/export/dpib/cfg4.f7ba6061a8e024189e641c86eb847ee4.dae/GMergedMesh/0/iidentity.npy 
    WARNING:__main__:NO PATH /usr/local/env/geant4/geometry/export/dpib/cfg4.f7ba6061a8e024189e641c86eb847ee4.dae/GMergedMesh/0/itransforms.npy 
    INFO:__main__:base /usr/local/env/geant4/geometry/export/dpib/cfg4.f7ba6061a8e024189e641c86eb847ee4.dae/GMergedMesh/0 
    [[         0          0          0 4294967295]
     [       720        362          1          0]
     [       720        362          2          1]
     [       960        482          3          2]
     [       576        288          4          2]
     [         0          0          5          2]]


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
                log.debug("path %s " % path )
                os.system("ls -l %s " % path)
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
        v_count = self.nodeinfo[0:no+i, 1]  # even when wish to ignore a solid, still have to offset vertices 
        v_offset = v_count.sum()
        v_number = self.nodeinfo[no+i, 1]
        log.info("no %s v_count %s v_number %s v_offset %s " % (no, repr(v_count), v_number, v_offset))
        return self.vertices[v_offset:v_offset+v_number] 

    def rz_(self, i):
        v = self.vertices_(i)
        r = np.linalg.norm(v[:,:2], 2, 1)

        rz = np.zeros((len(v),2), dtype=np.float32)
        rz[:,0] = r*np.sign(v[:,0])
        rz[:,1] = v[:,2]
        return rz 


if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)


    DPIB_ALL = os.path.expandvars("$IDPATH_DPIB_ALL");
    DPIB_PMT = os.path.expandvars("$IDPATH_DPIB_PMT");

    bases = [os.path.expandvars("$IDPATH/GMergedMesh/1"),
             "/tmp/GMergedMesh/baseGeometry",
             "/tmp/GMergedMesh/modifyGeometry",
             os.path.join(DPIB_ALL,"GMergedMesh/0"),
             os.path.join(DPIB_PMT,"GMergedMesh/0"),
           ]

    for base in bases:
        if not os.path.exists(base):continue

        mm = MergedMesh(base=base)
        if base.find(DPIB_ALL)>-1 or base.find(DPIB_PMT)>0:
            mm.node_offset = 1
        else:
            mm.node_offset = 0
        pass
        log.info("base %s " % base)
        print  "nodeinfo\n", mm.nodeinfo.view(np.int32)
        nv = mm.nodeinfo[:,1].sum()
        print  "nvert %5d v.shape %s " % (nv, repr(mm.vertices.shape))
        #print  "ce\n", mm.center_extent

        print "itransforms\n",mm.itransforms
        print "iidentity\n",mm.iidentity
        print "aiidentity\n", mm.aiidentity


        

        




