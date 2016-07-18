#!/usr/bin/env python
"""
MergedMesh
~~~~~~~~~~~~

Enables access to geocache data.

.. code-block:: py

    In [37]: filter(lambda _:_.find("IAV") > -1, vn.pv.names)
    Out[37]: 
    ['__dd__Geometry__AD__lvLSO--pvIAV0xc2d0348',
     '__dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00',
     '__dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xbf6b0e0',
     '__dd__Geometry__AD__lvLSO--pvIAV0xc2d0348',
     '__dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00',
     '__dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xbf6b0e0']

    In [38]: mm.center_extent
    Out[38]: 
    array([[ -16520.   , -802110.   ,   -7125.   ,    7710.562],
           [ -16520.   , -802110.   ,    3892.9  ,   34569.875],
           [ -12840.846, -806876.25 ,    5389.855,   22545.562],
           ..., 
           [ -12195.957, -799312.625,   -7260.   ,    5000.   ],
           [ -17081.184, -794607.812,   -7260.   ,    5000.   ],
           [ -16519.908, -802110.   ,  -12410.   ,    7800.875]], dtype=float32)

    In [39]: mm.center_extent.shape
    Out[39]: (12230, 4)


Exercise:

* add methods to enable access the center and extent of a 
  volume from its name 

* determine the coordinates of the centers of the IAVs
  in the three DayaBay sites

"""
import os, logging, numpy as np
from opticks.ana.base import ItemList 
from opticks.ana.nbase import vnorm

log = logging.getLogger(__name__)

class MergedMesh(object):
    def __init__(self, base):
        mdir = os.path.expandvars(base)
        for name in os.listdir(mdir):
            stem, ext = os.path.splitext(name)
            log.debug(" name %s ext [%s] " % (name, ext))
            if ext == ".npy": 
                path = os.path.join(mdir,name)
                log.debug("path %s " % path )
                os.system("ls -l %s " % path)
                a = np.load(path)
                log.debug(" stem %s shape %s " % (stem, repr(a.shape)))
                setattr(self, stem, a)
            else:
                pass
            pass
        pass

    def vertices_(self, i):
        """
        nodeinfo[:,1] provides the vertex counts, to convert a node index
        into a vertex range potentially with a node numbering offset need
        to add up all vertices prior to the one wish to access 

        ::

            In [6]: imm.nodeinfo
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
        #r = np.linalg.norm(v[:,:2], 2, 1)
        r = vnorm(v[:,:2])

        rz = np.zeros((len(v),2), dtype=np.float32)
        rz[:,0] = r*np.sign(v[:,0])
        rz[:,1] = v[:,2]
        return rz 






class VolumeNames(object):
    def __init__(self):
        self.lv = ItemList(txt="LVNames", offset=0)
        self.pv = ItemList(txt="PVNames", offset=0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mm = MergedMesh("$IDPATH/GMergedMesh/0")
    vn = VolumeNames()





