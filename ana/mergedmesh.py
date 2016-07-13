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

class VolumeNames(object):
    def __init__(self):
        self.lv = ItemList(txt="LVNames", offset=0)
        self.pv = ItemList(txt="PVNames", offset=0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mm = MergedMesh("$IDPATH/GMergedMesh/0")
    vn = VolumeNames()





