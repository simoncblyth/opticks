#!/usr/bin/env python
"""
GNodeLib.py
=============

Geocache nodelib to load is controlled by OPTICKS_KEYDIR envvar. Set that in ~/.opticks_config with::

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3  ## example key  
    export OPTICKS_KEYDIR=$(geocache-keydir)  # derived from OPTICKS_KEY see geocache-info

ipython -i GNodeLib.py::

     nlib.pv 
     nlib.lv
     nlib.tr
     nlib.id
     nlib.ce
     nlib.bb


Info for a node identified by flat index::

    epsilon:~ blyth$ GNodeLib.py 3154
    TR
    array([[      0.543,      -0.84 ,       0.   ,       0.   ],
           [      0.84 ,       0.543,       0.   ,       0.   ],
           [      0.   ,       0.   ,       1.   ,       0.   ],
           [ -18079.453, -799699.44 ,   -7100.   ,       1.   ]], dtype=float32)

    BB
    array([[ -20576.252, -802196.25 ,   -9600.   ,       1.   ],
           [ -15582.654, -797202.6  ,   -4600.   ,       1.   ]], dtype=float32)

    ID
    array([3154,   94,   18,    0], dtype=uint32)

    CE
    array([ -18079.453, -799699.44 ,   -7100.   ,    2500.   ], dtype=float32)

    PV
    '/dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100'

    LV
    '/dd/Geometry/AD/lvSST0xc234cd00x3f0b5e0'


TODO:

* make an equivalent GMergedMesh.py that accesses the same info 
  with "triplet" indexing  (repeat_idx, instance_index, prim_index) 


"""
import os, numpy as np, argparse, logging
log = logging.getLogger(__name__)

class Txt(list):
    def __init__(self, *args, **kwa):
        list.__init__(self, *args, **kwa)
    def __repr__(self):
        return "Txt:%d" % len(self)

txt_load = lambda _:Txt(map(str.strip, open(_).readlines()))

class Node(object):
    def __init__(self, nlib, idx):
        self.nlib = nlib
        self.idx = idx
    def __str__(self):
        return "\n".join(map(lambda k:"%2s\n%r\n" % (k, getattr(self.nlib,k.lower())[idx]), self.nlib.key2name.keys()))


class GNodeLib(object):

    RELDIR = "GNodeLib" 
    key2name = {
      "TR":"volume_transforms.npy",
      "BB":"volume_bbox.npy",
      "ID":"volume_identity.npy",
      "CE":"volume_center_extent.npy",
      "PV":"volume_PVNames.txt",
      "LV":"volume_LVNames.txt",
    }

    @classmethod   
    def Path(cls, key): 
        return os.path.expandvars("$OPTICKS_KEYDIR/%s/%s" % (cls.RELDIR,cls.key2name[key]))

    @classmethod   
    def Load(cls, key): 
        path = cls.Path(key)
        return np.load(path) if path[-4:] == ".npy" else txt_load(path)

    def __init__(self):
        for key in self.key2name.keys(): 
            setattr( self, key.lower(), self.Load(key) )
        pass

    def __str__(self):
        return "\n".join(map(lambda k:"%2s\n%r\n" % (k, getattr(self,k.lower())), self.key2name.keys()))

    def __getitem__(self, idx):
        return Node(self, idx)


def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "idx",  type=int, nargs="+", help="Node index to dump.")
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(     "-d","--dump", action="store_true", help="Dump lib repr" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  

if __name__ == '__main__':
    args = parse_args(__doc__)
    nlib = GNodeLib()
    if args.dump: 
        print(nlib)
    pass
    for idx in args.idx:
        nd = nlib[idx]
        print(nd)
    pass

