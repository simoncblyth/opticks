#!/usr/bin/env python
"""
GNodeLib.py
=============

::

   ipython -i -- GNodeLib.py --ulv --detail


* see also ggeo.py which provides "triplet" indexing  (repeat_idx, instance_index, prim_index) 

Geocache nodelib to load is controlled by OPTICKS_KEYDIR envvar. Set that in ~/.opticks_config with::

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3  ## example key  

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



Can also identify nodes using LV names or PV names and control 
output with slice specification::

    GNodeLib.py --lv HamamatsuR12860_PMT_20inch_body_log --sli 0:2

Dump a list of unique LV names with::

    GNodeLib.py --ulv 



"""
import os, json, numpy as np, argparse, logging
log = logging.getLogger(__name__)

from opticks.ana.key import key_

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
    def __repr__(self):
        return "### Node idx:%d " % self.idx
    def __str__(self):
        return "\n".join([repr(self),""]+list(map(lambda k:"%2s\n%r\n" % (k, getattr(self.nlib,k.lower())[idx]), self.nlib.k2name.keys())))


class GNodeLib(object):
    KEY = key_(os.environ["OPTICKS_KEY"])
    KEYDIR = KEY.keydir
    RELDIR = "GNodeLib" 
    k2name = {
      "TR":"all_volume_transforms.npy",
      "BB":"all_volume_bbox.npy",
      "ID":"all_volume_identity.npy",
      "NI":"all_volume_nodeinfo.npy",
      "CE":"all_volume_center_extent.npy",
      "PV":"all_volume_PVNames.txt",
      "LV":"all_volume_LVNames.txt",
    }

    @classmethod   
    def Path(cls, k): 
        keydir = cls.KEYDIR
        reldir = cls.RELDIR
        name = cls.k2name[k]
        return os.path.expandvars("{keydir}/{reldir}/{name}".format(**locals()))

    @classmethod   
    def Load(cls, k): 
        path = cls.Path(k)
        return np.load(path) if path[-4:] == ".npy" else np.loadtxt(path, dtype="|S100" )

    def pvfind(self, pvname_start, encoding='utf-8'):
        """
        :param pvname_start: string start of PV name
        :return indices: array of matching indices in pv name array (all nodes) 
        """
        return np.flatnonzero(np.char.startswith(self.pv, pvname_start.encode(encoding)))  

    def lvfind(self, lvname_start, encoding='utf-8'):
        """
        :param lvname_start: string start of LV name
        :return indices: array of matching indices in lv name array (all nodes: so will be many repeats)  
        """
        return np.flatnonzero(np.char.startswith(self.lv, lvname_start.encode(encoding)))  


    

    def __init__(self):
        for k in self.k2name.keys(): 
            setattr( self, k.lower(), self.Load(k) )
        pass
        self.lvidx = (( self.id[:,2] >> 16) & 0xffff )    
        self.bnidx = (( self.id[:,2] >>  0) & 0xffff ) 

    def __str__(self):
        return "\n".join(map(lambda k:"%2s\n%r\n" % (k, getattr(self,k.lower())), self.k2name.keys()))

    def __getitem__(self, idx):
        return Node(self, idx)


def slice_(sli):
    elem = []
    for s in sli.split(":"):
        try:
            elem.append(int(s))
        except ValueError:
            elem.append(None)
        pass
    return slice(*elem)


def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "idx",  type=int, nargs="*", help="Node index to dump.")
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(     "--pv", default=None, help="PV name to search for" ) 
    parser.add_argument(     "--lv", default=None, help="LV name to search for" ) 
    parser.add_argument(     "--ulv", default=False, action="store_true", help="Dump unique LV names" ) 
    parser.add_argument(     "--sli", default="0:10:1", help="Array slice to control output, 0:None for all" )
    parser.add_argument(     "--ce", default=False, action="store_true", help="Dump just center_extent" ) 
    parser.add_argument(     "--detail", default=False, action="store_true", help="Dump extra details" )
    parser.add_argument(     "-d","--dump", action="store_true", help="Dump lib repr" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    args.slice = slice_(args.sli)
    return args  

if __name__ == '__main__':
    args = parse_args(__doc__)
    print(repr(GNodeLib.KEY))
    nlib = GNodeLib()
    if args.dump: 
        print(nlib)
    pass
    for idx in args.idx:
        nd = nlib[idx]
        print(nd)
    pass

    idxs = []
    if args.pv:
        idxs = nlib.pvfind(args.pv)
        print("args.pv:%s matched %d nodes " % (args.pv, len(idxs))) 
    elif args.lv:
        idxs = nlib.lvfind(args.lv)
        print("args.lv:%s matched %d nodes " % (args.lv, len(idxs))) 
    elif args.ulv:
        ulv, ulv_count = np.unique(nlib.lv, return_counts=True) 
        print("args.ulv found %d unique LV names" % (len(ulv))) 
        print("\n".join(list(map(lambda _:_.decode('utf-8'),ulv[args.slice]))))
        print("unique lv in descending count order, with names of first few corresponding pv ")
        for i in sorted(range(len(ulv)),key=lambda i:ulv_count[i], reverse=True):
            detail = "" 
            if args.detail:
                w = np.where( nlib.lv == ulv[i] )              
                pv = nlib.pv[w][:3]
                detail = " ".join(list(map(lambda _:_.decode("utf-8"), pv )))     
            pass
            print("%10d : %50s : %s " % (ulv_count[i], ulv[i].decode("utf-8"), detail ))
        pass
    pass

    print("slice %s " % args.sli)

    print(idxs[args.slice])
    for idx in idxs[args.slice]:
        if args.ce:
            print(nlib.ce[idx])
        else:
            nd = nlib[idx]
            print(nd)
        pass
    pass 


