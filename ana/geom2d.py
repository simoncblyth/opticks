#!/usr/bin/env python
"""
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.geocache import keydir
from opticks.ana.prim import Dir


class Geom2d(object):
    """
    Scratch geometry, for designing flight paths
    """
    def __init__(self, kd, ridx=0):
        self.ridx = str(ridx)
        self.kd = kd
        self._pv = None
        self._lv = None
        self.ce = np.load(os.path.join(self.kd, "GMergedMesh", self.ridx,"center_extent.npy"))
        self.d = Dir(os.path.expandvars(os.path.join("$IDPATH/GParts",self.ridx)))     ## mm0 analytic
        self.select_gprim()
    
    def _get_pv(self):
        if self._pv is None:
            self._pv =  np.loadtxt(os.path.join(self.kd, "GNodeLib/PVNames.txt" ), dtype="|S100" )
        return self._pv
    pv = property(_get_pv)
    
    def _get_lv(self):
        if self._lv is None:
            self._lv =  np.loadtxt(os.path.join(self.kd, "GNodeLib/LVNames.txt" ), dtype="|S100" )
        return self._lv
    lv = property(_get_lv)

    def select_gprim(self, names=False):
        pp = self.d.prims
        sli = slice(0,None)
        gprim = []
        for p in pp[sli]:
            if p.lvIdx in [8,9]: continue   # too many  of these LV
            if p.numParts > 1: continue     # skip compounds for now
            gprim.append(p)
            #print(repr(p)) 
            #print(str(p)) 
            if names:
                vol = p.idx[0]
                pv = self.pv[vol]
                lv = self.lv[vol]
                print(pv)
                print(lv)
            pass
        pass
        self.gprim = gprim 


    def dump(self):
        for i,p in enumerate(self.gprim):
            assert len(p.parts) == 1 
            pt = p.parts[0]
            print(repr(p)) 
            #print(str(p))
            #print(pt) 
            #print(pt.tran) 
 
    def render(self, ax):   
        sc = 1000
        for i,p in enumerate(self.gprim):
            assert len(p.parts) == 1 
            pt = p.parts[0]
            sh = pt.as_shape("prim%s" % i, sc=sc ) 
            if sh is None: 
               print(str(p))
               continue
            #print(sh)
            for pa in sh.patches():
                ax.add_patch(pa)
            pass
        pass





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
    log.info(kd)
    assert os.path.exists(kd), kd 
    os.environ["IDPATH"] = kd    ## TODO: avoid having to do this, due to prim internals

    mm0 = Geom2d(kd, ridx=0)

    import matplotlib.pyplot as plt 

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    ax = fig.add_subplot(111)
    plt.title("mm0 geom2d")
    sz = 50 
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    mm0.render(ax)






