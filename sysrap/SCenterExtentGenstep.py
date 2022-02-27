#!/usr/bin/env python
import numpy as np
from opticks.ana.fold import Fold

X,Y,Z,W=0,1,2,3

class SCenterExtentGenstep(object):

    @classmethod
    def Lim(cls, xyz):
        lim = np.zeros( (3,2), dtype=np.float32 )
        for A in [X,Y,Z]: lim[A] = [xyz[:,A].min(), xyz[:,A].max() ]
        return lim 


    def __init__(self, path):
        if path is None:
            path = "/tmp/$USER/opticks/SCenterExtentGenstep"
        pass

        fold = Fold.Load(path)

        photons = fold.photons
        gs = fold.gs

        self.fold = fold
        self.peta = fold.peta
        self.gs = gs
        self.photons = photons
        self.plim = self.Lim( photons[:,0] )
        self.glim = self.Lim( gs[:,5] )
   

if __name__ == '__main__':


    #path = os.path.expandvars("$CFBASE/CSGIntersectSolidTest/$GEOM")
    path = None

    cegs = SCenterExtentGenstep(path)

    peta = cegs.peta
    gs = cegs.gs 
    photons = cegs.photons

    print("cegs.plim\n", cegs.plim)
    print("cegs.glim\n", cegs.glim)



 
