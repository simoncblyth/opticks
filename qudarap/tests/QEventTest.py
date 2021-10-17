#!/usr/bin/env python
"""

::

    ipython -i tests/QEventTest.py 

"""

import os, numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None
pass




class QEventTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/qudarap/QEventTest")
    def __init__(self, fold=FOLD):
        names = os.listdir(fold)
        stems = []
        for name in filter(lambda n:n.endswith(".npy"), names):
            stem = name[:-4]
            path = os.path.join(fold, name)
            a = np.load(path)
            print(" %10s : %15s : %s " % (stem, str(a.shape), path ))
            globals()[stem] = a 
            setattr( self, stem, a )
            stems.append(stem)
        pass
        self.stems = stems
    def dump(self):
        for stem in self.stems:
            a = getattr(self, stem)
            print(stem)
            print(a)
            #print(a.view(np.int32)) 
        pass


def test_transform():
    dtype = np.float32
    p0 = np.array([1,2,3,1], dtype)
    p1 = np.zeros( (len(cegs), 4), dtype ) 
    p1x = cegs[:,1]

    # transform p0 by each of the genstep transforms 
    for i in range(len(cegs)): p1[i] = np.dot( p0, cegs[i,2:] )
    pos = p1[:,:3]   
    return pos 


if __name__ == '__main__':
    t = QEventTest()
    t.dump()

    pos = ppa[:,0,:3]
    dir = ppa[:,1,:3]
    #pv = None

    if not pv is None:
        size = np.array([1280, 720])
        pl = pv.Plotter(window_size=size*2 ) 
        pl.add_points( pos, color='#FF0000', point_size=10.0 ) 
        pl.add_arrows( pos, dir, mag=50, color='#00FF00', point_size=1.0 ) 
        pl.show_grid()
        cp = pl.show()    
    pass



