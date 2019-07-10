#!/usr/bin/env python
"""

::

    ip tprofile.py 

::

    ip () 
    { 
        local py=${1:-dummy.py};
        shift;
        ipython --pdb $(which $py) -i $*
    }


"""
from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stamp_


class OpticksProfile(object):
    def __init__(self, ok):
        tfold = tagdir_( ok.det, ok.src, ok.tag, pfx=ok.pfx )  
        self.load(tfold)
        self.ok = ok 
        self.sli = slice(None)

    def load(self, tfold):
        
        path = os.path.join( tfold, "OpticksProfile.npy")   # quads 
        lpath = os.path.join( tfold, "OpticksProfileLabels.npy")  
        print("path:%s stamp:%s " % (path, stamp_(path) ))
        print("lpath:%s stamp:%s " % (lpath, stamp_(lpath) ))

        a = np.load(path)
        l = np.load(lpath)
        ll = list(l[:,0,:].view("|S64")[:,0])
        lll = np.array(ll)

        t,dt,v,dv = a[:,0,0], a[:,0,1], a[:,0,2], a[:,0,3]

        self.tfold = tfold 
        self.l = lll 

        self.a = a 

        self.t = t
        self.dt = dt
        self.v = v
        self.dv = dv

    def deltaVM(self, l0="_CG4::propagate", l1="CG4::propagate" ): 
        l = self.l  
        v = self.v
        p0 = np.where( l == l0 )[0][0]
        p1 = np.where( l == l1 )[0][0]
        v01 = v[p1] - v[p0]
        print(" l0:%s l1:%s p0:%d p1:%d v01:%f " % ( l0, l1, p0, p1, v01 )) 


    def line(self, i):
        return " %3d : %50s : %10.3f %10.3f %10.3f %10.3f   " % ( i, self.l[i], self.t[i], self.v[i], self.dt[i], self.dv[i] )

    def __repr__(self):
        return "\n".join(map(lambda i:self.line(i), np.arange(len(self.l))[self.sli]   ))

    def __getitem__(self, sli):
        self.sli = sli
        return self 



if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)

    op = OpticksProfile(ok) 
    op.deltaVM()

    a = op.a  
    l = op.l 

    plt.plot( op.t, op.v, 'o' )
    plt.ion()
    plt.show()

    print(op)


