#!/usr/bin/env python
"""

::

    ip abprofile.py --cat cvd_1_rtx_0_1M --pfx scan-pf-0 --tag 0
         OKG4Test run  


"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.profile import Profile


class ABProfile(object):
    def __init__(self, adir, bdir=None):

        if bdir is None:  
            # assume OK vs G4 mode : ie profiles from a bi-simulation
            pdir = adir       
            self.ap = Profile(pdir, "ab.pro.ap", g4=False) 
            self.bp = Profile(pdir, "ab.pro.bp", g4=True ) 
        else:
            # treat arguments as two profile directories, assumed to be OK (not G4)
            self.ap = Profile(adir, "ab.pro.ap", g4=False) 
            self.bp = Profile(bdir, "ab.pro.bp", g4=False) 
        pass 
        valid = self.ap.valid and self.bp.valid 
        if valid:
            boa = self.bp.tim/self.ap.tim if self.ap.tim > 0 else -1  
        else:
            boa = -2 
        pass 
        self.boa = boa
  
    def brief(self): 
        return "      ap.tim %-10.4f     bp.tim %-10.4f      bp.tim/ap.tim %-10.4f    " % (self.ap.tim, self.bp.tim, self.boa )   

    def __repr__(self):
        return "\n".join(["ab.pro", self.brief()] + self.ap.lines() + self.bp.lines() )






if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    from opticks.ana.plot import init_rcParams
    import matplotlib.pyplot as plt
    init_rcParams(plt)

    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)


    op = ABProfile(ok.tagdir) 
    print(op)

    ap = op.ap
    bp = op.bp

    plt.plot( ap.t, ap.v, 'o' )
    plt.plot( bp.t, bp.v, 'o' )

    plt.axvline( ap.t[ap.idx[0]], c="b" )
    plt.axvline( ap.t[ap.idx[1]], c="b" )

    plt.axvline( bp.t[bp.idx[0]], c="r" )
    plt.axvline( bp.t[bp.idx[1]], c="r" )

    plt.ion()
    plt.show()





    log.info("tagdir: %s " % ok.tagdir)


