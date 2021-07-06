#!/usr/bin/env python
"""

::

    LV=box abprofile.py 
    LV=box python2.7 abprofile.py 


    ip abprofile.py --cat cvd_1_rtx_0_1M --pfx scan-pf-0 --tag 0
         OKG4Test run  


"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.profile_ import Profile


class ABProfile(object):
    def __init__(self, adir, bdir ):
        assert not bdir is None

        anam = os.path.basename(adir)
        bnam = os.path.basename(bdir)
        ag4 = anam[0] == "-"
        bg4 = bnam[0] == "-"

        log.debug("adir %s anam %s ag4 %s" % (adir, anam, ag4)) 
        log.debug("bdir %s bnam %s bg4 %s" % (bdir, bnam, bg4)) 

        log.debug("[ ab.pro.ap")
        self.ap = Profile(adir, "ab.pro.ap", g4=ag4) 
        log.debug("]")
        log.debug("[ ab.pro.bp")
        self.bp = Profile(bdir, "ab.pro.bp", g4=bg4) 
        log.debug("]")

        valid = self.ap.valid and self.bp.valid 
        if valid:
            boa = self.bp.tim/self.ap.tim if self.ap.tim > 0 else -1  
            log.debug("self.bp.tim %s self.ap.tim %s boa %s" % (self.bp.tim, self.ap.tim, boa))
        else:
            boa = -2 
        pass 
        self.boa = boa

    def _get_missing(self):
        return self.ap.missing or self.bp.missing
    missing = property(_get_missing)
  
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

