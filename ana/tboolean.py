#!/usr/bin/env python
"""
tboolean.py 
=============================================

This is invoked by Opticks bi-simulation executables such as OKG4Test 
when using the option  "--anakey tboolean".  See optickscore/OpticksAna.cc.
It compares Opticks and G4 event history categories and deviations.


"""
from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.ana.nload import np_load
from opticks.ana.ab   import AB
from opticks.ana.seq import seq2msk







if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  

    log.info(ok.brief)

    ab = AB(ok)
    ab.dump()

    rc = ab.RC

    level = "fatal" if rc > 0 else "info"
    getattr(log, level)(" RC 0x%.2x %s " % (rc,bin(rc)) )

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(rc)
    else:
        pass
    pass

    a = ab.a
    b = ab.b

    if ab.is_comparable:
        #ab.aselhis = "TO BT BT SA"     # dev aligned comparisons
        ab.aselhis = None    # dev aligned comparisons

        ab.check_utaildebug()

    pass
   
       

