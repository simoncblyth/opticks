#!/usr/bin/env python
"""
g4gun.py: loads G4Gun event
===============================

To create the event use::

   ggv-
   ggv-g4gun


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)


from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

if __name__ == '__main__':

    args = opticks_main(src="G4Gun", det="G4Gun", tag="-1")

    try:
        evt = Evt(tag=args.tag, src=args.src, det=args.det)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc) 
    pass

    print evt




   

