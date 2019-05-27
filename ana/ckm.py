#!/usr/bin/env python
"""
::

    In [14]: evt.rpost_(slice(0,5)).shape
    Out[14]: (500000, 5, 4)


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

if __name__ == '__main__':
    args = opticks_main(tag="1",src="natural", det="g4live", doc=__doc__)
    np.set_printoptions(suppress=True, precision=3)

    evt = Evt(tag=args.tag, src=args.src, det=args.det, seqs=[], args=args)

    log.debug("evt") 
    print evt

    log.debug("evt.history_table") 
    evt.history_table(slice(0,20))
    log.debug("evt.history_table DONE") 
       

