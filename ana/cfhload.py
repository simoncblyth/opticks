#!/usr/bin/env python
"""
"""

import os, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH
log = logging.getLogger(__name__)

if __name__ == '__main__':
    ok = opticks_main(tag="1", src="torch", det="concentric")

    ctx = {'det':ok.det, 'tag':ok.tag }

    tagd = CFH.tagdir_(ctx)

    seq0s = os.listdir(tagd)

    seq0 = seq0s[0] 
  
    irec = len(seq0.split("_")) - 1

    ctx.update({'seq0':seq0, 'irec':str(irec) })

    log.info(" ctx %r " % ctx )

    for q in ok.qwn.replace(",",""):

        ctx.update({'qwn':q })

        h = CFH.load_(ctx)

        print h



