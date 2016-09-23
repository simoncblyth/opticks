#!/usr/bin/env python
"""
hismask.py: HisMask
========================


"""
import os, datetime, logging, sys
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import opticks_main
from opticks.ana.base import Abbrev, EnumFlags
from opticks.ana.seq import MaskType, SeqTable, SeqAna
from opticks.ana.nbase import count_unique_sorted
from opticks.ana.nload import A


class HisMask(MaskType):
    def __init__(self):
        log.debug("HisMask.__init__")
        flags = EnumFlags()
        abbrev = Abbrev()
        MaskType.__init__(self, flags, abbrev)
        log.debug("HisMask.__init__ DONE")


def test_HisMask(af):
     label = "TO BT SD"
     mask = af.code(label)
     label2 = af.label(mask)
     log.info( " %30s -> %d -> %10s " % (label, mask, label2 ))

def test_HisMask_SeqTable(aa, af):
     hflags = aa[:,3,3].view(np.uint32)
     cu = count_unique_sorted(hflags)
     st = SeqTable(cu, af)
     print st 

def test_HisMask_SeqAna(aa, af):
     hflags = aa[:,3,3].view(np.uint32)
     sa = SeqAna(hflags, af)
     print sa.table 


if __name__ == '__main__':
     args = opticks_main(src="torch", tag="10", det="PmtInBox")

     af = HisMask()
     test_HisMask(af)

     try:
         ht = A.load_("ht",args.src,args.tag,args.det)
         log.info("loaded ht %s %s shape %s " %  (ht.path, ht.stamp, repr(ht.shape)))
         #test_HisMask_SeqTable(ht, af)
         test_HisMask_SeqAna(ht, af)
     except IOError as err:
         log.warning("no ht")

     try:
         ox = A.load_("ox",args.src,args.tag,args.det)
         log.info("loaded ox %s %s shape %s " %  (ox.path, ox.stamp, repr(ox.shape)))
         #test_HisMask_SeqTable(ox, af)
         test_HisMask_SeqAna(ox, af)
     except IOError as err:
         log.warning("no ht")


    



