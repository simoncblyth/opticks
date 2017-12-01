#!/usr/bin/env python
"""
histype.py: HisType
========================

::

    histype.py --det PmtInBox --tag 10 --src torch 
    histype.py --det dayabay  --tag 1  --src torch 

"""
import os, datetime, logging, sys
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import opticks_main
from opticks.ana.base import Abbrev, IniFlags
from opticks.ana.seq import SeqType, SeqTable, SeqAna
from opticks.ana.nbase import count_unique_sorted
from opticks.ana.nload import A

def test_HistoryTable(ht, seqhis):
     log.info("test_HistoryTable")  
     for seq in ht.labels:
         seqs = [seq]
         s_seqhis = map(lambda _:seqhis == af.code(_), seqs )
         psel = np.logical_or.reduce(s_seqhis)      

         n = len(seqhis[psel])
         assert n == ht.label2count.get(seq)
         print "%10s %s " % (n, seq ) 
     pass
     log.info("test_HistoryTable DONE")  

def test_roundtrip(af):
     x=0x8cbbbcd
     l = af.label(x)
     c = af.code(l)
     print "%x %s %x " % ( x,l,c )
     assert x == c 


class HisType(SeqType):
    def __init__(self):
        flags = IniFlags()
        abbrev = Abbrev("$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json")
        SeqType.__init__(self, flags, abbrev)



if __name__ == '__main__':
     args = opticks_main(src="torch", tag="10", det="PmtInBox")

     af = HisType()

     try:
         ph = A.load_("ph",args.src,args.tag,args.det)
     except IOError as err:
         log.fatal(err) 
         sys.exit(args.mrc)

     log.info("loaded ph %s %s shape %s " %  (ph.path, ph.stamp, repr(ph.shape)))

     seqhis = ph[:,0,0]

     cu = count_unique_sorted(seqhis)

     ht = SeqTable(cu, af, smry=True)
     
     test_HistoryTable(ht, seqhis)

     test_roundtrip(af)







