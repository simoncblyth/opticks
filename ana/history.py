#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import Abbrev, IniFlags
from opticks.ana.seq import SeqType, SeqTable, SeqAna
from opticks.ana.nbase import count_unique_sorted
from opticks.ana.nload import A

def test_HistoryTable(ht, seqhis):
     for seq in ht.labels:
         seqs = [seq]
         s_seqhis = map(lambda _:seqhis == af.code(_), seqs )
         psel = np.logical_or.reduce(s_seqhis)      

         n = len(seqhis[psel])
         assert n == ht.label2count.get(seq)
         print "%10s %s " % (n, seq ) 

def test_roundtrip(af):
     x=0x8cbbbcd
     l = af.label(x)
     c = af.code(l)
     print "%x %s %x " % ( x,l,c )
     assert x == c 


class HisType(SeqType):
    def __init__(self):
        flags = IniFlags()
        abbrev = Abbrev()
        SeqType.__init__(self, flags, abbrev)



if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     af = HisType()

     #src, tag, det = "torch", "5", "rainbow"
     #src, tag, det = "cerenkov", "1", "juno"
     src, tag, det = "torch", "4", "PmtInBox"

     ph = A.load_("ph"+src,tag,det)

     seqhis = ph[:,0,0]

     cu = count_unique_sorted(seqhis)

     ht = SeqTable(cu, af)
     
     test_HistoryTable(ht, seqhis)

     test_roundtrip(af)


