#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
histype.py: HisType
========================

::


    LV=box histype.py 

    histype.py --det PmtInBox --tag 10 --src torch 
    histype.py --det dayabay  --tag 1  --src torch 


::

    export OPTICKS_ANA_DEFAULTS="det=tboolean-box,src=torch,tag=1,pfx=."

    cd /tmp
    OPTICKS_EVENT_BASE=tboolean-box histype.py

::


    In [21]: from opticks.ana.histype import HisType

    In [22]: histype = HisType()

    In [25]: histype.code("TO BT AB")
    Out[25]: 1229

    In [26]: ab.a.seqhis
    Out[26]: 
    A()sliced
    A([36045, 36045,  2237, ..., 36045, 36045, 36045], dtype=uint64)

    In [27]: ab.a.seqhis.shape
    Out[27]: (100000,)

    In [29]: np.where(ab.a.seqhis == histype.code("TO BT AB"))[0]
    Out[29]: array([ 2084,  4074, 15299, 20870, 25748, 26317, 43525, 51563, 57355, 61602, 65894, 71978, 77062, 78744, 79117, 86814])

    In [30]: np.where(ab.a.seqhis == histype.code("TO BT AB"))[0].shape
    Out[30]: (16,)


"""
import os, datetime, logging, sys
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import PhotonCodeFlags
from opticks.ana.seq import SeqType, SeqTable, SeqAna
from opticks.ana.nbase import count_unique_sorted
from opticks.ana.nload import A

class HisType(SeqType):
    def __init__(self):
        flags = PhotonCodeFlags() 
        SeqType.__init__(self, flags, flags.abbrev)



def test_HistoryTable(ht, seqhis):
     log.info("[")  

     #print("ht.labels",ht.labels)

     for seq in ht.labels:
         seqs = [seq]
         s_seqhis = list(map(lambda _:seqhis == af.code(_), seqs ))
         psel = np.logical_or.reduce(s_seqhis)      

         n = len(seqhis[psel])
         assert n == ht.label2count.get(seq)
         print("%10s %s " % (n, seq )) 
     pass
     log.info("]")  

def test_roundtrip(af):
     log.info("[")
     x=0x8cbbbcd
     l = af.label(x)
     c = af.code(l)
     print("%x %s %x " % ( x,l,c ))
     assert x == c 
     log.info("]")


def test_load_SeqTable(ok, af):
     try:
         ph = A.load_("ph",ok.src,ok.tag,ok.det, pfx=ok.pfx)
     except IOError as err:
         log.fatal(err) 
         sys.exit(0)

     log.info("loaded ph %s %s shape %s " %  (ph.path, ph.stamp, repr(ph.shape)))

     seqhis = ph[:,0,0]

     cu = count_unique_sorted(seqhis)

     ht = SeqTable(cu, af, smry=True)
      
     
     test_HistoryTable(ht, seqhis)



def test_HisType(af):
    log.info("[")
    print(af)
    print("af.abbrev.name2abbr",af.abbrev.name2abbr)
    print("af.flags.names",af.flags.names)
    print("af.flags.codes",af.flags.codes)
    print("af.abbr2code",af.abbr2code)
    log.info("]")


if __name__ == '__main__':
     from opticks.ana.main import opticks_main
     ok = opticks_main()

     af = HisType()

     #test_HisType(af)

     #test_roundtrip(af)

     test_load_SeqTable(ok, af)







