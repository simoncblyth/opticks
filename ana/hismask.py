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
hismask.py: HisMask
========================

::

    In [12]: from opticks.ana.hismask import HisMask
    In [13]: hm = HisMask()
    In [16]: hm.label(2114)
    Out[16]: 'BT|SD|SI'





"""
import os, datetime, logging, sys
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import PhotonMaskFlags
from opticks.ana.seq import MaskType, SeqTable, SeqAna
from opticks.ana.nbase import count_unique_sorted
from opticks.ana.nload import A


class HisMask(MaskType):
    """ 
    """ 
    def __init__(self):
        log.debug("HisMask.__init__")
        flags = PhotonMaskFlags()
        MaskType.__init__(self, flags, flags.abbrev)
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
     print(st) 

def test_HisMask_SeqAna(aa, af):
     hflags = aa[:,3,3].view(np.uint32)
     sa = SeqAna(hflags, af)
     print(sa.table) 


if __name__ == '__main__':
     from opticks.ana.main import opticks_main
     #ok = opticks_main(src="torch", tag="10", det="PmtInBox")
     ok = opticks_main()

     af = HisMask()
     test_HisMask(af)

     try:
         ht = A.load_("ht",ok.src,ok.tag,ok.det, pfx=ok.pfx)
         log.info("loaded ht %s %s shape %s " %  (ht.path, ht.stamp, repr(ht.shape)))
         #test_HisMask_SeqTable(ht, af)
         test_HisMask_SeqAna(ht, af)
     except IOError as err:
         log.warning("no ht")

     try:
         ox = A.load_("ox",ok.src,ok.tag,ok.det, pfx=ok.pfx)
         log.info("loaded ox %s %s shape %s " %  (ox.path, ox.stamp, repr(ox.shape)))
         #test_HisMask_SeqTable(ox, af)
         test_HisMask_SeqAna(ox, af)
     except IOError as err:
         log.warning("no ht")


    



