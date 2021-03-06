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
Progressive sequencing, ie looking at the 
frequencies of steps as they develop as obtained
by a step by step growing mask.

This is essentially just the same as SeqAna but with 
a progressive mask.


"""
import logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nload import A
from opticks.ana.nbase import count_unique_sorted

cusfmt_ = lambda cus:"\n".join(["%16x  %8d " % (q, n) for q, n in cus])
msk_ = lambda n:(1 << 4*(n+1)) - 1  # msk_(0)=0xf msk_(1)=0xff msk_(2)=0xfff  

if __name__ == '__main__':
    args = opticks_main(doc=__doc__, tag="1", src="torch", det="laser", c2max=2.0, tagoffset=0)
    np.set_printoptions(precision=4, linewidth=200, formatter={'int':hex})

    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    dbg = False
    ph = A.load_("ph",args.src,args.utag,args.det,dbg, optional=True)

    seqhis = ph[:,0,0]


    for i in range(10):
        msk = msk_(i)
        sqh = seqhis & msk
        isqh = count_unique_sorted(sqh)
        print "%16x --------------- " % msk
        print cusfmt_(isqh)




