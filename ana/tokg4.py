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
tokg4.py 
=============================================

Loads test events from Opticks and Geant4 and 
created by OKG4Test and 
compares their bounce histories.

"""
import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.ana.ab   import AB
from opticks.ana.nbase import vnorm
from opticks.ana.seq import seq2msk



if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="g4live", pfx="source", tagoffset=0)
    #ok = opticks_main(doc=__doc__)  

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

    




if 0:
    if a.valid:
        a0 = a.rpost_(0)
        #a0r = np.linalg.norm(a0[:,:2],2,1)
        a0r = vnorm(a0[:,:2])
        if len(a0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (a0r.min(),a0r.max())))

    if b.valid:
        b0 = b.rpost_(0)
        #b0r = np.linalg.norm(b0[:,:2],2,1)
        b0r = vnorm(b0[:,:2])
        if len(b0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (b0r.min(),b0r.max())))


