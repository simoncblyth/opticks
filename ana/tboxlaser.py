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
tboxlaser.py 
=============================================

Loads test events from Opticks and Geant4 and 
created by OKG4Test and 
compares their bounce histories.

"""
import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(doc=__doc__, tag="1", src="torch", det="boxlaser", c2max=2.0, tagoffset=0)

    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    a_seqs = []
    b_seqs = []

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=a_seqs, args=args)
        b = Evt(tag="-%s" % args.utag , src=args.src, det=args.det, seqs=b_seqs, args=args)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    print "A",a
    print "B",b

    log.info( " a : %s " % a.brief)
    log.info( " b : %s " % b.brief )

    tables = ["seqhis_ana"] + ["seqhis_ana_%d" % imsk for imsk in range(1,8)] + ["seqmat_ana"] 
    Evt.compare_table(a,b, tables, lmx=20, c2max=None, cf=True)

    Evt.compare_table(a,b, "pflags_ana hflags_ana".split(), lmx=20, c2max=None, cf=True)



