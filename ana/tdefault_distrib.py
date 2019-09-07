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
tdefault_distrib.py 
=============================================


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 18,10.2   # plt.gcf().get_size_inches()   after maximize
    import matplotlib.gridspec as gridspec
except ImportError:
    print "matplotlib missing : you need this to make plots"
    plt = None 

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt
from opticks.ana.cf import CF 
from opticks.ana.cfplot import cfplot, qwns_plot, qwn_plot, multiplot


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    args = opticks_main(tag="1", src="torch", det="default")
    log.info(" args %s " % repr(args))
    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    plt.ion()
    plt.close()

    select = slice(1,2)
    #select = slice(0,8)

    try:
        cf = CF(tag=args.tag, src=args.src, det=args.det, select=select )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    cf.dump()

    irec = 1 

    #multiplot(cf, pages=["XYZT","ABCR"])
  
    #qwn_plot( cf.ss[0], "T", -1, c2_ymax=2000)
    #qwn_plot( cf, "R", irec)
    #qwns_plot( cf, "XYZT", irec)
    #qwns_plot( cf, "ABCR", irec)


    binsx,ax,bx,lx = cf.rqwn("X",irec)
    binsy,ay,by,ly = cf.rqwn("Y",irec)
    binsz,az,bz,lz = cf.rqwn("Z",irec)

   






 
