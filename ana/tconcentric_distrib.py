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
tconcentric_distrib.py 
=============================================

::

    tconcentric-d --noplot --sel 0:20   
    tconcentric-d --noplot --sel 0:40
    tconcentric-d --noplot --sel 0:100

    # make rst chi2 table for all records of first many seq lines, 
    #  (142 recline for 0:20, 295 for 0:40, 897 for 0:100)
    #
    # skipping the plotting makes this fast, allowing 
    # chi2 distrib comparisons to be made for many thousands
    # of distribs in seconds 897*8 = 7176 


Issue with hexchar irec, fail to write into a,b,c...::

    tconcentric-d --noplot --sel 11:12

    delta:ana blyth$ ll /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/
    total 0
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 9
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 8
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 7
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 6
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 5
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 4
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 3
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 2
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 1
    drwxr-xr-x  10 blyth  wheel  340 Nov 14 18:03 0



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
from opticks.ana.ab import AB 
from opticks.ana.cfplot import cfplot, qwns_plot, multiplot


if __name__ == '__main__':
    ok = opticks_main(sel="0:5")
    log.info(" ok %s " % repr(ok.brief))

    plt.ion()
    plt.close()

    try:
        ab = AB(ok)
    except IOError as err:
        log.fatal(err)
        sys.exit(ok.mrc)
    
    print ab

    log.info(" sel %r qwn %s qwns %s " % (ok.sel, ok.qwn, ok.qwns )) 

    st = ab.stats( ok.sel.start, ok.sel.stop, ok.qwn, rehist=ok.rehist )
    
    #print st 
    print st[st.chi2sel()]

    if ok.plot:
        multiplot(ok, ab, ok.sel.start, ok.sel.stop, ok.qwn )
    else:
        log.info("plotting skipped by --noplot option")
    pass

 
