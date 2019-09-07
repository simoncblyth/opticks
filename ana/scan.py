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
scan.py
============

scan.py is sibling to bench.py : aiming for a cleaner 
and more general approach to metadata presentation 
based on meta.py (rather than the old metadata.py)


"""
import os, re, logging, sys, argparse
from collections import OrderedDict as odict
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.datedfolder import DatedFolder
from opticks.ana.meta import Meta


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else "." 
    cmdline = " ".join([os.path.basename(sys.argv[0])]+sys.argv[1:])
    print(cmdline)

    dirs, dfolds, dtimes = DatedFolder.find(base)
    assert len(dfolds) == len(dtimes)
    print( "dirs : %d  dtimes : %d " % (len(dirs), len(dtimes) ))


    ## arrange into groups of runs with the same runstamp/datedfolder
    assert len(dfolds) == len(dtimes) 
    order = sorted(range(len(dfolds)), key=lambda i:dtimes[i])   ## sorted by run datetimes

    dump_ = lambda m,top:"\n".join(map(lambda kv:"  %30s : %s " % (kv[0],kv[1]),m.d[top].items() )) 

    photons_ = lambda m:m["parameters.NumPhotons"]

    q=odict()
    q["ok1"] = "OpticksEvent_launch.launch001"
    q["ok2"] = "DeltaTime.OPropagator::launch_0"
    q["ok3"] = "OpticksEvent_prelaunch.prelaunch000"
    q["ok4"] = "DeltaTime.OpSeeder::seedPhotonsFromGenstepsViaOptiX_0"
    q["g4"]  = "DeltaTime.CG4::propagate_0"

    for k,v in q.items():
        print(" %4s : %s " % (k,v)) 
    pass

        
    for i in order:
        df = dfolds[i] 
        dt = dtimes[i] 

        udirs = filter(lambda _:_.endswith(df),dirs)
        #print("\n".join(udirs))
        if len(udirs) == 0: continue

        mm = [Meta(p, base) for p in udirs]
        assert len(mm) == 2  

        tag0 = int(mm[0].parentfold)  
        tag1 = int(mm[1].parentfold)  

        ok = 0 if tag0 > 0 else 1 
        g4 = 1 if tag1 < 0 else 0
        assert ok ^ g4, ( ok, g4, tag0, tag1 )

        #print(mm[ok])
        #print(mm[g4])

        nn = list(set(map(photons_, mm)))
        assert len(nn) == 1
        n = nn[0]

        vq = odict()
        for k,v in q.items():
            vq[k] = float(mm[ok if k.startswith("ok") else g4][v])
        pass

        divi_ = lambda num,den:num/den if den != 0 else 0

        vq["g4/ok1"] = divi_(vq["g4"],vq["ok1"])
        vq["g4/ok2"] = divi_(vq["g4"],vq["ok2"])

        svq = " ok1:%(ok1)10.4f  ok2:%(ok2)10.4f  g4:%(g4)10.4f   g4/ok1:%(g4/ok1)10.1f  g4/ok2:%(g4/ok2)10.1f   ok3:%(ok3)10.4f ok4:%(ok4)10.4f  " % vq  

        print(" %s   tag0:%-3d tag1:%-3d  n:%-7d     %s     " % (df, tag0, tag1, n, svq  )  )

        #print(dump_(mm[0],"parameters"))
        #print(dump_(mm[1],"OpticksEvent_launch"))

        





