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
tmeta.py : Load a single events metadata
===================================================

::

    simon:ana blyth$ tmeta.py --det dayabay --src torch --tag 1
    /Users/blyth/opticks/ana/tmeta.py --det dayabay --src torch --tag 1
    writing opticks environment to /tmp/blyth/opticks/opticks_env.bash 
    [2016-08-19 15:51:42,378] p23598 {/Users/blyth/opticks/ana/tmeta.py:21} INFO - loaded metadata from /tmp/blyth/opticks/evt/dayabay/torch/1 :                       /tmp/blyth/opticks/evt/dayabay/torch/1 571d76cd06acc1e992c211d6833dd0ff a32520a5215239cf54ee03d61ed154f6  100000     4.2878 CFG4_MODE  
              photonData : 571d76cd06acc1e992c211d6833dd0ff 
             NumGensteps : 1 
              NumRecords : 1000000 
              NumG4Event : 10 
               TimeStamp : 20160819_143439 
               BounceMax : 9 
                    UDet : dayabay 
              recordData : a32520a5215239cf54ee03d61ed154f6 
                     Cat :  
               RecordMax : 10 
                 cmdline :  
                     Tag : 1 
                    mode : CFG4_MODE 
              NumPhotons : 100000 
                Detector : dayabay 
           genstepDigest : 871b90ef849f4c13ef95fa4015d1f210 
                    Type : torch 
                  RngMax : 3000000 
            sequenceData : 5964de1100b8d5784865b8e9e39dca34 
    NumPhotonsPerG4Event : 10000 
      configureGenerator : 0.064189000000624219 
         indexPhotonsCPU : 0.089951999998447718 
                    save : 0.14543499999854248 
               configure : 4.0000013541430235e-06 
                   _save : 3.9000002288958058e-05 
       configureStepping : 4.7999998059822246e-05 
               propagate : 4.2877840000001015 
              _propagate : 4.6000001020729542e-05 
              initialize : 0.008572999999159947 
        configurePhysics : 0.072821000001567882 
          postinitialize : 0.00013499999840860255 
       configureDetector : 0.3900549999998475 



"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nload import tagdir_ 
from opticks.ana.metadata import Metadata

if __name__ == '__main__':
    args = opticks_main(tag="10",src="torch", det="PmtInBox", doc=__doc__)
    np.set_printoptions(suppress=True, precision=3)

    mdir = tagdir_(args.det, args.src, args.tag)
    md = Metadata(mdir)
    log.info("loaded metadata from %s : %s " % (mdir, repr(md)))
    md.dump()

  

