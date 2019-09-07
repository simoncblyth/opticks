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
tevt.py : Loads a single event
===================================================

Expected output below shows the dimensions of the constitent numpy arrays that comprise the event::

    Evt(-4,"torch","PmtInBox","PmtInBox/torch/-4 : ", seqs="[]")
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :       (100000, 4, 4) : (photons) final photon step 
       wl :            (100000,) : (photons) wavelength 
     post :          (100000, 4) : (photons) final photon step: position, time 
     dirw :          (100000, 4) : (photons) final photon step: direction, weight  
     polw :          (100000, 4) : (photons) final photon step: polarization, wavelength  
    flags :            (100000,) : (photons) final photon step: flags  
       c4 :            (100000,) : (photons) final photon step: dtype split uint8 view of ox flags 
    rx_raw :   (100000, 10, 2, 4) : (records) photon step records RAW:before reshaping 
       rx :   (100000, 10, 2, 4) : (records) photon step records 
       ph :       (100000, 1, 2) : (records) photon history flag/material sequence 
       ps :       (100000, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 30) 
       rs :   (100000, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 30) 
      rsr :   (100000, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 30) 


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

if __name__ == '__main__':
    args = opticks_main(tag="10",src="torch", det="PmtInBox", doc=__doc__)
    np.set_printoptions(suppress=True, precision=3)

    for utag in args.utags:
        try:
            evt = Evt(tag=utag, src=args.src, det=args.det, seqs=[], args=args)
        except IOError as err:
            log.fatal(err)
            sys.exit(args.mrc)

        log.debug("evt") 
        print evt
        log.debug("evt.history_table") 
        evt.history_table(slice(0,20))
        log.debug("evt.history_table DONE") 
       

