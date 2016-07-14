#!/usr/bin/env python
"""
pmt_test_evt.py : Loads a single PmtInBox event
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
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment() 

    evt = Evt(tag="-4", src="torch", det="PmtInBox", seqs=[])
    print evt




   

