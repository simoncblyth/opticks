#!/usr/bin/env python

import os, sys, logging, numpy as np
from opticks.ana.ab import AB 

log = logging.getLogger(__name__)

if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()


    ## TODO: propagate special things regarding the event via metadata, not manually  
    wavelengths = "360,380,400,420,440,460,480".split(",")

    for i in range(7):
        wavelength = wavelengths[i]
        print(" input_photon start wavelength %s " % wavelength)
        itag = i + 1  
        ab = AB(ok, str(itag))  
        ab.seqhis_splits()
    pass

