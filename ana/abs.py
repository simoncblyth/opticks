#!/usr/bin/env python
"""

::

    ## slot 1 counts using input photons at 7 wavelengths in OK and G4, for diametric diagonal paths (InwardsCubeCorners17999?)  
    ##
    ##      AB     RE      SC     BT    

    In [3]: counts.reshape(-1,4)                                                                                                                                                                      
    Out[3]: 
    array([[16040, 63928,    32,     0],
           [16012, 63966,    22,     0],
           [16019, 63919,    62,     0],
           [16055, 63885,    60,     0],
           [15843, 63243,   914,     0],
           [15917, 63167,   916,     0],
           [13237, 12934, 46479,  7350],
           [13268, 12948, 46507,  7277],
           [12882,  3636, 47795, 15687],
           [13027,  3709, 47669, 15595],
           [15538,  3168, 43614, 17680],
           [15417,  3278, 43348, 17957],
           [14691,  2319, 41745, 21245],
           [14695,  2264, 41777, 21264]], dtype=int32)

    In [4]: np.sum( counts.reshape(-1,4), axis=1 )                                                                                                                                                    
    Out[4]: array([80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000, 80000])


"""
import os, sys, logging, numpy as np
from opticks.ana.ab import AB 

log = logging.getLogger(__name__)

if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()

    ## TODO: propagate special things regarding the event via metadata, not manually  
    wavelengths = "360,380,400,420,440,460,480".split(",")

    nevt = len(wavelengths)

    slot = 1 
    labs = "AB RE SC BT" 
    nlab = len(labs.split())
    nab = 2 

    counts = np.zeros( (nevt, nab, nlab), dtype=np.int32 ) 
    for i in range(nevt):
        wavelength = wavelengths[i]
        itag = i + 1  
        name = "ab%d" % itag
        print("%s : input_photon start wavelength %s " % (name, wavelength) )
        ab = AB(ok, str(itag))  
        counts[i] = ab.seqhis_counts(slot, labs )
        globals()[name] = ab 
    pass

